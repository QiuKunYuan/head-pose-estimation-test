import os
import time
import logging
import argparse
import warnings

import cv2
import numpy as np
from collections import deque

import torch
import torch.nn as nn
from torchvision import transforms

# 假设这些是你项目中的模块
from models import get_model, SCRFD
from utils.general import compute_euler_angles_from_rotation_matrices, draw_cube, draw_axis

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation with facial landmarks and attention analysis.')
    parser.add_argument("--network", type=str, default="", help="Model name (auto-detected if empty)")
    parser.add_argument("--input", type=str, default='0', help="Path to input video file or camera id")
    parser.add_argument("--view", action="store_true", help="Display the inference results")
    parser.add_argument("--draw-type", type=str, default='cube', choices=['cube', 'axis'],
                        help="Draw cube or axis for head pose")
    parser.add_argument('--weights', type=str, required=True, help='Path to head pose estimation model weights')
    parser.add_argument("--output", type=str, default="", help="Path to save output file")

    # 专注度分析参数
    parser.add_argument("--attention", action="store_true", help="Enable attention analysis")
    parser.add_argument("--no-pose", action="store_true", help="Disable pose visualization")
    parser.add_argument("--landmarks", action="store_true", help="Enable facial landmarks detection")
    parser.add_argument("--detector-weights", type=str, default="./weights/det_10g.onnx",
                        help="Path to face detection model weights")

    # 眼部识别参数
    parser.add_argument("--eye-tracking", action="store_true", help="Enable eye tracking and blink detection")
    parser.add_argument("--gaze-estimation", action="store_true", help="Enable gaze direction estimation")

    # 性能优化参数
    parser.add_argument("--skip-frames", type=int, default=0, help="Skip n frames between processing")
    parser.add_argument("--scale-factor", type=float, default=1.0, help="Scale down frame for processing")
    parser.add_argument("--show-fps", action="store_true", help="Show FPS counter")

    return parser.parse_args()


def pre_process(image):
    """预处理图像用于头部姿态估计"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image_batch = image.unsqueeze(0)
    return image_batch


def expand_bbox(x_min, y_min, x_max, y_max, factor=0.2):
    """Expand the bounding box by a given factor."""
    width = x_max - x_min
    height = y_max - y_min

    x_min_new = x_min - int(factor * height)
    y_min_new = y_min - int(factor * width)
    x_max_new = x_max + int(factor * height)
    y_max_new = y_max + int(factor * width)

    return max(0, x_min_new), max(0, y_min_new), x_max_new, y_max_new


def detect_model_type(state_dict):
    """根据state_dict自动检测模型类型"""
    keys = list(state_dict.keys())

    if any('features.0.0.weight' in key for key in keys):
        return "mobilenetv2"
    elif any('conv1.weight' in key for key in keys):
        return "resnet18"
    else:
        return "resnet18"


def load_head_pose_model(weights_path, network_name="", device="cuda"):
    """加载头部姿态模型，自动检测模型类型"""
    try:
        state_dict = torch.load(weights_path, map_location=device)

        if not network_name:
            detected_network = detect_model_type(state_dict)
            logging.info(f"Auto-detected model type: {detected_network}")
            network_name = detected_network

        head_pose = get_model(network_name, num_classes=6, pretrained=False)

        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value

        head_pose.load_state_dict(new_state_dict, strict=False)
        head_pose.to(device)
        head_pose.eval()
        logging.info(f"Head Pose Estimation model ({network_name}) loaded successfully.")

        return head_pose

    except Exception as e:
        logging.error(f"Failed to load head pose model: {e}")
        raise


# 眼部识别类
class EyeAnalyzer:
    def __init__(self, ear_threshold=0.2, consecutive_frames=3):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.eye_aspect_ratios = deque(maxlen=consecutive_frames)
        self.blink_count = 0
        self.eyes_closed = False
        self.eyes_closed_start_time = None
        self.eyes_closed_duration = 0

    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """计算眼睛纵横比(EAR) - 用于眨眼检测"""
        # 提取6个眼部关键点坐标
        # 假设landmarks包含: [p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y, p6x, p6y]
        if len(eye_landmarks) < 12:
            return 0.3  # 默认值

        # 计算垂直距离
        A = np.linalg.norm(np.array(eye_landmarks[1:3]) - np.array(eye_landmarks[5:7]))
        B = np.linalg.norm(np.array(eye_landmarks[3:5]) - np.array(eye_landmarks[7:9]))

        # 计算水平距离
        C = np.linalg.norm(np.array(eye_landmarks[0:2]) - np.array(eye_landmarks[6:8]))

        # 计算EAR
        ear = (A + B) / (2.0 * C) if C != 0 else 0.3
        return ear

    def detect_blink(self, left_ear, right_ear):
        """检测眨眼"""
        avg_ear = (left_ear + right_ear) / 2.0
        self.eye_aspect_ratios.append(avg_ear)

        # 检测眨眼（EAR低于阈值）
        if len(self.eye_aspect_ratios) >= self.consecutive_frames:
            if all(ear < self.ear_threshold for ear in self.eye_aspect_ratios):
                if not self.eyes_closed:
                    self.eyes_closed = True
                    self.eyes_closed_start_time = time.time()
                    return True
            else:
                if self.eyes_closed:
                    self.eyes_closed = False
                    self.blink_count += 1
                    if self.eyes_closed_start_time:
                        self.eyes_closed_duration = time.time() - self.eyes_closed_start_time
        return False

    def get_eye_state(self, left_ear, right_ear):
        """获取眼睛状态"""
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < self.ear_threshold:
            return "Closed", (0, 0, 255)  # 红色
        elif avg_ear < self.ear_threshold + 0.1:
            return "Half-closed", (0, 165, 255)  # 橙色
        else:
            return "Open", (0, 255, 0)  # 绿色

    def estimate_gaze_direction(self, left_eye_landmarks, right_eye_landmarks):
        """估计视线方向（简化版）"""
        # 计算眼睛中心点
        left_eye_center = np.mean(np.array(left_eye_landmarks).reshape(-1, 2), axis=0)
        right_eye_center = np.mean(np.array(right_eye_landmarks).reshape(-1, 2), axis=0)

        # 计算眼睛水平位置差异
        eye_center_x = (left_eye_center[0] + right_eye_center[0]) / 2

        # 简化的视线方向估计
        if eye_center_x < 0.4:  # 向左看
            return "Looking Left", (255, 0, 0)
        elif eye_center_x > 0.6:  # 向右看
            return "Looking Right", (255, 0, 0)
        else:  # 向前看
            return "Looking Forward", (0, 255, 0)


# 面部关键点检测类
class FacialLandmarkDetector:
    def __init__(self, model_path):
        try:
            self.detector = SCRFD(model_path=model_path)
            logging.info("Facial landmark detector initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize facial landmark detector: {e}")
            raise

    def detect_landmarks(self, image):
        """检测人脸和关键点"""
        bboxes, landmarks = self.detector.detect(image)
        return bboxes, landmarks

    def extract_eye_landmarks(self, landmarks):
        """从面部关键点中提取眼部关键点"""
        # 假设SCRFD返回5个关键点：[左眼, 右眼, 鼻子, 左嘴角, 右嘴角]
        # 这里我们简化处理，实际可能需要更复杂的关键点提取
        left_eye = landmarks[0:2]  # 左眼中心
        right_eye = landmarks[2:4]  # 右眼中心

        # 为简化演示，我们生成模拟的眼部关键点
        # 实际应用中应该使用包含更多眼部关键点的检测器
        left_eye_landmarks = self._generate_eye_landmarks(left_eye)
        right_eye_landmarks = self._generate_eye_landmarks(right_eye)

        return left_eye_landmarks, right_eye_landmarks

    def _generate_eye_landmarks(self, eye_center):
        """生成模拟的眼部关键点（用于演示）"""
        cx, cy = eye_center
        # 生成6个眼部关键点（模拟）
        landmarks = [
            cx - 10, cy - 5,  # 左眼角
            cx, cy - 8,  # 上眼睑中点
            cx + 10, cy - 5,  # 右眼角
            cx + 10, cy + 5,  # 右下眼角
            cx, cy + 8,  # 下眼睑中点
            cx - 10, cy + 5  # 左下眼角
        ]
        return landmarks

    def draw_eye_details(self, image, left_eye_landmarks, right_eye_landmarks, eye_state, gaze_direction):
        """绘制眼部详细信息"""
        # 绘制左眼关键点
        for i in range(0, len(left_eye_landmarks), 2):
            x = int(left_eye_landmarks[i])
            y = int(left_eye_landmarks[i + 1])
            cv2.circle(image, (x, y), 2, (0, 255, 255), -1)

        # 绘制右眼关键点
        for i in range(0, len(right_eye_landmarks), 2):
            x = int(right_eye_landmarks[i])
            y = int(right_eye_landmarks[i + 1])
            cv2.circle(image, (x, y), 2, (0, 255, 255), -1)

        # 绘制眼部状态
        cv2.putText(image, f"Eyes: {eye_state[0]}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_state[1], 2)

        # 绘制视线方向
        cv2.putText(image, f"Gaze: {gaze_direction[0]}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, gaze_direction[1], 2)

        return image


# 性能监控类
class PerformanceMonitor:
    def __init__(self, window_size=30):
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()

    def update(self):
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.frame_times.append(frame_time)
        self.last_time = current_time
        return frame_time

    def get_fps(self):
        if not self.frame_times:
            return 0
        avg_frame_time = np.mean(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0


# 增强的专注度分析类
class EnhancedAttentionAnalyzer:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.pose_history = deque(maxlen=window_size)
        self.attention_history = deque(maxlen=window_size)
        self.student_data = []
        self.eye_analyzer = EyeAnalyzer()

    def calculate_attention_score(self, pitch, yaw, roll, eye_state, gaze_direction, blink_detected):
        """计算综合专注度得分，包含眼部特征"""
        # 1. 头部姿态评分 (40%)
        head_score = self._calculate_head_pose_score(pitch, yaw, roll)

        # 2. 眼部状态评分 (30%)
        eye_score = self._calculate_eye_score(eye_state, gaze_direction)

        # 3. 眨眼频率评分 (30%)
        blink_score = self._calculate_blink_score(blink_detected)

        # 综合得分
        total_score = head_score * 0.4 + eye_score * 0.3 + blink_score * 0.3

        # 状态判定
        if total_score >= 85:
            status = "Highly Focused"
            color = (0, 255, 0)
        elif total_score >= 70:
            status = "Focused"
            color = (0, 200, 100)
        elif total_score >= 50:
            status = "Moderate"
            color = (0, 255, 255)
        elif total_score >= 30:
            status = "Distracted"
            color = (0, 100, 255)
        else:
            status = "Very Distracted"
            color = (0, 0, 255)

        return status, int(total_score), color

    def _calculate_head_pose_score(self, pitch, yaw, roll):
        """计算头部姿态得分"""
        pitch_score = max(0, 100 - abs(pitch) * 1.5)
        yaw_score = max(0, 100 - abs(yaw) * 1.2)
        return (pitch_score + yaw_score) / 2

    def _calculate_eye_score(self, eye_state, gaze_direction):
        """计算眼部状态得分"""
        # 眼睛状态得分
        if eye_state[0] == "Open":
            eye_state_score = 100
        elif eye_state[0] == "Half-closed":
            eye_state_score = 60
        else:  # Closed
            eye_state_score = 20

        # 视线方向得分
        if gaze_direction[0] == "Looking Forward":
            gaze_score = 100
        else:
            gaze_score = 50

        return (eye_state_score + gaze_score) / 2

    def _calculate_blink_score(self, blink_detected):
        """计算眨眼频率得分"""
        # 简化的眨眼评分（正常眨眼是好的，频繁眨眼可能表示疲劳）
        if blink_detected:
            return 80  # 正常眨眼
        else:
            return 95  # 没有眨眼（可能更专注）


def draw_attention_info(frame, bbox, status, score, pitch, yaw, roll, color, eye_info=""):
    """在图像上绘制专注度信息"""
    x1, y1, x2, y2 = bbox

    # 绘制边界框
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # 绘制专注度信息
    text = f"{status}: {score}%"
    cv2.putText(frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 绘制眼部信息（如果有）
    if eye_info:
        cv2.putText(frame, eye_info, (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return frame


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 初始化人脸检测
    try:
        face_detector = FacialLandmarkDetector(model_path=params.detector_weights)
    except Exception as e:
        logging.error(f"Failed to initialize face detector: {e}")
        return

    # 初始化头部姿态估计模型
    try:
        head_pose = load_head_pose_model(params.weights, params.network, device)
    except Exception as e:
        logging.error(f"Failed to load head pose model: {e}")
        return

    # 初始化专注度分析器和眼部分析器
    attention_analyzer = None
    eye_analyzer = None

    if params.attention:
        attention_analyzer = EnhancedAttentionAnalyzer()
        eye_analyzer = EyeAnalyzer()
        logging.info("Attention analysis with eye tracking enabled.")

    # 初始化性能监控
    performance_monitor = PerformanceMonitor()

    # 初始化视频捕获
    video_source = params.input
    if video_source.isdigit() or video_source == '0':
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        logging.error("Cannot open video source")
        return

    # 获取视频属性
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化视频写入器
    out = None
    if params.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(params.output, fourcc, 20.0, (original_width, original_height))

    frame_count = 0

    logging.info("Starting inference with eye tracking... Press 'q' to quit")

    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1

            # 性能监控更新
            frame_time = performance_monitor.update()
            fps = performance_monitor.get_fps() if params.show_fps else None

            # 缩放帧用于处理
            if params.scale_factor < 1.0:
                process_width = int(original_width * params.scale_factor)
                process_height = int(original_height * params.scale_factor)
                process_frame = cv2.resize(frame, (process_width, process_height))
            else:
                process_frame = frame.copy()

            # 人脸检测和关键点检测
            bboxes, landmarks = face_detector.detect_landmarks(process_frame)

            for bbox, landmark in zip(bboxes, landmarks):
                # 转换坐标回原始分辨率
                if params.scale_factor < 1.0:
                    scale_x = original_width / process_width
                    scale_y = original_height / process_height
                    x_min = int(bbox[0] * scale_x)
                    y_min = int(bbox[1] * scale_y)
                    x_max = int(bbox[2] * scale_x)
                    y_max = int(bbox[3] * scale_y)
                else:
                    x_min, y_min, x_max, y_max = map(int, bbox[:4])

                # 扩展边界框
                x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max)

                # 确保裁剪区域有效
                if x_min >= x_max or y_min >= y_max:
                    continue

                # 头部姿态估计
                try:
                    face_roi = frame[y_min:y_max, x_min:x_max]
                    if face_roi.size == 0:
                        continue

                    image = pre_process(face_roi)
                    image = image.to(device)

                    rotation_matrix = head_pose(image)
                    euler = np.degrees(compute_euler_angles_from_rotation_matrices(rotation_matrix))
                    p_pred_deg = euler[:, 0].cpu().numpy()[0]
                    y_pred_deg = euler[:, 1].cpu().numpy()[0]
                    r_pred_deg = euler[:, 2].cpu().numpy()[0]

                    # 眼部识别和分析
                    eye_state = ("Unknown", (255, 255, 255))
                    gaze_direction = ("Unknown", (255, 255, 255))
                    blink_detected = False

                    if params.eye_tracking and eye_analyzer is not None:
                        # 提取眼部关键点
                        left_eye_landmarks, right_eye_landmarks = face_detector.extract_eye_landmarks(landmark)

                        # 计算眼睛纵横比
                        left_ear = eye_analyzer.calculate_eye_aspect_ratio(left_eye_landmarks)
                        right_ear = eye_analyzer.calculate_eye_aspect_ratio(right_eye_landmarks)

                        # 检测眨眼
                        blink_detected = eye_analyzer.detect_blink(left_ear, right_ear)

                        # 获取眼睛状态
                        eye_state = eye_analyzer.get_eye_state(left_ear, right_ear)

                        # 估计视线方向
                        if params.gaze_estimation:
                            gaze_direction = eye_analyzer.estimate_gaze_direction(left_eye_landmarks,
                                                                                  right_eye_landmarks)

                        # 绘制眼部详细信息
                        if params.landmarks:
                            frame = face_detector.draw_eye_details(frame, left_eye_landmarks, right_eye_landmarks,
                                                                   eye_state, gaze_direction)

                    # 专注度分析
                    if params.attention and attention_analyzer is not None:
                        status, score, color = attention_analyzer.calculate_attention_score(
                            p_pred_deg, y_pred_deg, r_pred_deg, eye_state, gaze_direction, blink_detected
                        )

                        # 绘制专注度信息
                        bbox_rect = [x_min, y_min, x_max, y_max]
                        eye_info = f"Eyes: {eye_state[0]}, Gaze: {gaze_direction[0]}"
                        frame = draw_attention_info(
                            frame, bbox_rect, status, score,
                            p_pred_deg, y_pred_deg, r_pred_deg, color, eye_info
                        )

                    # 头部姿态可视化
                    if not params.no_pose:
                        width = x_max - x_min
                        if params.draw_type == "cube":
                            draw_cube(
                                frame,
                                np.array([y_pred_deg]),
                                np.array([p_pred_deg]),
                                np.array([r_pred_deg]),
                                bbox=[x_min, y_min, x_max, y_max],
                                size=width
                            )
                        else:
                            draw_axis(
                                frame,
                                np.array([y_pred_deg]),
                                np.array([p_pred_deg]),
                                np.array([r_pred_deg]),
                                bbox=[x_min, y_min, x_max, y_max],
                                size_ratio=0.5
                            )

                except Exception as e:
                    continue

            # 显示统计信息
            if params.show_fps and fps:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if eye_analyzer and params.eye_tracking:
                cv2.putText(frame, f"Blinks: {eye_analyzer.blink_count}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if params.view:
                cv2.imshow('Student Attention Analysis with Eye Tracking', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            # 写入视频文件
            if out is not None:
                out.write(frame)

    # 资源清理
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    # 输出眼部统计
    if eye_analyzer:
        logging.info(f"Total blinks detected: {eye_analyzer.blink_count}")

    logging.info("Processing completed.")


if __name__ == '__main__':
    args = parse_args()
    main(args)
