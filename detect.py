import os
import time
import logging
import argparse
import warnings

import cv2
import numpy as np
from collections import deque

import torch
from torchvision import transforms

from models import get_model, SCRFD
from utils.general import compute_euler_angles_from_rotation_matrices, draw_cube, draw_axis

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation inference.')
    parser.add_argument("--network", type=str, default="resnet18", help="Model name, default `resnet18`")
    parser.add_argument(
        "--input",
        type=str,
        default='assets/in_video.mp4',
        help="Path to input video file or camera id"
    )
    parser.add_argument("--view", action="store_true", help="Display the inference results")
    parser.add_argument(
        "--draw-type",
        type=str,
        default='cube',
        choices=['cube', 'axis'],
        help="Draw cube or axis for head pose"
    )
    parser.add_argument('--weights', type=str, required=True, help='Path to head pose estimation model weights')
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output file")
    # 新增参数
    parser.add_argument("--attention", action="store_true", help="Enable attention analysis")
    parser.add_argument("--no-pose", action="store_true", help="Disable pose visualization (only show attention)")
    # 性能优化参数
    parser.add_argument("--skip-frames", type=int, default=1, help="Skip n frames between processing (0=no skip)")
    parser.add_argument("--scale-factor", type=float, default=1.0,
                        help="Scale down frame for processing (0.5=half size)")
    parser.add_argument("--show-fps", action="store_true", help="Show FPS counter")

    return parser.parse_args()


def pre_process(image):
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


# 专注度分析类
class AttentionAnalyzer:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.pose_history = deque(maxlen=window_size)
        self.student_data = []

    def calculate_attention_score(self, pitch, yaw, roll):
        """计算专注度得分 (0-100)"""
        # 正常听课角度范围
        pitch_score = max(0, 100 - abs(pitch) * 2) if abs(pitch) > 10 else 100
        yaw_score = max(0, 100 - abs(yaw) * 1.5) if abs(yaw) > 15 else 100

        # 综合得分
        total_score = (pitch_score * 0.6 + yaw_score * 0.4)

        # 状态判定
        if total_score >= 80:
            status = "Focused"
            color = (0, 255, 0)  # 绿色
        elif total_score >= 60:
            status = "Normal"
            color = (0, 255, 255)  # 黄色
        else:
            status = "Distracted"
            color = (0, 0, 255)  # 红色

        return status, int(total_score), color

    def update_student_data(self, bbox, pitch, yaw, roll, status, score):
        """更新学生数据"""
        self.student_data.append({
            'bbox': bbox,
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'status': status,
            'score': score,
            'timestamp': time.time()
        })

    def get_class_statistics(self):
        """获取课堂统计信息"""
        if not self.student_data:
            return None

        scores = [data['score'] for data in self.student_data]
        statuses = [data['status'] for data in self.student_data]

        # 修复：使用正确的英文状态名称
        focus_count = statuses.count('Focused')
        normal_count = statuses.count('Normal')
        distracted_count = statuses.count('Distracted')

        return {
            'avg_score': np.mean(scores),
            'focus_count': focus_count,
            'normal_count': normal_count,
            'distracted_count': distracted_count,
            'total_frames': len(self.student_data)
        }


def draw_attention_info(frame, bbox, status, score, pitch, yaw, roll, color):
    """在图像上绘制专注度信息"""
    x1, y1, x2, y2 = bbox

    # 绘制边界框
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # 绘制专注度信息
    text = f"{status}: {score}%"
    cv2.putText(frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # # 绘制姿态角度
    # pose_text = f"P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f}"
    # cv2.putText(frame, pose_text, (x1, y1 - 25),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    return frame


def draw_class_statistics(frame, statistics, student_count, fps=None):
    """在图像上绘制课堂统计信息"""
    # 统计信息背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 160), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # 绘制统计信息
    cv2.putText(frame, f"Students: {student_count}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if statistics:
        cv2.putText(frame, f"Avg Score: {statistics['avg_score']:.1f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Focused: {statistics['focus_count']}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, f"Distracted: {statistics['distracted_count']}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logging.info(f"GPU: {gpu_name}")
        logging.info(f"GPU Memory: {gpu_memory:.1f} GB")

        # GPU 优化设置
        torch.backends.cudnn.benchmark = True


    try:
        face_detector = SCRFD(model_path="./weights/det_10g.onnx")
        logging.info("Face Detection model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occured while loading pre-trained weights of face detection model. Exception: {e}")

    try:
        head_pose = get_model(params.network, num_classes=6, pretrained=False)
        state_dict = torch.load(params.weights, map_location=device)
        head_pose.load_state_dict(state_dict)

        # 🚀 模型移到 GPU
        head_pose.to(device)
        head_pose.eval()

        logging.info("Head Pose Estimation model weights loaded.")
    except Exception as e:
        logging.info(
            f"Exception occured while loading pre-trained weights of head pose estimation model. Exception: {e}")



    # 初始化专注度分析器
    attention_analyzer = None
    if params.attention:
        attention_analyzer = AttentionAnalyzer()
        logging.info("Attention analysis enabled.")

    # 初始化性能监控
    performance_monitor = PerformanceMonitor()

    # Initialize video capture
    video_source = params.input
    if video_source.isdigit() or video_source == '0':
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # 获取原始视频属性
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # 计算处理分辨率
    if params.scale_factor < 1.0:
        process_width = int(original_width * params.scale_factor)
        process_height = int(original_height * params.scale_factor)
        logging.info(f"Processing resolution: {process_width}x{process_height} (scale: {params.scale_factor})")
    else:
        process_width = original_width
        process_height = original_height

    # Initialize VideoWriter if saving video
    out = None
    if params.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(params.output, fourcc, original_fps, (original_width, original_height))

    frame_count = 0
    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                logging.info("Failed to obtain frame or EOF")
                break

            frame_count += 1

            # # 跳帧处理
            # if params.skip_frames > 0 and frame_count % (params.skip_frames + 1) != 0:
            #     # 只显示和保存，不处理
            #     if params.view:
            #         cv2.imshow('Student Attention Analysis', frame)
            #         key = cv2.waitKey(1) & 0xFF
            #         if key == ord('q'):
            #             break
            #     if out is not None:
            #         out.write(frame)
            #     continue

            # 性能监控更新
            frame_time = performance_monitor.update()
            fps = performance_monitor.get_fps()

            # 缩放帧用于处理（不限制人数）
            if params.scale_factor < 1.0:
                process_frame = cv2.resize(frame, (process_width, process_height))
            else:
                process_frame = frame.copy()

            # 人脸检测（不限制人数）
            bboxes, keypoints = face_detector.detect(process_frame)

            # 记录当前帧的学生数据
            current_students = []

            for bbox, keypoint in zip(bboxes, keypoints):
                # 将检测框坐标转换回原始分辨率
                if params.scale_factor < 1.0:
                    scale_x = original_width / process_width
                    scale_y = original_height / process_height
                    x_min = int(bbox[0] * scale_x)
                    y_min = int(bbox[1] * scale_y)
                    x_max = int(bbox[2] * scale_x)
                    y_max = int(bbox[3] * scale_y)
                else:
                    x_min, y_min, x_max, y_max = map(int, bbox[:4])

                width = x_max - x_min
                x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max)

                # 确保裁剪区域有效
                if x_min >= x_max or y_min >= y_max:
                    continue

                # 从原始帧中裁剪
                image = frame[y_min:y_max, x_min:x_max]
                if image.size == 0:
                    continue

                image = pre_process(image)
                image = image.to(device)

                # 验证输入数据在 GPU 上
                if frame_count % 100 == 0:  # 每100帧打印一次
                    print(f"输入数据设备: {image.device}")
                    print(f"模型设备: {next(head_pose.parameters()).device}")


                start = time.time()
                rotation_matrix = head_pose(image)

                # 验证输出在 GPU 上
                if frame_count % 100 == 0:
                    print(f"输出数据设备: {rotation_matrix.device}")
                    print(f"推理时间: {(time.time() - start) * 1000:.1f}ms")

                logging.info('Head pose estimation: %.2f ms' % ((time.time() - start) * 1000))

                euler = np.degrees(compute_euler_angles_from_rotation_matrices(rotation_matrix))
                p_pred_deg = euler[:, 0].cpu().numpy()[0]
                y_pred_deg = euler[:, 1].cpu().numpy()[0]
                r_pred_deg = euler[:, 2].cpu().numpy()[0]

                # 专注度分析
                if params.attention and attention_analyzer is not None:
                    status, score, color = attention_analyzer.calculate_attention_score(
                        p_pred_deg, y_pred_deg, r_pred_deg
                    )

                    # 绘制专注度信息
                    bbox_rect = [x_min, y_min, x_max, y_max]
                    frame = draw_attention_info(
                        frame, bbox_rect, status, score,
                        p_pred_deg, y_pred_deg, r_pred_deg, color
                    )

                    # 更新学生数据
                    attention_analyzer.update_student_data(
                        bbox_rect, p_pred_deg, y_pred_deg, r_pred_deg, status, score
                    )

                    current_students.append({'status': status, 'score': score})

                # 头部姿态可视化（如果不禁用）
                if not params.no_pose:
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

            # 绘制课堂统计信息
            if params.attention and attention_analyzer is not None:
                statistics = attention_analyzer.get_class_statistics()
                frame = draw_class_statistics(frame, statistics, len(current_students),
                                              fps if params.show_fps else None)

            if params.view:
                cv2.imshow('Student Attention Analysis', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r') and params.attention and attention_analyzer is not None:
                    # 显示实时报告
                    stats = attention_analyzer.get_class_statistics()
                    if stats:
                        print("\n=== 实时课堂报告 ===")
                        print(f"平均专注度: {stats['avg_score']:.1f}")
                        print(f"专注人数: {stats['focus_count']}")
                        print(f"分心人数: {stats['distracted_count']}")
                        print(f"处理帧数: {stats['total_frames']}")
                        if params.show_fps:
                            print(f"当前FPS: {fps:.1f}")

            # Write the frame to the video file if saving
            if out is not None:
                out.write(frame)

    # 最终报告
    if params.attention and attention_analyzer is not None:
        final_stats = attention_analyzer.get_class_statistics()
        if final_stats:
            print("\n=== 最终课堂分析报告 ===")
            print(f"总分析帧数: {final_stats['total_frames']}")
            print(f"最终平均专注度: {final_stats['avg_score']:.1f}")
            print(f"专注人数: {final_stats['focus_count']}")
            print(f"分心人数: {final_stats['distracted_count']}")
            print(f"专注比例: {final_stats['focus_count'] / final_stats['total_frames'] * 100:.1f}%")

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    main(args)