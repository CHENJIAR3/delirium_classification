import numpy as np
import logging
from models.model import VideoNet3D
import cv2
from utils.experimental_setting import args
import torch


from torchvision.models.video import (MC3_18_Weights,
                                      R3D_18_Weights,
                                      MViT_V1_B_Weights,
                                      MViT_V2_S_Weights,
                                      R2Plus1D_18_Weights,
                                      Swin3D_B_Weights,
                                      Swin3D_S_Weights,
                                      Swin3D_T_Weights)

# 初始化模型
def initialize_model(args):
    model_type = args.model_type
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = f"./saved_models/best_{model_type}.pth"

    model = VideoNet3D(args).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        if model_type == "mc3_18":
            transform = MC3_18_Weights.DEFAULT.transforms()
        if model_type == "resnet":
            transform = R3D_18_Weights.DEFAULT.transforms()
        if model_type == "mvit_v1":
            transform = MViT_V1_B_Weights.DEFAULT.transforms()
        if model_type == "mvit_v2":
            transform = MViT_V1_B_Weights.DEFAULT.transforms()
        if model_type == "r2+":
            transform = MViT_V1_B_Weights.DEFAULT.transforms()
        if model_type == "s3d":
            transform = MViT_V1_B_Weights.DEFAULT.transforms()
        if model_type == "swin_b":
            transform = Swin3D_B_Weights.DEFAULT.transforms()
        if model_type == "swin_s":
            transform = Swin3D_S_Weights.DEFAULT.transforms()
        if model_type == "swin_t":
            transform = Swin3D_T_Weights.DEFAULT.transforms()
        return model, device,transform
    except Exception as e:
        logging.error(f"Model loading failed: {str(e)}")
        raise


def load_fixed_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, args.num_frames, dtype=int)
    frames = []

    for cnt in range(max(frame_indices) + 1):
        ret, frame = cap.read()
        if not ret:
            break
        if cnt in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    # 处理帧数不足的情况
    if len(frames) < args.num_frames:
        frames += [frames[-1]] * (args.num_frames - len(frames))

    # 转换为张量
    frames = np.stack(frames, axis=0)
    frames = np.transpose(frames, (0, 3, 1, 2))  # (T, H, W, C) -> (T, C, H, W)
    frames = torch.from_numpy(frames).float() / 255.0
    frames = MC3_18_Weights.DEFAULT.transforms()(frames)
    return frames
def rotate_image(image, angle=-90):
    # 获取图像的高度和宽度
    (h, w) = image.shape[:2]

    if w<h:
        # 计算图像的中心点
        center = (w // 2, h // 2)
        # 获取旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 执行旋转
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated
    else:
        return image
def load_sliding_frames(video_path,max_num=10,transform=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    starts = list(range(0, total_frames - args.num_frames + 1, args.num_frames))
    max_num = min( total_frames // args.num_frames, max_num )
    frames_all = []
    for start in starts[:max_num]:
        cap = cv2.VideoCapture(video_path)

        cap.set(cv2.CAP_PROP_POS_FRAMES,start)
        frames = []
        for _ in range(args.num_frames):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = rotate_image(frame)
                # frame = self.transform_frame(frame)
                # frame = cv2.resize(frame, (224, 224))
                # frame = torch.from_numpy(frame).float() / 255.0  # uint8 to float
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1])
                else:
                    blank_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                    frames.append(blank_frame)
        cap.release()
        frames = np.stack(frames, axis=0)
        frames = np.transpose(frames, (0, 3, 1, 2))  # (T, H, W, C) -> (T, C, H, W)
        frames = torch.from_numpy(frames).float() / 255.0
        if transform is not None:
            frames = transform(frames)
        frames_all.append(frames)
    frames_all = np.stack(frames_all, axis=0)
    frames_all = torch.from_numpy(frames_all).float()
    return frames_all