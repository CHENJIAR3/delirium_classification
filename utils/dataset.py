import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import glob
from torch.utils.data import DataLoader,random_split
import matplotlib.pyplot as plt
from torchvision import transforms


from torchvision.models.video import (mc3_18,MC3_18_Weights,
                                      r3d_18, R3D_18_Weights,
                                      mvit_v1_b,MViT_V1_B_Weights,
                                      mvit_v2_s,MViT_V2_S_Weights,
                                      s3d,S3D_Weights,
                                      r2plus1d_18,R2Plus1D_18_Weights,
                                      swin3d_b,Swin3D_B_Weights,
                                      swin3d_s,Swin3D_S_Weights,
                                      swin3d_t,Swin3D_T_Weights)
class VideoDataset(Dataset):
    def __init__(self, data_dir, args):
        self.data_dir = data_dir
        self.transform = args.transform
        self.rotate_flag = args.rotate_flag
        self.crop_flag = args.crop_flag
        # if self.transform == None:
            # "mc3_18","mvit_v1","mvit_v2", "r2+",
            #                    "resnet", "s3d",
            # "swin_b", "swin_s", "swin_t"
        if args.model_type == "mc3_18":
            self.transform = MC3_18_Weights.DEFAULT.transforms()
        if args.model_type == "resnet":
            self.transform = R3D_18_Weights.DEFAULT.transforms()
        if args.model_type == "mvit_v1":
            self.transform = MViT_V1_B_Weights.DEFAULT.transforms()
        if args.model_type == "mvit_v2":
            self.transform = MViT_V2_S_Weights.DEFAULT.transforms()
        if args.model_type == "r2+":
            self.transform = R2Plus1D_18_Weights.DEFAULT.transforms()
        if args.model_type == "s3d":
            self.transform = S3D_Weights.DEFAULT.transforms()
        if args.model_type == "swin_b":
            self.transform = Swin3D_B_Weights.DEFAULT.transforms()
        if args.model_type == "swin_s":
            self.transform = Swin3D_S_Weights.DEFAULT.transforms()
        if args.model_type == "swin_t":
            self.transform = Swin3D_T_Weights.DEFAULT.transforms()

        self.num_frames = args.num_frames
        self.mode = args.mode
        self.step = args.step
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        print(self.class_to_idx)
        self.video_list = []

        for cls in self.classes:
            class_dir = os.path.join(data_dir, cls)
            for video in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video)
                if self.mode == 'fixed':
                    self.video_list.append((video_path, self.class_to_idx[cls]))
                elif self.mode == 'complete':
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    self.video_list.append((video_path, total_frames, self.class_to_idx[cls]))
                elif self.mode == 'sliding':
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    starts = list(range(0, total_frames - self.num_frames + 1, self.step))
                    if starts:
                        for start in starts:
                            self.video_list.append((video_path, start, self.class_to_idx[cls]))
                    else:
                        self.video_list.append((video_path, 0, self.class_to_idx[cls]))
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        if self.mode == 'fixed':
            video_path, label = self.video_list[idx]
            frames = self.load_fixed_frames(video_path)
        elif self.mode == 'complete':
            video_path, total_frames, label = self.video_list[idx]
            frames = self.load_complete_video(video_path, total_frames)
        elif self.mode == 'sliding':
            video_path, start_frame, label = self.video_list[idx]

            frames = self.load_sliding_frames(video_path, start_frame)
            # if self.rotate_img:
            #     print("start")
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # if self.transform:

        frames = np.transpose(frames, (0, 3, 1, 2))  # (T, H, W, C) -> (T, C,  H, W)
        frames = torch.from_numpy(frames).float()/255.0
        frames = self.transform(frames)
        label = torch.tensor(label).long()
        return frames, label
    def transform_frame(self,frame):
        frame = cv2.resize(frame, (256, 256))

        # Center crop to 224x224
        h, w, _ = frame.shape
        start_h = (h - 224) // 2
        start_w = (w - 224) // 2
        frame = frame[start_h:start_h + 112, start_w:start_w + 112, :]
        frame = torch.from_numpy(frame).float()/ 255.0
        #   # uint8 to float
        mean_tensor = torch.tensor([0.45, 0.45, 0.45]).view(1, 1, 3)
        std_tensor = torch.tensor([0.225, 0.225, 0.225]).view(1, 1, 3)
        frame =   (frame - mean_tensor)/std_tensor
        return frame
    def rotate_image(self,image, angle=-90):
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

    def random_crop(self,image, crop_ratio=0.8):
        """随机裁剪保留核心区域"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
        y = np.random.randint(0, h - new_h)
        x = np.random.randint(0, w - new_w)
        return image[y:y + new_h, x:x + new_w]
        # else:
        #     return image

    def load_fixed_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if cnt in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.rotate_image(frame)
                # frame = self.transform_frame(frame)

                # Center crop to 224x224

                # frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
                if len(frames) == self.num_frames:
                    break
            cnt += 1
        cap.release()
        if len(frames) < self.num_frames:
            frames.extend([frames[-1]] * (self.num_frames - len(frames)))
        frames = np.stack(frames, axis=0)
        return frames

    def load_complete_video(self, video_path, total_frames):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.rotate_image(frame)
            # frame = self.transform_frame(frame)
            # frame = cv2.resize(frame,(224,224))
            # frame = torch.from_numpy(frame).float() / 255.0  # uint8 to float

            # frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()
        if len(frames) < total_frames:
            frames.extend([frames[-1]] * (total_frames - len(frames)))
        frames = np.stack(frames, axis=0)
        return frames

    def load_sliding_frames(self, video_path, start_frame,isrotate=False):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.rotate_image(frame)
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1])
                else:
                    blank_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                    frames.append(blank_frame)
        cap.release()
        frames = np.stack(frames, axis=0)
        return frames
def show_image(videopath):
    cap = cv2.VideoCapture(videopath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cnt in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # plt.imshow(frame)
            # plt.axis('off')  # 关闭坐标轴
            # plt.show()
 # (T, H, W, C) -> (T, C,  H, W)
            frame = np.transpose(frame, (2,0,1))
            frame = torch.from_numpy(frame).float() / 255.0
            frame = torch.unsqueeze(frame,dim=0)
            frame = MC3_18_Weights.DEFAULT.transforms()(frame)
            # frame = cv2.resize(frame, (224, 224))
            frame = np.transpose(frame.squeeze(), (1,2,0))
            plt.imshow(frame)
            plt.axis('off')  # 关闭坐标轴
            plt.show()
            #
            #
            # plt.imshow(frame)
            # plt.axis('off')  # 关闭坐标轴
            # plt.show()
            # 释放视频捕获对象
            cap.release()

if __name__ == "__main__":
    videopaths = glob.glob("../data/*/*.mp4")
    show_image(videopaths[0])
    dataset = VideoDataset("../data", num_frames=100, mode='sliding', step=10)
    #
    # dataset = VideoDataset("../data", num_frames=100, mode='fixed', step=10)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    # for batch in dataloader:
    #     break
    # inputs, labels = batch
    # print(torch.mean(inputs[0],dim=(1,2,3)),torch.std(inputs[0],dim=(1,2,3)))
