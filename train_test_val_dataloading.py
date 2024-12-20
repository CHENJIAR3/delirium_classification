import glob
import os
import shutil
from torch.utils.data import random_split
import torch
torch.manual_seed(3569)
def clear_dir(folder_path):
    # 列出文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                # 删除文件夹及其内容
                shutil.rmtree(dir_path)
                print(f"Deleted folder: {dir_path}")
            except Exception as e:
                print(f"Failed to delete folder {dir_path}. Reason: {e}")

def mkdatadir_labels(datadir,labels):
    clear_dir(datadir)
    os.makedirs(datadir, exist_ok=True)
    os.makedirs(datadir+"/"+labels[0].split("/")[-1], exist_ok=True)
    os.makedirs(datadir+"/"+labels[1].split("/")[-1], exist_ok=True)
if __name__ == "__main__":

    labels = glob.glob("./data/*")
    video_paths = glob.glob("./data/*/*.mp4")

    indices = torch.randperm(len(video_paths))
    shuffled_video_paths = [video_paths[i] for i in indices]

    # 计算划分点
    train_idx = int(0.7 * len(video_paths))
    # test_idx = int(0.8 * len(video_paths))


    train_dir = 'dataset/train'
    val_dir = 'dataset/val'
    test_dir = 'dataset/test'


    mkdatadir_labels(train_dir,labels)
    mkdatadir_labels(val_dir,labels)
    mkdatadir_labels(test_dir,labels)
    #
    # # 创建目录
    #
    # # 获取阳性视频列表

    train_paths = shuffled_video_paths[:train_idx]
    # val_paths = shuffled_video_paths[train_idx:test_idx]
    test_paths = shuffled_video_paths[train_idx:]
    #
    # # 移动文件
    for video in train_paths:
        shutil.copy(video, os.path.join(train_dir, video.split("/",maxsplit=2)[-1]))

    # for video in val_paths:
    #     shutil.copy(video, os.path.join(val_dir, video.split("/",maxsplit=2)[-1]))

    for video in test_paths:
        shutil.copy(video, os.path.join(test_dir, video.split("/",maxsplit=2)[-1]))

