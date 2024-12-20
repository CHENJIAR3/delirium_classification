import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.model import VideoNet3D
from utils.dataset import VideoDataset
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import pandas as pd
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from models.model import VideoNet3D
from utils.dataset import VideoDataset
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from tqdm import tqdm
from utils.experimental_setting import args
torch.manual_seed(3569)
import datetime
formatted_date = datetime.datetime.now().strftime("%Y-%m-%d")

def model_train(args,data_dir='./dataset'):
    device = args.device
    # 确保'saved_models'目录存在
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    # 模型定义
    model = VideoNet3D(num_classes=args.num_classes,model_type=args.model_type)
    model = model.to(args.device)
    transform = model.transforms()
    # 数据加载
    train_dataset = VideoDataset(data_dir+'/train', transform=transform, num_frames=args.num_frames,mode=args.mode,step=args.step)
    train_set, internal_test_set = random_split(train_dataset, [int(0.8 * len(train_dataset)), len(train_dataset) - int(0.8 * len(train_dataset))])
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(internal_test_set, batch_size=args.batch_size, shuffle=False)

    # val_dataset = VideoDataset(data_dir+'/val', transform=transform, num_frames=args.num_frames,mode=args.mode,step=args.step)
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)



    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1), momentum=0.9

    best_val_loss = float('inf')
    epoch_loss = np.zeros(args.epochs)
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        # model.freeze_model()
        running_loss = 0.0
        for inputs, labels in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss[epoch] = running_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}, Training Loss: {epoch_loss[epoch]:.4f}')

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'./saved_models/best_{args.model_type}.pth')
            print('Best model saved.')
        if epoch%1 == 0:
            val_acc = model_test(args,"./dataset/val",test_dataloader=val_dataloader)
        # scheduler.step()
    return val_acc,best_val_loss,epoch_loss


def model_test(args,test_dir='./dataset/test',test_dataloader=None):
    device = args.device
    # 参数设置

    # 模型定义
    model = VideoNet3D(num_classes=args.num_classes,model_type=args.model_type)
    model = model.to(device)
    transform = model.transforms()
    # 数据加载
    if test_dir is not None and test_dataloader is None:
        test_dataset = VideoDataset(test_dir,transform=transform, num_frames=args.num_frames,mode=args.mode)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 模型加载
    model = VideoNet3D(num_classes=args.num_classes,model_type=args.model_type)
    model.load_state_dict(torch.load(f'./saved_models/best_{args.model_type}.pth', map_location=device))
    model = model.to(device)
    model.eval()

    # 测试
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(f"Pred:{predicted.cpu().numpy()},label:{labels.cpu().numpy()}")
    test_acc = np.round(100 * correct / total,2)
    print(f'Accuracy on test set: {test_acc:.2f}%')
    return test_acc

if __name__ == '__main__':
    model_types = [ "mc3_18","mvit_v1", "mvit_v2", "r2+",
                   "resnet", "s3d", "swin_b", "swin_s", "swin_t"]
    result = []
    args.device="cuda:2"
    for model_type in  model_types:
        args.model_type = model_type
        # vit,swin_b
        # val_acc,best_val_loss,epoch_loss = model_train(args)
        # best_train_loss = np.round(np.min(epoch_loss),2)
        test_acc = model_test(args)
        # val_acc = np.round(val_acc,2)
        # best_val_loss = np.round(best_val_loss,2)
        # result.append([best_train_loss,val_acc,best_val_loss,test_acc])
    # df = pd.DataFrame(result, columns=["train_loss", "val_acc", "val_loss", "test_acc"],
    #                   index=model_types)  # model_test(args, test_dir="dataset/val")
    # df.to_csv(f"./results/{formatted_date}.csv")
    # model_test(args, test_dir="dataset/train")
