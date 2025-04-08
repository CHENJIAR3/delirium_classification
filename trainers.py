
import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from models.model import VideoNet3D,count_parameters
from utils.dataset import VideoDataset
import torch.nn as nn
from tqdm import tqdm
torch.manual_seed(3569)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from thop import profile

def model_train(model,args,data_dir='./dataset'):
    device = args.device
    print(device)
    # 确保'saved_models'目录存在
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    # 模型定义
    model.to(device)
    if args.frozen:
        model.freeze_model()
    transform = model.transforms()
    # 数据加载
    train_dataset = VideoDataset(data_dir+'/train', args)
    train_set, internal_test_set = random_split(train_dataset, [int(0.8 * len(train_dataset)), len(train_dataset) - int(0.8 * len(train_dataset))])
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,prefetch_factor=100)
    val_dataloader = DataLoader(internal_test_set, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers,prefetch_factor=100)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    epoch_loss = np.zeros(args.epochs)
    # 训练循环
    for epoch in range(args.epochs):
        print(epoch)
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
            if args.pretrained == False:
                torch.save(model.state_dict(), f'./saved_models/best_{args.model_type}_no_pretrained.pth')

            else:
                if args.frozen == True:
                    torch.save(model.state_dict(), f'./saved_models/best_{args.model_type}_frozen.pth')
                else:
                    torch.save(model.state_dict(), f'./saved_models/best_{args.model_type}.pth')

            print('Best model saved.')
        if epoch%1 == 0:
            args.testset = "val"
            val_acc = model_val(args,test_dataloader=val_dataloader)
        # scheduler.step()
    return val_acc,best_val_loss,epoch_loss


def model_val(args,test_dir='./dataset/test',test_dataloader=None):
    device = args.device
    # 参数设置
    # 数据加载
    if test_dir is not None and test_dataloader is None:
        test_dataset = VideoDataset(test_dir,args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 模型加载
    model = VideoNet3D(args)
    if args.pretrained == False:
        model.load_state_dict(
            torch.load(f'./saved_models/best_{args.model_type}_no_pretrained.pth', map_location=device))
    else:
        if args.frozen == True:
            model.load_state_dict(
                torch.load(f'./saved_models/best_{args.model_type}_frozen.pth', map_location=device))
        else:
            model.load_state_dict(
                torch.load(f'./saved_models/best_{args.model_type}.pth', map_location=device))

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
            # print(f"Pred:{predicted.cpu().numpy()},label:{labels.cpu().numpy()}")
    val_acc = np.round(100 * correct / total,2)
    print(f'Accuracy on {args.testset} set: {val_acc:.2f}%')
    return val_acc

def model_test(model,args,test_dir='./dataset/test'):
    from app_function import load_sliding_frames
    import glob

    device = args.device
    bs = args.batch_size * 4
    model = model.to(device)
    model.eval()
    transform = model.transforms()

    test_videos = glob.glob(test_dir + "/*/*.mp4")
    preds = []
    labels = []
    with torch.no_grad():
        for test_path in tqdm(test_videos):
            # 使用with语句管理数据加载
            with torch.cuda.stream(torch.cuda.Stream()):  # 创建独立CUDA流
                data = load_sliding_frames(test_path, max_num=args.max_num,transform=transform).to(device,non_blocking=True)
            label = 0 if "阳性病人" in test_path else 1
            if len(data) > bs:
                pred = []
                # 分块处理并立即释放中间结果
                for i in range(0, len(data), bs):
                    chunk = data[i:i + bs]
                    output = model(chunk)
                    chunk_pred = output.argmax(dim=1).float().cpu().numpy()
                    pred.append(chunk_pred)
                    del chunk, output  # 立即释放
                    torch.cuda.empty_cache() if i % 10 == 0 else None  # 定期清理

                pred = np.concatenate(pred)
                final_pred = np.mean(pred) > args.decision_threshold
            else:
                output = model(data)
                pred = output.argmax(dim=1).float().mean()
                final_pred = pred.item() > args.decision_threshold
                del output  # 立即释放
            preds.append([final_pred])

            labels.append(label)
        # bs


    preds = np.asarray(preds)
    # preds = torch.concat(preds).cpu().detach().numpy()
    labels = np.asarray(labels)
    df = get_metrics(args, preds, labels)
    flops, _ = profile(model, inputs=(data[:1,:,:,:,:],))
    params = count_parameters(model)
    print(f"FLOPs: {flops / 1e9:.2f} G")
    print(f"Params:{params/1e6:.2f} M")
    df["FLOPs(G)"] = flops / 1e9
    df["Params(M)"] = params / 1e6

    return df
def get_metrics(args,preds,labels):
    import pandas as pd
    class_names = ["delirium","non-delirium"]
    df = pd.DataFrame({"acc": [],
                       "pre":[],
                       "rec":[],
                       "f1":[]})
    for i in range(args.num_classes):
        df.loc[class_names[i], "acc"] = accuracy_score(labels==i, preds==i)
        df.loc[class_names[i], "pre"] = precision_score(labels==i, preds==i)
        df.loc[class_names[i], "rec"] = recall_score(labels==i, preds==i)
        df.loc[class_names[i], "f1"] = f1_score(labels==i, preds==i)
    df.loc["Avg"] = df.mean()
    return df
