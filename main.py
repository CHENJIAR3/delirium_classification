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
from trainers import model_train,model_test

if __name__ == '__main__':
    model_types = [ "r2+"]
    # "mc3_18","mvit_v1","mvit_v2","r2+",
    #     #                    "resnet","s3d","swin_b","swin_s","swin_t"
    result = []
    args.device="cuda:1"
    test_dir = './dataset/test'
    args.trainflag = True
    for model_type in  model_types:

        args.model_type = model_type

        for exp in ["no_pretrained"]:
            if exp == "no_pretrained": args.pretrained = False;args.frozen = False
            if exp == "frozen": args.pretrained = True;args.frozen = True
            if exp == "main": args.pretrained = True;args.frozen = False

            if args.trainflag:
                model = VideoNet3D(args)
                model = model.to(args.device)
                val_acc,best_val_loss,epoch_loss = model_train(model,args,data_dir="./dataset")
                best_train_loss = np.round(np.min(epoch_loss),2)
            model = VideoNet3D(args)
            if args.pretrained == False:
                model.load_state_dict(
                    torch.load(f'./saved_models/best_{args.model_type}_no_pretrained.pth', map_location=args.device))
            else:
                if args.frozen == True:
                    model.load_state_dict(
                        torch.load(f'./saved_models/best_{args.model_type}_frozen.pth', map_location=args.device))
                else:
                    model.load_state_dict(
                        torch.load(f'./saved_models/best_{args.model_type}.pth', map_location=args.device))
            # for max_num in [1,5,10,15]:
            #     args.max_num = max_num
            test_df = model_test(model,args,test_dir)
            if args.pretrained == False:
                test_df.to_csv(f"./results/{model_type}_{formatted_date}_no_pretrained.csv")

            else:
                if args.frozen == True:
                    test_df.to_csv(f"./results/{model_type}_{formatted_date}_frozen.csv")

                else:
                    test_df.to_csv(f"./results/{model_type}_{formatted_date}.csv")
