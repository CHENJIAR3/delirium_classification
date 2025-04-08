import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.modules.module import T
# from pytorch_video.models.timesformer import TimeSformer
from torchvision.models.video import (mc3_18,MC3_18_Weights,
                                      mvit_v1_b, MViT_V1_B_Weights,
                                      mvit_v2_s, MViT_V2_S_Weights,
                                      r2plus1d_18,R2Plus1D_18_Weights,
                                      r3d_18, R3D_18_Weights,
                                      s3d,S3D_Weights,
                                      swin3d_b,Swin3D_B_Weights,
                                      swin3d_s,Swin3D_S_Weights,
                                      swin3d_t,Swin3D_T_Weights)
from thop import profile

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=10, output_dim=2):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
class VideoNet3D(nn.Module):
    def __init__(self, args):
        super(VideoNet3D, self).__init__()
        if args.model_type == "mc3_18":
            self.pretrain_weight = MC3_18_Weights.DEFAULT
            if args.pretrained:
                self.feature_extractor = mc3_18(weights=self.pretrain_weight)
            else:
                self.feature_extractor = mc3_18()

            # self.feature_extractor.fc = nn.Linear(512, fea_dim)

        if args.model_type == "mvit_v1":
            self.pretrain_weight = MViT_V1_B_Weights.DEFAULT
            if args.pretrained:

                self.feature_extractor = mvit_v1_b(weights=self.pretrain_weight)
            else:
                self.feature_extractor = mvit_v1_b()
            # self.feature_extractor.head = nn.Linear(768, fea_dim)
        if args.model_type == "mvit_v2":
            self.pretrain_weight = MViT_V2_S_Weights.DEFAULT
            if args.pretrained:
                self.feature_extractor = mvit_v2_s(weights=self.pretrain_weight)
            else:
                self.feature_extractor = mvit_v2_s()
            # self.feature_extractor.head = nn.Linear(768, fea_dim)
        if args.model_type == "r2+":
            self.pretrain_weight = R2Plus1D_18_Weights.DEFAULT
            if args.pretrained:
                self.feature_extractor = r2plus1d_18(weights=self.pretrain_weight)
            else:
                self.feature_extractor = r2plus1d_18()
            # self.feature_extractor.fc = nn.Linear(512, fea_dim)

        if args.model_type == "resnet":
            self.pretrain_weight = R3D_18_Weights.DEFAULT
            if args.pretrained:
                self.feature_extractor = r3d_18(weights=self.pretrain_weight)
            else:
                self.feature_extractor = r3d_18()
                # self.feature_extractor.fc = nn.Linear(512, fea_dim)

        if args.model_type == "s3d":
            self.pretrain_weight = S3D_Weights.DEFAULT
            if args.pretrained:
                self.feature_extractor = s3d(weights=self.pretrain_weight)
            else:
                self.feature_extractor = s3d()
            # self.feature_extractor.classifier = nn.Linear(1024, fea_dim)

        if args.model_type == "swin_b":
            self.pretrain_weight = Swin3D_B_Weights.DEFAULT
            if args.pretrained:
                self.feature_extractor = swin3d_b(weights=self.pretrain_weight)
            else:
                self.feature_extractor =  swin3d_b()
            # self.feature_extractor.head = nn.Linear(1024, fea_dim)
        if args.model_type == "swin_s":
            self.pretrain_weight = Swin3D_S_Weights.DEFAULT
            if args.pretrained:
                self.feature_extractor = swin3d_s(weights=self.pretrain_weight)
            else:
                self.feature_extractor =  swin3d_s()
            # self.feature_extractor.head = nn.Linear(1024, fea_dim)

        if args.model_type == "swin_t":
            self.pretrain_weight = Swin3D_T_Weights.DEFAULT
            if args.pretrained:
                self.feature_extractor = swin3d_t(weights=self.pretrain_weight)
            else:
                self.feature_extractor =  swin3d_t()
            # self.feature_extractor.head = nn.Linear(1024, fea_dim)
        # self.freeze_model()
        self.pool = torch.nn.AdaptiveAvgPool1d(args.fea_dim)

        self.mlp = MLPClassifier(input_dim=args.fea_dim, hidden_dim=(args.fea_dim)//2,output_dim=args.num_classes)
        # self.fc1 = nn.Linear(num_classes, 5*num_classes)
        # self.activation1 = nn.GELU()
        # self.fc2 = nn.Linear(5*num_classes, num_classes)
        # self.activation2 = nn.Sigmoid()
    def transforms(self):
        return self.pretrain_weight.transforms()
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = self.mlp(x)
        # x = self.fc1(x)
        # x = self.activation1(x)
        # x = self.fc2(x)
        # # x = self.activation2(x)
        # # x = self.resnet(x)
        # x = F.softmax(
        #     x,dim=1
        # )
        return x
    def freeze_model(self):
        # 获取最后一层的索引
        last_layer_index = len(list(self.feature_extractor.parameters())) - 1

        # 冻结除最后一层之外的所有层
        for i, param in enumerate(self.feature_extractor.parameters()):
            if i != last_layer_index:
                param.requires_grad = False
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() )
if __name__ =="__main__":
    x = torch.randn(1,3,16,224,224)
    model_types = ["mc3_18"]
    from utils.experimental_setting import args

    # "mc3_18","mvit_v1","mvit_v2","r2+",
    #                    "resnet","s3d","swin_b","swin_s","swin_t"
    for model_type in model_types:
        args.model_type = model_type
        model = VideoNet3D(args=args)
    #     print(model_type,model(x))
    # # model.freeze_model()
    # print(model(x))
    print(count_parameters(model))
    flops, params = profile(model, inputs=(x,),verbose=False)
    print(f"FLOPs: {flops / 1e9:.2f} G")
    print(f"Params:{params/1e6:.2f} M")
    from torchinfo import summary

    summary(model, input_size=x[:1, :, :, :, :].shape)



