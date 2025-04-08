import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing')
parser.add_argument('--data_type', type=str, default="original", help='Data type:original or processed')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--fea_dim',default=20,type=int)

parser.add_argument('--lr', type=float, default=1e-4, help='learning_rate')

parser.add_argument('--mode', type=str, default="sliding", help='video process mode')
parser.add_argument('--model_type', type=str, default="mc3_18", help='video process models')
parser.add_argument('--max_num', type=int, default=-1, help='The input number in the testing setting')

parser.add_argument('--num_frames', type=int, default=16, help='Number of frames to extract from each video')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes in the dataset')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')

parser.add_argument('--step', type=int, default=16, help='video process step')
parser.add_argument('--trainflag',default=True)

parser.add_argument('--transform',default=None)
parser.add_argument('--decision_threshold',default=0.5)
parser.add_argument('--pretrained',default=True,help="是否预训练模型")
parser.add_argument('--frozen',default=False,help="是否冻结预训练模型")

parser.add_argument('--rotate_flag', type=bool, default=False, help='是否旋转')
parser.add_argument('--crop_flag', type=bool, default=False, help='是否剪裁')


args = parser.parse_args()
