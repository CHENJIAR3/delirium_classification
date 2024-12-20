import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and testing')
parser.add_argument('--num_frames', type=int, default=16, help='Number of frames to extract from each video')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes in the dataset')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning_rate')
parser.add_argument('--mode', type=str, default="sliding", help='video process mode')
parser.add_argument('--step', type=int, default=16, help='video process step')

args = parser.parse_args()
