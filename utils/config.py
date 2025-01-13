import os
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def TestParserArguments():
    args = argparse.ArgumentParser()

    # Directory Setting
    args.add_argument('--data_root', type=str, default='', help='Root directory for the dataset, containing data patches.')
    args.add_argument('--exp', type=str, default='./exp', help='Directory to save experiment outputs such as model checkpoints and logs.')

    # Network
    args.add_argument('--network', type=str, default='efficientnet-b4', help='Classifier network architecture to use. Default is EfficientNet-B4, but other options like ResNet34 can also be used.')
    args.add_argument('--resume', type=str, default='', help='Path to pre-trained weights to resume training or fine-tune the model.')
    args.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for the fully connected (FC) layer to prevent overfitting. Default is 0.5.')
    args.add_argument('--use_gpu', default="True", type=str2bool, help='Whether to use a GPU for computation. Set to False to run on CPU only.')
    args.add_argument('--gpu_id', default="0", type=str, help='GPU ID to use for training or testing when multiple GPUs are available.')

    args.add_argument('--batch_size', type=int, default=15, help='Number of samples per batch for training or testing. Default is 15.')
    args.add_argument('--mode', type=str, default='test', help='Mode of operation. Use "test" for submission.')

    # Test parameters
    args.add_argument('--modal', type=str, default='X-ray', help='Test data modality to use. Options are "X-ray" or "VFA".')
    args.add_argument('--stype', type=str, default='osteoporosis', help='Test label type to use. Options are "osteoporosis" or "fracture".')
    
    args = args.parse_args()

    # Normalization target image

    if args.use_gpu:
        args.ngpu = len(args.gpu_id.split(","))
    else:
        args.gpu_id = 'cpu'
        args.ngpu = 'cpu'  

    
    args.num_classes=1

    return args