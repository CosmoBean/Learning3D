import numpy as np
import argparse

import torch
from models import seg_model, seg_pp_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg, rotate


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, nargs='*', default=[0], help="indices of the objects to visualize")
    parser.add_argument('--pp', action='store_true', help='Use PointNet++ instead of PointNet')
    parser.add_argument('--rot', type=float, default=0, help='Rotate test point clouds by this many degrees')
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    model = (seg_pp_model if args.pp else seg_model)(args.num_seg_class).to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg{}/{}.pt'.format("_pp" if args.pp else "", args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    np.random.seed(0)
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind]).to(torch.long)
    test_data = rotate(test_data, args.rot)

    with torch.no_grad():
        pred_label = torch.cat([model(x.to(args.device)).argmax(-1).cpu() for x in test_data.split(args.batch_size)])

    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    obj_acc = pred_label.eq(test_label).float().mean(1)
    print("worst objects: {}".format([(i.item(), round(obj_acc[i].item(), 4)) for i in obj_acc.argsort()[:5]]))

    # Visualize Segmentation Result (Pred VS Ground Truth)
    for i in args.i:
        print("object {}: accuracy {:.4f}".format(i, obj_acc[i]))
        viz_seg(test_data[i], test_label[i], "{}/gt_{}_{}.gif".format(args.output_dir, args.exp_name, i), args.device)
        viz_seg(test_data[i], pred_label[i], "{}/pred_{}_{}.gif".format(args.output_dir, args.exp_name, i), args.device)
