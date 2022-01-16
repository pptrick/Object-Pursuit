import os
import torch
import argparse
from utils.util import create_dir

from application.oneshot._dataset import select_dataset
from application.oneshot._models import select_model
from application.oneshot._train import train_nshot

def nshot_get_args():
    parser = argparse.ArgumentParser(description='One shot learning',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general
    parser.add_argument('-dim', '--z_dim', dest='z_dim', type=int, nargs='?', default=100,
                        help='dimension of z')
    parser.add_argument('-o', '--out', dest='output_dir', type=str, nargs='?', default="./checkpoints_nshot/",
                        help='output directory')
    
    # data
    parser.add_argument('-imgs', '--imgs', dest='img_dir', type=str, nargs='?', help='directory of input images')
    parser.add_argument('-masks', '--masks', dest='mask_dir', type=str, nargs='?', help='directory of input masks')
    
    # pretrain
    parser.add_argument('-bases', '--bases', dest='bases_dir', type=str, nargs='?', default=None,
                        help='directory of bases (base list)')
    parser.add_argument('-backbone', '--backbone', dest='pretrained_backbone', type=str, nargs='?', default=None,
                        help='pretrained backbone path')
    parser.add_argument('-hypernet', '--hypernet', dest='pretrained_hypernet', type=str, nargs='?', default=None,
                        help='pretrained hypernet path')
    
    # select
    parser.add_argument('-resize', '--resize', dest='resize', type=tuple, nargs='?', default=(256, 256),
                        help='resize of training image')
    parser.add_argument('-model', '--model', dest='model', type=str, default="coeffnet", choices=["unet", "deeplab", "singlenet", "coeffnet"],
                        help='select model')
    parser.add_argument('-dataset', '--dataset', dest='dataset', type=str, default="DAVIS", choices=["iThor", "KITTI", "DAVIS"],
                        help='select dataset')
    
    # dataset setting
    parser.add_argument('-n', '--n', dest='n', type=int, nargs='?', default=1,
                        help='n-shot n: number of training samples')
    parser.add_argument('-nval', '--nval', dest='n_val', type=int, nargs='?', default=400,
                        help='number of validation samples')
    parser.add_argument('-shuffle', '--shuffle', dest='shuffle_seed', type=int, nargs='?', default=3,
                        help='shuffle seed for training set')
    
    # training setting
    parser.add_argument('-batch_size', '--batch_size', dest='batch_size', type=int, nargs='?', default=8,
                        help='batch size for nshot training')
    parser.add_argument('-epochs', '--epochs', dest='epochs', type=int, nargs='?', default=20,
                        help='epochs for nshot training')
    parser.add_argument('-lr', '--lr', dest='lr', type=float, nargs='?', default=0.0004,
                        help='learning rate for nshot training')
    parser.add_argument('-eval_step', '--eval_step', dest='eval_step', type=float, nargs='?', default=10,
                        help='eval interval')
    parser.add_argument('-save_ckpt', '--save_ckpt', dest='save_ckpt', action="store_true",
                        help='if true, save checkpoints')
    parser.add_argument('-save_viz', '--save_viz', dest='save_viz', action="store_true",
                        help='if true, save visualization prediction')
    parser.add_argument('-use_backbone', '--use_backbone', dest='use_backbone', action="store_true",
                        help='if true, the weights of the backbone will not be predicted by the hypernet')
    parser.add_argument('-use_dice_loss', '--use_dice_loss', dest='use_dice_loss', action="store_true",
                        help='if true, the accuracy will be reported in dice loss')
    
    return parser.parse_args()

def main(args):
    default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # add output dir
    create_dir(args.output_dir)
    
    net = select_model(model=args.model,
                       device=default_device,
                       z_dim=args.z_dim,
                       base_dir=args.bases_dir,
                       pretrained_hypernet=args.pretrained_hypernet,
                       pretrained_backbone=args.pretrained_backbone,
                       use_backbone=args.use_backbone)
    
    train_dataset, test_dataset = select_dataset(dataset=args.dataset,
                                                 img_dir=args.img_dir,
                                                 mask_dir=args.mask_dir,
                                                 resize=args.resize,
                                                 n_test=args.n_val,
                                                 n=args.n,
                                                 shuffle_seed=args.shuffle_seed)
    
    train_nshot(net,
                default_device,
                train_dataset,
                test_dataset,
                ckpt_path=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                eval_step=args.eval_step,
                save_ckpt=args.save_ckpt,
                save_viz=args.save_viz,
                use_dice=args.use_dice_loss,
                args=args)
