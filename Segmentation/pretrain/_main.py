import os
import torch
import argparse

from pretrain._train import *
from pretrain._dataset import *
from pretrain._model import *

def pretrain_get_args():
    parser = argparse.ArgumentParser(description='Pretrain hypernet (and backbone, if exist) by multi-object joint training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general
    parser.add_argument('-dim', '--z_dim', dest='z_dim', type=int, nargs='?', default=100,
                        help='dimension of z')
    parser.add_argument('-o', '--out', dest='output_dir', type=str, nargs='?', default="./checkpoints_nshot/",
                        help='output directory')
    
    # data
    parser.add_argument('-data', '--data', dest='data_dir', type=str, nargs='?', help='directory of input data, the upper dir')
    
    # select
    parser.add_argument('-resize', '--resize', dest='resize', type=tuple, nargs='?', default=(256, 256),
                        help='resize of training image')
    parser.add_argument('-dataset', '--dataset', dest='dataset', type=str, default="DAVIS", choices=["iThor", "KITTI", "DAVIS"],
                        help='select dataset')
    
    # training setting
    parser.add_argument('-batch_size', '--batch_size', dest='batch_size', type=int, nargs='?', default=8,
                        help='batch size for nshot training')
    parser.add_argument('-epochs', '--epochs', dest='epochs', type=int, nargs='?', default=100,
                        help='epochs for nshot training')
    parser.add_argument('-lr', '--lr', dest='lr', type=float, nargs='?', default=0.0004,
                        help='learning rate for nshot training')
    parser.add_argument('-eval_step', '--eval_step', dest='eval_step', type=int, nargs='?', default=10,
                        help='eval interval')
    parser.add_argument('-eval_n', '--eval_n', dest='eval_n', type=int, nargs='?', default=-1,
                        help='eval data number')
    parser.add_argument('-save_ckpt', '--save_ckpt', dest='save_ckpt', action="store_true",
                        help='if true, save checkpoints')
    parser.add_argument('-use_backbone', '--use_backbone', dest='use_backbone', action="store_true",
                        help='if true, the weights of the backbone will not be predicted by the hypernet')
    parser.add_argument('-use_dice_loss', '--use_dice_loss', dest='use_dice_loss', action="store_true",
                        help='if true, the accuracy will be reported in dice loss')
    parser.add_argument('-num_balance', '--num_balance', dest='num_balance', action="store_true",
                        help='if true, make data sample numbers of all objects the same')
    parser.add_argument('-freeze_backbone', '--freeze_backbone', dest='freeze_backbone', action="store_true",
                        help='if true, the backbone will not be updated during training')
    parser.add_argument('-trainset_only', '--trainset_only', dest='trainset_only', action="store_true",
                        help='if true, only use training set in the whole dataset during training')
    
    return parser.parse_args()

def main(args):
    default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # get dataset
    if args.dataset == "DAVIS":
        dataloader, dataset = Davis_Multi_Dataloader(args.data_dir, 
                                                     args.batch_size, 
                                                     resize=args.resize,
                                                     num_workers=8,
                                                     num_balance=args.num_balance,
                                                     random_crop=True,
                                                     trainset_only=args.trainset_only)
        class_num = dataset.class_num
    elif args.dataset == "iThor":
        dataloader, dataset = iThor_Multi_Dataloader(args.data_dir, 
                                                     args.batch_size, 
                                                     resize=args.resize,
                                                     num_workers=8,
                                                     num_balance=args.num_balance,
                                                     random_crop=True)
        class_num = dataset.class_num
    else:
        raise NotImplementedError
    
    # get model
    net = get_multinet(class_num, 
                 args.z_dim, 
                 default_device, 
                 use_backbone=args.use_backbone,
                 freeze_backbone=args.freeze_backbone
                 )
    
    # train
    joint_train(net=net,
                device=default_device,
                dataloader_train=dataloader,
                dataset_eval=dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                ckpt_path=args.output_dir,
                eval_step=args.eval_step,
                n_val=args.eval_n,
                save_ckpt=args.save_ckpt,
                use_dice=args.use_dice_loss,
                args=args)
    