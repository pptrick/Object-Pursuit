import torch
import argparse
from utils.gen_bases import genBases
from object_pursuit.pursuit import pursuit
from object_pursuit.eval import evalPursuit

default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser(description='Run Pursuit Algorithm (or remove redundancy of pretrained bases)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dim', '--z_dim', dest='z_dim', type=int, nargs='?', default=100,
                        help='dimension of z')
    parser.add_argument('-d', '--data', dest='data_dir', type=str, nargs='?',
                        help='training data directory')
    parser.add_argument('-o', '--out', dest='output_dir', type=str, nargs='?', default="./checkpoints",
                        help='output directory')
    parser.add_argument('-zs', '--zs', dest='zs_dir', type=str, nargs='?', default="./Bases",
                        help='directory of initial zs (object list)')
    parser.add_argument('-bases', '--bases', dest='bases_dir', type=str, nargs='?', default="./Bases",
                        help='directory of initial bases (base list)')
    parser.add_argument('-backbone', '--backbone', dest='pretrained_backbone', type=str, nargs='?', default="./checkpoints/checkpoint.pth",
                        help='pretrained backbone path')
    parser.add_argument('-hypernet', '--hypernet', dest='pretrained_hypernet', type=str, nargs='?', default="./checkpoints/checkpoint.pth",
                        help='pretrained hypernet path')
    parser.add_argument('-resize', '--resize', dest='resize', type=tuple, nargs='?', default=(256, 256),
                        help='resize of training image')
    parser.add_argument('-order', '--order', dest='order', type=str, default="sequence", choices=["sequence", "random"],
                        help='object order')
    parser.add_argument('-dataset', '--dataset', dest='dataset', type=str, default="CO3D", choices=["iThor", "CO3D", "DAVIS"],
                        help='select dataset')
    parser.add_argument('-thres', '--thres', dest='thres', type=float, default=0.6,
                        help='pursuit quality measure threshold (of whether an object can be expressed)')
    parser.add_argument('-use_backbone', '--use_backbone', dest='use_backbone', action="store_true",
                        help='if true, the weights of the backbone will not be predicted by the hypernet')
    parser.add_argument('-save_interval', '--save_interval', dest='save_interval', type=int, default=0,
                        help='the interval object number of saving checkpoints during pursuit')
    parser.add_argument('-eval', '--eval', dest='eval', action="store_true",
                        help='use this flag to evaluate pursuit result (eval mode)')
    
    return parser.parse_args()

if __name__ == '__main__':    
    args = get_args()
    if not args.eval:
        pursuit(z_dim=args.z_dim, 
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                device=default_device,
                dataset=args.dataset,
                initial_zs=args.zs_dir,
                pretrained_bases=args.bases_dir,
                pretrained_backbone=args.pretrained_backbone,
                pretrained_hypernet=args.pretrained_hypernet,
                resize=args.resize,
                select_strat=args.order,
                express_threshold=args.thres,
                use_backbone=args.use_backbone,
                save_temp_interval=args.save_interval,
                log_info=f"Data: {args.order}; threshold: {args.thres}")
    else:
        evalPursuit(z_dim=args.z_dim, 
                    device=default_device, 
                    dataset=args.dataset, 
                    data_dir=args.data_dir, 
                    ckpt_dir=args.output_dir, 
                    batch_size=8, 
                    use_backbone=args.use_backbone)