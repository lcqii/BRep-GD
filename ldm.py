import os
from utils import *
from tensorboardX import SummaryWriter
args = get_args_ldm()


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

from dataset import *
from trainer import *

def run(args):
    train_writer = SummaryWriter(os.path.join(args.dir_name,args.env, 'tensorboard_train'))
    if args.option == 'surfpos':
        train_dataset = SurfPosData(args.data, args.list, validate=False, aug=args.data_aug, args=args)
        val_dataset = train_dataset
        ldm = SurfPosTrainer(args, train_dataset, val_dataset,train_writer)

    elif args.option == 'surfz':
        train_dataset = SurfZData(args.data, args.list, validate=False, aug=args.data_aug, args=args)
        val_dataset = SurfZData(args.data, args.list, validate=True, aug=False, args=args)
        ldm = SurfZTrainer(args, train_dataset, val_dataset,train_writer)

    elif args.option == 'GEdgePos':
        train_dataset = GEdgePosData(args.data, args.list, validate=False, aug=args.data_aug, args=args)
        val_dataset = GEdgePosData(args.data, args.list, validate=True, aug=False, args=args)
        ldm = GEdgePosTrainer(args, train_dataset, val_dataset,train_writer)
    elif args.option == 'GEdgeZ':
        train_dataset = GEdgeZData(args.data, args.list, validate=False, aug=args.data_aug, args=args)
        val_dataset = GEdgeZData(args.data, args.list, validate=True, aug=False, args=args)
        ldm = GEdgeZTrainer(args, train_dataset, val_dataset,train_writer)




    else:
        assert False, 'please choose between [surfpos, surfz, edgepos, edgez]'

    print('Start training...')
    
    for _ in range(args.train_nepoch):
        ldm.train_one_epoch()        
        if ldm.epoch % args.test_nepoch == 0:
            with torch.no_grad():
                ldm.test_val()

        if ldm.epoch % args.save_nepoch == 0:
            ldm.save_model()

    return


if __name__ == "__main__":
    run(args)
