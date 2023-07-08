import argparse
import os
import pprint
import shutil
import matplotlib.pyplot as plt
import _init_paths

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision

from lib.core.config import config
from lib.core.config import update_config
from lib.core.config import update_dir
from lib.core.config import get_model_name
from lib.core.function import train, validate
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger
from lib.utils.balanced_parallel import DataParallelModel
from lib.utils.balanced_parallel import DataParallelCriterion

import lib.core.loss as loss
import lib.dataset as dataset
import lib.models as models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # General
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    
    # Update config
    update_config(args.cfg)

    # Training
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int,
                        default=config.WORKERS)

    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def main():
    # Initialize
    best_perf = 0.0
    best_model = False

    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir = create_logger(config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # CUDNN related setting
    cudnn.fastest = config.CUDNN.FASTEST
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(config, is_train=True)

    # Copy model file
    this_dir = os.path.dirname(__file__)

    shutil.copy2(
        args.cfg,
        final_output_dir
    )

    # Model information
    logger.info(pprint.pformat(model))
    logger.info("Net total params: {:.2f}M".format(
        sum(params.numel() for params in model.parameters()) / 1000000.))

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).to(device)

    # Loss and optimizer
    joint_loss_fn = eval('loss.'+config.LOSS.FN)
    criterion = joint_loss_fn(num_joints=config.MODEL.NUM_JOINTS, norm=config.LOSS.NORM)
    criterion = DataParallelCriterion(criterion).to(device)
    optimizer = get_optimizer(config, model)

    # Resume from a trained model
    if not(config.MODEL.RESUME is ''):
        checkpoint = torch.load(config.MODEL.RESUME)
        if 'epoch' in checkpoint.keys():
            config.TRAIN.BEGIN_EPOCH = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info('=> continue from pretrained model {}'.format(config.MODEL.RESUME))
        else:
            model.load_state_dict(checkpoint)
            logger.info('=> resume from pretrained model {}'.format(config.MODEL.RESUME))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    # Load the dataset
    train_loader = torch.utils.data.DataLoader(
        eval('dataset.'+config.DATASET.DATASET)(
            cfg=config,
            root=config.DATASET.ROOT,
            image_set=config.DATASET.TRAIN_SET,
            is_train=True
        ),
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        eval('dataset.'+config.DATASET.DATASET)(
            cfg=config,
            root=config.DATASET.ROOT,
            image_set=config.DATASET.TEST_SET,
            is_train=False
        ),
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        logger.info('Current learning rate: %s' %lr_scheduler.get_lr())

        # Train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch)

        # Evaluate on validation set
        accuracy = validate(epoch, valid_loader, model, final_output_dir)

        lr_scheduler.step()
        perf_indicator = 200. - accuracy if config.DATASET.DATASET == 'h36m' else accuracy

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
            logger.info('Found new best, performance: %s' %best_perf)
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': get_model_name(config),
            'state_dict': model.state_dict(),
            'perf': best_perf,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)


if __name__ == '__main__':
    main()
