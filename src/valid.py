import argparse
import pprint
import _init_paths

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from lib.core.config import config
from lib.core.config import update_config
from lib.core.config import update_dir
from lib.core.function import validate
from lib.utils.utils import create_logger

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

    # Testing
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    logger, final_output_dir = create_logger(config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # CUDNN related setting
    cudnn.fastest = config.CUDNN.FASTEST
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(config, is_train=False)

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).to(device)


    # Resume from a trained model
    if not(config.MODEL.RESUME is ''):
        checkpoint = torch.load(config.MODEL.RESUME)
        if 'epoch' in checkpoint.keys():
            config.TRAIN.BEGIN_EPOCH = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info('=> resume from pretrained model {}'.format(config.MODEL.RESUME))
        else:
            model.load_state_dict(checkpoint)
            logger.info('=> resume from pretrained model {}'.format(config.MODEL.RESUME))

    # Load the dataset
    train_loader = torch.utils.data.DataLoader(
        eval('dataset.'+config.DATASET.DATASET)(
            cfg=config,
            root=config.DATASET.ROOT,
            image_set=config.DATASET.TRAIN_SET,
            is_train=False
        ),
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
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

    # Evaluate on validation set
    #accuracy = validate(0, train_loader, model, final_output_dir)
    accuracy = validate(0, valid_loader, model, final_output_dir)
    #print(accuracy)


if __name__ == '__main__':
    main()

