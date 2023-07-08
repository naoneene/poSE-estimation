import argparse
import pprint
import time
import _init_paths
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed

from lib.core.config import config
from lib.core.config import update_config
from lib.core.loss import get_joint_location_result
from lib.utils.utils import create_logger

import lib.dataset as dataset
import lib.models as models

from refiner.models.linear_model import LinearModel as refinet
from refiner.models.sem_gcn import SemGCN

from refiner.data import Human36M
from refiner.utils import adj_mx_from_skeleton


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate entire network')
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

    logger, final_output_dir = create_logger(config, args.cfg, 'evaluation')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # CUDNN related setting
    cudnn.fastest = config.CUDNN.FASTEST
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # poSEnet
    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(config, is_train=False)
    # SemGCN
    joints_group = [[2, 3], [5, 6], [1, 4], [0, 7], [8], [9, 10], [15, 16], [12, 13], [11, 14]]
    adj = adj_mx_from_skeleton(config.MODEL.NUM_JOINTS)
    gcn = SemGCN(adj, hid_dim=128, num_layers=4, p_dropout=0.0, nodes_group=joints_group)
    # Linear
    refiner = refinet()

    # Model information
    logger.info(pprint.pformat(model))
    logger.info(pprint.pformat(gcn))
    logger.info(pprint.pformat(refiner))
    logger.info("Network total params: {:.2f}M".format(
        sum(params.numel() for params in model.parameters()) / 1000000.0 +
        sum(params.numel() for params in gcn.parameters()) / 1000000.0 +
        sum(params.numel() for params in refiner.parameters()) / 1000000.0
    ))

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).to(device)
    gcn = torch.nn.DataParallel(gcn, device_ids=gpus).to(device)
    refiner = torch.nn.DataParallel(refiner, device_ids=gpus).to(device)

    # Resume from a trained model
    logger.info('=> loading pretrained weights...')
    checkpoint = torch.load(config.MODEL.RESUME)
    model.load_state_dict(checkpoint)
    model.eval()

    checkpoint = torch.load(config.MODEL.RESUME_GCN)
    gcn.load_state_dict(checkpoint['state_dict'])
    gcn.eval()

    checkpoint = torch.load(config.MODEL.RESUME_REF)
    refiner.load_state_dict(checkpoint['state_dict'])
    refiner.eval()

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
    test_loader = torch.utils.data.DataLoader(
        dataset=Human36M(is_train=False),
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS
    )

    # Evaluate on validation set
    end = time.time()
    preds = []
    with torch.no_grad():
        for i, (batch_data, batch_label, batch_label_weight, meta) in enumerate(valid_loader):
        #for i, (out_psn, _, targets) in enumerate(test_loader):
            batch_end = time.time()
            batch_data = batch_data.cuda()
            batch_label = batch_label.cuda()

            # Compute output
            outputs = model(batch_data)
            out_psn = torch.from_numpy(np.asarray(get_joint_location_result(256, 256, outputs)[:, :, 0:3], dtype=np.float32))
            #out_psn = (outputs - outputs[:, 0:1, :]) / 256 * 2000.
            del outputs

            #out_psn = out_psn.view(-1, out_psn.size(1) // 3, 3)[:, :, :3].cuda()

            outputs = gcn(out_psn)
            outputs -= outputs[:, 0:1, :]
            out_gcn = outputs.view(-1, outputs.shape[1] * 3)
            del outputs

            outputs = refiner(out_gcn[:, 3:]).view(out_gcn.size(0), -1, 3)
            outputs = torch.cat([torch.zeros(outputs.size(0), 1, outputs.size(2)).cuda(), outputs], axis=1)
            outputs[:, :, 0:2] = out_gcn.view(-1, out_gcn.shape[1] // 3, 3)[:, :, 0:2]

            out_arr = outputs.cpu().detach().numpy()

            for j in range(out_arr.shape[0]):
                preds.append(out_arr[j])
            
            del batch_data, batch_label, outputs
            batch_time = time.time() - batch_end

            if i % config.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time:.3f}s'.format(0, i, len(valid_loader), batch_time=batch_time)
                print(msg)

    preds = np.asarray(preds)

    calc_time = time.time() - end
    print('Preds time: {}m{}s'.format(calc_time // 60, calc_time - calc_time // 60))
    acc = test_loader.dataset.evaluate(preds)


if __name__ == '__main__':
    main()

