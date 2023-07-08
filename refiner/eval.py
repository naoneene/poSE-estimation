from __future__ import print_function, absolute_import, division

import argparse
import os
import time
import logging
import numpy as np
import pprint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths

from lib.core.config import config

from refiner.models.linear_model import LinearModel as refinet
from refiner.models.sem_gcn import SemGCN

from refiner.data import Human36M
from refiner.utils import adj_mx_from_skeleton


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_gcn', type=str, default=None, help='path to load a pretrained checkpoint')
    parser.add_argument('--load_refiner', type=str, default=None, help='path to load a pretrained checkpoint')
    parser.add_argument('--mode', type=str, default='eval', help='mode: [train, test]')
    args = parser.parse_args()

    return args


# Test the model
def test(model, refiner, test_loader):
    
    # Switch to evaluate mode
    model.eval()
    refiner.eval()

    preds = []
    with torch.no_grad():
        for i, (inputs, _, targets) in enumerate(test_loader):

            # Forward
            inputs = inputs.view(-1, inputs.size(1) // 3, 3)[:, :, 0:3]
            inputs = torch.cat([torch.zeros(inputs.size(0), 1, inputs.size(2)), inputs], axis=1).to(device)

            outputs = model(inputs)
            outputs = outputs - outputs[:, 0:1, :]
            out_gcn = outputs.view(-1, outputs.shape[1] * 3)
            del outputs

            outputs = refiner(out_gcn[:, 3:])

            output_rs = outputs.view(-1, outputs.shape[1] // 3, 3)
            output_rs[:, :, 0:2] = out_gcn.view(-1, out_gcn.shape[1] // 3, 3)[:, 1:, 0:2]
            outputs = output_rs.view(-1, output_rs.shape[1] * 3)

            out_arr = outputs.cpu().detach().numpy()

            for j in range(out_arr.shape[0]):
                preds.append(out_arr[j])

    preds = np.asarray(preds)

    error = test_loader.dataset.evaluate(preds)

    return error


def main():
    # Initialize
    args = parse_args()

    best_err = 1000

    log_dir = os.path.join('refiner/output')
    time_str = time.strftime('%Y-%m-%d-%H-%M')

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(log_dir,'%s_log_%s.log'%(args.mode, time_str)),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    # CUDNN related setting
    cudnn.fastest = config.CUDNN.FASTEST
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    h36m_skeleton_joints_group = [[2, 3], [5, 6], [1, 4], [0, 7], [8], [9, 10], [15, 16], [12, 13], [11, 14]]
    adj = adj_mx_from_skeleton(config.MODEL.NUM_JOINTS)
    model = SemGCN(adj, hid_dim=256, num_layers=4, p_dropout=0.0, nodes_group=h36m_skeleton_joints_group).to(device)
    refiner = refinet().to(device)

    # Model information
    logger.info(pprint.pformat(model))
    logger.info("Network total params: {:.2f}M".format(
        sum(params.numel() for params in model.parameters()) / 1000000.0 +
        sum(params.numel() for params in refiner.parameters()) / 1000000.0)
    )

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).to(device)
    refiner = torch.nn.DataParallel(refiner, device_ids=gpus).to(device)

    # Resume from a trained model
    ckpt = torch.load('refiner/output/test/best_496_gcn.pth.tar')
    model.load_state_dict(ckpt['state_dict'])

    ckpt = torch.load('refiner/output/test/best_363_ref1.pth.tar')
    refiner.load_state_dict(ckpt['state_dict'])

    # Load the dataset
    test_loader = torch.utils.data.DataLoader(
        dataset=Human36M(is_train=False),
        batch_size=256,
        shuffle=False,
        num_workers=config.WORKERS
    )

    test(model, refiner, test_loader)

if __name__ == '__main__':
    main()

