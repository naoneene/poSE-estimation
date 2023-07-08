import argparse
import os
import time
import logging
import shutil
import numpy as np
import pprint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
#import lib.models.refiner as refinet

#from lib.models.refiner import weight_init
from lib.core.config import config
from lib.core.loss import BoneLoss

from refiner.models.linear_model import LinearModel as refinet
from refiner.data import Human36M
from refiner.utils import lr_decay, AverageMeter, save_ckpt, AdamW, CyclicLR
#from refiner.RuberLoss import RuberLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
    parser.add_argument('--load', type=str, default=None, help='path to load a pretrained checkpoint')
    parser.add_argument('--resume', type=bool, default=False, help='seperate pretrained checkpoint and pretrained model')
    parser.add_argument('--mode', type=str, default='train', help='mode: [train, test]')
    parser.add_argument('--num_epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay', type=int, default=10000, help='# steps of lr decay')
    parser.add_argument('--lr_gamma', type=float, default=0.96)
    args = parser.parse_args()

    return args


# Train the model
def train(model, train_loader, optimizer, global_step, lr_now, criterion, args, logger): #lr_now <-> scheduler
    epoch_loss = AverageMeter()

    # Switch to train mode
    model.train()
    start = time.time()

    for i, (_, inputs, targets) in enumerate(train_loader):
        global_step += 1
        
        #lr_now = scheduler.get_lr()
        if global_step % args.lr_decay == 0 or global_step == 1:
            lr_now = lr_decay(optimizer, global_step, args.lr, args.lr_decay, args.lr_gamma)

        inputs = inputs[:, 3:].to(device)
        targets = targets[:, 3:].to(device)
        #inputs = inputs.view(-1, inputs.size(1) // 3, 3)[:, 1:, 2].to(device)
        #targets = targets.view(-1, targets.size(1) // 3, 3)[:, 1:, 2].to(device)

        # Forward
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        epoch_loss.update(loss.item(), targets.size(0))
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
        optimizer.step()
        #scheduler.step()

    logger.info('Avg Loss: %.5f' % epoch_loss.avg)

    return global_step, lr_now#[-1]


# Test the model
def test(model, test_loader, mode):
    
    # Switch to evaluate mode
    model.eval()

    preds = []
    with torch.no_grad():
        for i, (_, inputs, targets) in enumerate(test_loader):

            # Forward
            inputs = inputs[:, 3:].to(device)
            #inputs = inputs.view(-1, inputs.size(1) // 3, 3).to(device)
            #inputs_cat = inputs[:, 1:, :2]

            outputs = model(inputs).view(inputs.size(0), -1, 3)
            #outputs = model(inputs[:, 1:, 2])
            #outputs = torch.cat([inputs_cat, outputs.view(-1, outputs.size(1), 1)], axis=2)
            outputs = torch.cat([torch.zeros(outputs.size(0), 1, outputs.size(2)).to(device), outputs], axis=1)

            #if mode == 'test':
                #output_rs = outputs.view(-1, inputs.shape[1] // 3, 3)
                #output_rs[:, :, 0:2] = inputs.view(-1, inputs.shape[1] // 3, 3)[:, :, 0:2]
                #outputs = output_rs.view(-1, output_rs.shape[1] * 3)

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

    model = refinet().to(device)
    #model = refinet.get_refiner(config, is_train=True, weights=None, dim=3).to(device)
    #model.apply(weight_init)

    # Model information
    logger.info(pprint.pformat(model))
    logger.info("Network total params: {:.2f}M".format(
        sum(params.numel() for params in model.parameters()) / 1000000.0))

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).to(device)

    # Loss and optimizer
    criterion = nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean').to(device)
    #criterion = RuberLoss(size_average=None, reduce=None, reduction='mean').to(device)
    criterion_2 = BoneLoss(config.MODEL.NUM_JOINTS, size_average=None, reduce=None, norm=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Resume from a trained model
    if args.load and args.resume:
        logger.info("=> continue from '{}'".format(args.load))
        ckpt = torch.load(args.load)

        start_epoch = ckpt['epoch']
        best_err = ckpt['err']
        global_step = ckpt['step']
        lr_now = ckpt['lr']

        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info("=> checkpoint loaded (epoch: {} | err: {})".format(start_epoch, best_err))
    elif args.load:
        logger.info("=> loading pretrained from '{}'".format(args.load))
        ckpt = torch.load(args.load)
        model.load_state_dict(ckpt['state_dict'])

    # Load the dataset
    train_loader = torch.utils.data.DataLoader(
        dataset=Human36M(is_train=True),
        batch_size=256,
        shuffle=True,
        num_workers=config.WORKERS
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=Human36M(is_train=False),
        batch_size=256,
        shuffle=False,
        num_workers=config.WORKERS
    )
    
    lr_scheduler = CyclicLR(
        optimizer, base_lr=args.lr*0.1, max_lr=args.lr, step_size_up=args.lr_decay,
        mode='exp_range', gamma=args.lr_gamma, cycle_momentum=False
    )

    if args.mode == 'train':
        logger.info("Starting training for {} epoch(s)".format(args.num_epochs))

        global_step = 0
        lr_now = args.lr


        for epoch in range(args.num_epochs):
            logger.info("Current learning rate: [%.6f]" % (lr_now))
            logger.info("Epoch: [%s|%s]" % (epoch, args.num_epochs))

            global_step, lr_now = train(model, train_loader, optimizer, global_step,
                                        lr_now, criterion, args, logger) #lr_now <-> lr_scheduler
            
            logger.info('Evaluation')

            error = test(model, test_loader, args.mode)

            is_best = error < best_err
            best_err = min(error, best_err)
            
            save_ckpt({'epoch': epoch + 1,
                       'lr': lr_now,
                       'step': global_step,
                       'state_dict': model.state_dict(),
                       'err': best_err,
                       'optimizer': optimizer.state_dict()},
                      ckpt_path=log_dir,
                      is_best=is_best)

            if is_best:
                logger.info('Found new best, error: %s' % error)

    elif args.mode == 'test':
        test(model, test_loader, args.mode)
    else:
        print('Error!')

if __name__ == '__main__':
    main()
