import logging
import time

import numpy as np
import torch

from lib.utils.img_utils import trans_coords_from_patch_to_org_3d
from lib.core.loss import get_joint_location_result
from lib.core.config import config

logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Switch to train mode
    model.train()
    end = time.time()

    for i, (batch_data, batch_label, batch_label_weight, meta) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()
        batch_label_weight = batch_label_weight.cuda()

        batch_size = batch_data.size(0)
        
        # Forward
        preds = model(batch_data)
        loss = criterion(preds, batch_label, batch_label_weight)
        del batch_data, batch_label, batch_label_weight, preds

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record loss
        losses.update(loss.item(), batch_size)
        del loss
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, len(train_loader),
                      batch_time=batch_time,
                      speed=batch_size/batch_time.val,
                      data_time=data_time,
                      loss=losses)
            logger.info(msg)


def validate(epoch, val_loader, model, final_output_path):
    print("Validation stage")

    # Switch to evaluate mode
    model.eval()

    preds_in_patch_with_score = []
    with torch.no_grad():
        for i, (batch_data, batch_label, batch_label_weight, meta) in enumerate(val_loader):
            batch_data = batch_data.cuda()
            batch_label = batch_label.cuda()
            batch_label_weight = batch_label_weight.cuda()

            # Compute output
            preds = model(batch_data)

            preds_in_patch_with_score.append(get_joint_location_result(256, 256, preds))
            
            del batch_data, batch_label, batch_label_weight, preds

        temp = np.asarray(preds_in_patch_with_score)

        # Dirty solution for partial batches
        if len(temp.shape) < 2:
            tp = np.zeros(((temp.shape[0] - 1) * temp[0].shape[0] + temp[-1].shape[0], temp[0].shape[1], temp[0].shape[2]))

            start = 0
            end = temp[0].shape[0]

            for t in temp:
                tp[start:end] = t
                start = end
                end += t.shape[0]

            temp = tp
        else:
            temp = temp.reshape((temp.shape[0] * temp.shape[1], temp.shape[2], temp.shape[3]))

        preds_in_patch_with_score = temp[0: len(val_loader.dataset)]

        # Evaluate
        imdb = val_loader.dataset.db
        preds_in_img_with_score = []
        
        for n_sample in range(len(val_loader.dataset)):
            preds_in_img_with_score.append(
                trans_coords_from_patch_to_org_3d(preds_in_patch_with_score[n_sample], imdb[n_sample]['center_x'],
                                              imdb[n_sample]['center_y'], imdb[n_sample]['width'],
                                              imdb[n_sample]['height'], 256, 256,
                                              2000, 2000))
        preds_in_img_with_score = np.asarray(preds_in_img_with_score)

        name_value, perf = val_loader.dataset.evaluate(preds_in_img_with_score.copy(), preds_in_patch_with_score.copy(), final_output_path)
        for name, value in name_value:
            logger.info('Epoch[%d] Validation-%s %f', epoch, name, value)

        return perf


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
