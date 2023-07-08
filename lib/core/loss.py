import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import math

# Bone loss
class BoneLoss(nn.Module):
    def __init__(self, num_joints, size_average=True, reduce=True, norm=False):
        super(BoneLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.num_joints = num_joints
        self.norm = norm

        self.pairs = np.array([[0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], \
                               [8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], \
                               [1, 4], [2, 5], [3, 6], [14, 11], [15, 12], [16, 13]], dtype=np.int)
        self.gamma = 10.
        self.loss_fn = nn.SmoothL1Loss()
        self.cosine_similarity = nn.CosineEmbeddingLoss()

    def forward(self, inputs, targets):
        #inputs = inputs.view(-1, inputs.shape[1] // 3, 3)
        #targets = targets.view(-1, targets.shape[1] // 3, 3)

        preds = torch.zeros([inputs.size(0), len(self.pairs), 3], dtype=torch.float32).cuda()
        gt = torch.zeros([targets.size(0), len(self.pairs), 3], dtype=torch.float32).cuda()
        labels = torch.ones([3], dtype=torch.float32).cuda()

        _assert_no_grad(targets)
        _assert_no_grad(gt)

        for i in range(len(self.pairs)):
            preds[:, i, :] = (preds[:, self.pairs[i][0], :] - preds[:, self.pairs[i][0], :])
            gt[:, i, :] = (gt[:, self.pairs[i][0], :] - gt[:, self.pairs[i][0], :])

        if self.norm:
            preds = preds / torch.norm(preds, 1)
            gt = gt / torch.norm(gt, 1)

        return self.gamma * self.cosine_similarity(preds, gt, labels) * self.loss_fn(preds, gt)

# Joint loss
def weighted_mse_loss(input, target, weights, size_average, norm=False):

    if norm:
        input = input / torch.norm(input, 1)
        target = target / torch.norm(target, 1)

    out = (input - target) ** 2
    out = out * weights
    if size_average:
        return out.sum() / len(input)
    else:
        return out.sum()

def weighted_l1_loss(input, target, weights, size_average, norm=False):

    if norm:
        input = input / torch.norm(input, 1)
        target = target / torch.norm(target, 1)

    out = torch.abs(input - target)
    out = out * weights
    if size_average:
        return out.sum() / len(input)
    else:
        return out.sum()

def weighted_smooth_l1_loss(input, target, weights, size_average, norm=False):

    if norm:
        input = input / torch.norm(input, 1)
        target = target / torch.norm(target, 1)

    diff = input - target
    abs = torch.abs(diff)
    out = torch.where(abs<1., 0.5*diff**2, abs-0.5)

    out = out * weights
    if size_average:
        return out.sum() / len(input)
    else:
        return out.sum()

# Loss functions classes
class L2JointLocationLoss(nn.Module):
    def __init__(self, num_joints, size_average=True, reduce=True, norm=False):
        super(L2JointLocationLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.num_joints = num_joints
        self.norm = norm

    def forward(self, preds, *args):
        gt_joints = args[0]
        gt_joints_vis = args[1]

        num_joints = int(gt_joints_vis.shape[1] / 3)
        hm_width = preds.shape[-1]
        hm_height = preds.shape[-2]
        hm_depth = preds.shape[-3] // self.num_joints

        print(num_joints)

        pred_jts = softmax_integral_tensor(preds, self.num_joints, self.output_3d, hm_width, hm_height, hm_depth)

        _assert_no_grad(gt_joints)
        _assert_no_grad(gt_joints_vis)
        return weighted_mse_loss(pred_jts, gt_joints, gt_joints_vis, self.size_average, self.norm)

class L1JointLocationLoss(nn.Module):
    def __init__(self, num_joints, size_average=False, reduce=True, norm=False):
        super(L1JointLocationLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.num_joints = num_joints
        self.norm = norm

    def forward(self, preds, *args):
        gt_joints = args[0]
        gt_joints_vis = args[1]

        hm_width = preds.shape[-1]
        hm_height = preds.shape[-2]
        hm_depth = preds.shape[-3] // self.num_joints

        pred_jts = softmax_integral_tensor(preds, self.num_joints, True, hm_width, hm_height, hm_depth)

        _assert_no_grad(gt_joints)
        _assert_no_grad(gt_joints_vis)
        return weighted_l1_loss(pred_jts, gt_joints, gt_joints_vis, self.size_average, self.norm)

class SmoothL1JointLocationLoss(nn.Module):
    def __init__(self, num_joints, size_average=True, reduce=True, norm=False):
        super(SmoothL1JointLocationLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.num_joints = num_joints
        self.norm = norm

    def forward(self, preds, *args):
        gt_joints = args[0]
        gt_joints_vis = args[1]

        hm_width = preds.shape[-1]
        hm_height = preds.shape[-2]
        hm_depth = preds.shape[-3] // self.num_joints

        pred_jts = softmax_integral_tensor(preds, self.num_joints, True, hm_width, hm_height, hm_depth)

        _assert_no_grad(gt_joints)
        _assert_no_grad(gt_joints_vis)
        return weighted_smooth_l1_loss(pred_jts, gt_joints, gt_joints_vis, self.size_average, self.norm)

# Soft argmax
def generate_3d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim, z_dim):
    assert isinstance(heatmaps, torch.Tensor)

    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, z_dim, y_dim, x_dim))

    accu_x = heatmaps.sum(dim=2)
    accu_x = accu_x.sum(dim=2)
    accu_y = heatmaps.sum(dim=2)
    accu_y = accu_y.sum(dim=3)
    accu_z = heatmaps.sum(dim=3)
    accu_z = accu_z.sum(dim=3)

    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(float(x_dim)), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(float(y_dim)), devices=[accu_y.device.index])[0]
    accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(float(z_dim)), devices=[accu_z.device.index])[0]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    return accu_x, accu_y, accu_z

def softmax_integral_tensor(preds, num_joints, output_3d, hm_width, hm_height, hm_depth):
    # Global soft max
    preds = preds.reshape((preds.shape[0], num_joints, -1))
    preds = F.softmax(preds, 2)

    # Integrate heatmap into joint location
    if output_3d:
        x, y, z = generate_3d_integral_preds_tensor(preds, num_joints, hm_width, hm_height, hm_depth)
    else:
        assert 0, 'Not Implemented!'
    x = x / float(hm_width) - 0.5
    y = y / float(hm_height) - 0.5
    z = z / float(hm_depth) - 0.5
    preds = torch.cat((x, y, z), dim=2)
    preds = preds.reshape((preds.shape[0], num_joints * 3))
    return preds

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

# Sub functions
def generate_joint_location_label(patch_width, patch_height, joints, joints_vis):
    joints[:, 0] = joints[:, 0] / patch_width - 0.5
    joints[:, 1] = joints[:, 1] / patch_height - 0.5
    joints[:, 2] = joints[:, 2] / patch_width

    joints = joints.reshape((-1))
    joints_vis = joints_vis.reshape((-1))
    return joints, joints_vis

def reverse_joint_location_label(patch_width, patch_height, joints):
    joints = joints.reshape((joints.shape[0] // 3, 3))

    joints[:, 0] = (joints[:, 0] + 0.5) * patch_width
    joints[:, 1] = (joints[:, 1] + 0.5) * patch_height
    joints[:, 2] = joints[:, 2] * patch_width
    return joints

def get_joint_location_result(patch_width, patch_height, preds):
    hm_width = preds.shape[-1]
    hm_height = preds.shape[-2]

    hm_depth = hm_width
    num_joints = preds.shape[1] // hm_depth

    pred_jts = softmax_integral_tensor(preds, num_joints, True, hm_width, hm_height, hm_depth)
    coords = pred_jts.detach().cpu().numpy()
    coords = coords.astype(float)
    coords = coords.reshape((coords.shape[0], int(coords.shape[1] / 3), 3))
    # project to original image size
    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * patch_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * patch_height
    coords[:, :, 2] = coords[:, :, 2] * patch_width
    scores = np.ones((coords.shape[0], coords.shape[1], 1), dtype=float)

    # add score to last dimension
    coords = np.concatenate((coords, scores), axis=2)

    return coords

def get_label_func():
    return generate_joint_location_label

def get_result_func():
    return get_joint_location_result

