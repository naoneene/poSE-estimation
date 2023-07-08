import os
import logging
import numpy as np
import pickle as pkl

from torch.utils.data import Dataset

from lib.utils.prep_h36m import compute_similarity_transform, define_actions


H36M_NAMES = ['']*17
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[4]  = 'LHip'
H36M_NAMES[5]  = 'LKnee'
H36M_NAMES[6]  = 'LFoot'
H36M_NAMES[7] = 'Spine'
H36M_NAMES[8] = 'Neck'
H36M_NAMES[9] = 'Nose'
H36M_NAMES[10] = 'Head'
H36M_NAMES[11] = 'LShoulder'
H36M_NAMES[12] = 'LElbow'
H36M_NAMES[13] = 'LWrist'
H36M_NAMES[14] = 'RShoulder'
H36M_NAMES[15] = 'RElbow'
H36M_NAMES[16] = 'RWrist'

logger = logging.getLogger(__name__)


class Human36M(Dataset):
    def __init__(self, is_train):
        fname = 'refiner/data/train.pkl' if is_train else 'refiner/data/valid.pkl'

        self.is_train = is_train
        self.cam_input = True # True = randomly permutate the camera-reconstructed inputs

        self.data, self.data_cam, self.labels, self.data_mean, self.data_std, \
        self.cam_mean, self.cam_std, self.labels_mean, self.labels_std, self.img_name = self.get_db(fname)

        logger.info('=> loaded %s samples from %s' % (len(self.data), fname))

    def get_db(self, fname):
        with open(fname, 'rb') as anno_file:
            anno = pkl.load(anno_file)

        # Change to img-pelvis coordinate
        #for i in range(len(anno['img_pred'])):
            #anno['img_pred'][i] = (anno['img_pred'][i] - anno['img_pred'][i][0]) / 256 * 2000

        data = np.asarray(anno['img_pred'], dtype=np.float32).reshape(len(anno['img_pred']), -1)
        data_cam = np.asarray(anno['pred'], dtype=np.float32).reshape(len(anno['pred']), -1)
        labels = np.asarray(anno['gt'], dtype=np.float32).reshape(len(anno['gt']), -1)
        img_name = np.asarray(anno['image'])

        # Remove hip joint
        #data = np.delete(data, np.s_[0:3], axis=1)
        #data_cam = np.delete(data_cam, np.s_[0:3], axis=1)
        #labels = np.delete(labels, np.s_[0:3], axis=1)

        if os.path.exists('refiner/data/norm.pkl'):
            with open('refiner/data/norm.pkl', 'rb') as f:
                data_mean, data_std, cam_mean, cam_std, labels_mean, labels_std = pkl.load(f)
        elif self.is_train:
            data_mean, data_std = data.mean(axis=0), data.std(axis=0)
            cam_mean, cam_std = data_cam.mean(axis=0), data_cam.std(axis=0)
            labels_mean, labels_std = labels.mean(axis=0), labels.std(axis=0)

            with open('refiner/data/norm.pkl', 'wb') as f:
                pkl.dump((data_mean, data_std, cam_mean, cam_std, labels_mean, labels_std), f)

        # Z-score normalized
        #data = (data - data_mean) / data_std
        #data_cam = (data_cam - cam_mean) / cam_std

        # Linear normalized
        #data = data / 2000.
        #data_cam = data_cam / 2000.

        if self.is_train:
            #labels = (labels - labels_mean) / labels_std
            #labels = labels / 2000.
            if self.cam_input:
                rnd = np.random.permutation(labels.shape[0])
                #data, labels = data[rnd], labels[rnd]
                #data_cam, labels = data_cam[rnd], labels[rnd]

        return data, data_cam, labels, data_mean, data_std, cam_mean, cam_std, labels_mean, labels_std, img_name

    def __getitem__(self, index):
        return self.data[index], self.data_cam[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


    def evaluate(self, preds, actionwise=False):
        #preds = preds * self.cam_std + self.cam_mean
        #preds = preds * 2000.
        preds = preds.reshape((preds.shape[0], -1, 3))

        dist = []
        dist_align = []
        dist_14 = []
        dist_14_align = []
        dist_x = []
        dist_y = []
        dist_z = []
        dist_per_joint = []

        j14 = [0,1,2,3,4,5,6,10,11,12,13,14,15,16]

        if actionwise:
            acts = define_actions('All')
            dist_actions = {}
            dist_actions_align = {}
            for act in acts:
                dist_actions[act] = []
                dist_actions_align[act] = []

        for i in range(len(preds)):

            pre_3d_kpt = preds[i]
            gt_3d_kpt = self.labels[i].reshape((-1, 3))

            if (pre_3d_kpt.shape[0] == 16):
                pre_3d_kpt = np.concatenate((np.zeros([1, 3]), pre_3d_kpt.copy()), axis=0)
                gt_3d_kpt = np.concatenate((np.zeros([1, 3]), gt_3d_kpt.copy()), axis=0)
                if i == 0:
                    print("=> concatenated")

            joint_num = pre_3d_kpt.shape[0]
            if actionwise:
                img_name = self.img_name[i]
                action_idx = int(img_name[img_name.find('act')+4:img_name.find('act')+6]) - 2
                action = define_actions(action_idx)

            # align
            _, Z, T, b, c = compute_similarity_transform(gt_3d_kpt, pre_3d_kpt, compute_optimal_scale=True)
            pre_3d_kpt_align = (b * pre_3d_kpt.dot(T)) + c

            diff = (gt_3d_kpt - pre_3d_kpt)
            diff_align = (gt_3d_kpt - pre_3d_kpt_align)

            e_jt = []
            e_jt_align = []
            e_jt_14 = []
            e_jt_14_align = []
            e_jt_x = []
            e_jt_y = []
            e_jt_z = []

            for n_jt in range(0, joint_num):
                e_jt.append(np.linalg.norm(diff[n_jt])) if n_jt > 0 else e_jt.append(0)
                e_jt_align.append(np.linalg.norm(diff_align[n_jt])) #if n_jt > 0 else e_jt_align.append(0)
                e_jt_x.append(np.sqrt(diff[n_jt][0]**2)) if n_jt > 0 else e_jt_x.append(0)
                e_jt_y.append(np.sqrt(diff[n_jt][1]**2)) if n_jt > 0 else e_jt_y.append(0)
                e_jt_z.append(np.sqrt(diff[n_jt][2]**2)) if n_jt > 0 else e_jt_z.append(0)

            for jt in j14:
                e_jt_14.append(np.linalg.norm(diff[jt])) if n_jt > 0 else e_jt.append(0)
                e_jt_14_align.append(np.linalg.norm(diff_align[jt]))

            dist.append(np.array(e_jt).mean())
            dist_align.append(np.array(e_jt_align).mean())
            dist_14.append(np.array(e_jt_14).mean())
            dist_14_align.append(np.array(e_jt_14_align).mean())
            dist_x.append(np.array(e_jt_x).mean())
            dist_y.append(np.array(e_jt_y).mean())
            dist_z.append(np.array(e_jt_z).mean())
            dist_per_joint.append(np.array(e_jt))

            if actionwise:
                dist_actions[action].append(np.array(e_jt).mean())
                dist_actions_align[action].append(np.array(e_jt_align).mean())

        per_joint_error = np.array(dist_per_joint).mean(axis=0).tolist()
        joint_names = H36M_NAMES

        logger.info('======== JOINTS ========')
        for idx in range(len(joint_names)):
            logger.info('%s: %s' % (joint_names[idx], per_joint_error[idx]))
        if actionwise:
            logger.info('======= ACTIONS ========')
            for k, v in dist_actions.items():
                logger.info('%s: %s' % ((k, np.array(v).mean())))
            logger.info('========================')

            logger.info('========================')
            for k, v in dist_actions_align.items():
                logger.info('%s: %s' % ((k, np.array(v).mean())))
            logger.info('========================')

        results = {
            'hm36_17j': np.asarray(dist).mean(),
            'hm36_17j_align': np.array(dist_align).mean(),
            'hm36_17j_14': np.asarray(dist_14).mean(),
            'hm36_17j_14_al': np.array(dist_14_align).mean(),
            'hm36_17j_x': np.array(dist_x).mean(),
            'hm36_17j_y': np.array(dist_y).mean(),
            'hm36_17j_z': np.array(dist_z).mean(),
        }

        logger.info('======= RESULTS ========')
        for k, v in results.items():
            logger.info('%s: %s' % (k, np.array(v).mean()))
        logger.info('========================')

        return np.asarray(dist_z).mean()
