import os
import logging
import numpy as np
import pickle as pkl

n = 1
fname = 'refiner/data/valid.pkl'
with open(fname, 'rb') as anno_file:
    anno = pkl.load(anno_file)

data = np.asarray(anno['pred'], dtype=np.float32).reshape(len(anno['pred']), -1)
labels = np.asarray(anno['gt'], dtype=np.float32).reshape(len(anno['gt']), -1)

print('loaded %s samples from %s' % (len(data), fname))
print(anno['pred'][n])
print((anno['img_pred'][n] - anno['img_pred'][n][0]) / 256 * 2000)
print(anno['gt'][n])
