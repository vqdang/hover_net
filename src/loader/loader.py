
import random

import matplotlib.pyplot as plt
import numpy as np
from tensorpack.dataflow import (AugmentImageComponent, AugmentImageComponents,
                                 BatchData, BatchDataByShape, CacheData,
                                 PrefetchDataZMQ, RNGDataFlow, RepeatedData)

####
class DatasetSerial(RNGDataFlow):
    """
    Produce ``(image, label)`` pair, where 
        ``image`` has shape (H, W, C (3(BGR))) and ranges in [0,255].
        ``Label`` is an int image of shape (H, W, classes) in range [0, no.classes - 1].
        ``dist_sample
    """

    def __init__(self, path_list):
        self.path_list = path_list
    ##
    def size(self):
        return len(self.path_list)
    ##
    def get_data(self):
        idx_list = list(range(0, len(self.path_list)))
        random.shuffle(idx_list)
        for idx in idx_list:

            data = np.load(self.path_list[idx])

            # TODO: reverse here after debug RGB-ID
            # split stack channel into image and label
            img = data[...,:3]
            ann = data[...,4:] # instance ID map
           
            img = img.astype('uint8')            
            yield [img, ann]

#### 
def valid_generator(ds, shape_aug=None, input_aug=None, label_aug=None, batch_size=16, nr_procs=1):
    ### augment both the input and label
    ds = ds if shape_aug is None else AugmentImageComponents(ds, shape_aug, (0, 1), copy=True)
    ### augment just the input
    ds = ds if input_aug is None else AugmentImageComponent(ds, input_aug, index=0, copy=False)
    ### augment just the output
    ds = ds if label_aug is None else AugmentImageComponent(ds, label_aug, index=0, copy=True)
    #
    ds = BatchData(ds, batch_size, remainder=True)
    ds = CacheData(ds) # cache all inference images 
    # TODO: check sampling correct or not
    ds = PrefetchDataZMQ(ds, nr_procs)
    return ds

####
def train_generator(ds, shape_aug=None, input_aug=None, label_aug=None, batch_size=16, nr_procs=8):
    ### augment both the input and label
    ds = ds if shape_aug is None else AugmentImageComponents(ds, shape_aug, (0, 1), copy=True)
    ### augment just the input i.e index 0 within each yield of DatasetSerial
    ds = ds if input_aug is None else AugmentImageComponent(ds, input_aug, index=0, copy=False)
    ### augment just the output i.e index 1 within each yield of DatasetSerial
    ds = ds if label_aug is None else AugmentImageComponent(ds, label_aug, index=1, copy=True)
    #
    ds = BatchDataByShape(ds, batch_size, idx=0)
    ds = PrefetchDataZMQ(ds, nr_procs)
    return ds

#### 
def visualize(datagen, batch_size, view_size=4):
    """
    Read the batch from 'datagen' and display 'view_size' number of
    of images and their corresponding Ground Truth
    """
    def prep_imgs(img, ann):
        cmap = plt.get_cmap('viridis')
        ann_chs = np.dsplit(ann, ann.shape[-1])
        for i, ch in enumerate(ann_chs):
            # take RGB from RGBA heat map
            ch = cmap(np.squeeze(ch))
            ann_chs[i] = ch[...,:3] 
        img = img.astype('float32') / 255.0
        prepped_img = np.concatenate([img] + ann_chs, axis=1)
        return prepped_img

    assert view_size <= batch_size, 'Number of displayed images must <= batch size'
    ds = RepeatedData(datagen, -1)    
    ds.reset_state()
    for imgs, segs in ds.get_data():
        for idx in range (0, view_size):
            displayed_img = prep_imgs(imgs[idx], segs[idx])
            plt.subplot(view_size, 1, idx+1)
            plt.imshow(displayed_img)
        plt.show()
    return
###

###########################################################################
