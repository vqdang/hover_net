# Training and Inference Instructions

## Choosing the network

The model to use and the selection of other hyperparameters is selected in `config.py`. The models available are:
- HoVer-Net: `model/graph.py`
- DIST: `model/dist.py`
- Micro-Net: `model/micronet.py`
- DCAN: `model/dcan.py`
- SegNet: `model/segnet.py`
- U-Net: `model/unet.py`
- FCN8: `model/fcn8.py`

We also include a modification of HoVer-Net, where the distance maps from the nuclear centroids are used, instead of our proposed horizontal and vertical maps. This is also located at `model/graph.py`.

To use the above models, modify `mode` and `self.model_type` in `config.py` as follows:

- HoVer-Net: `mode='hover'` , `self.model_type=np_hv`
- HoVer-Net (distance map modification): `mode='hover'` , `self.model_type=np_dist`
- DIST: `mode='other'` , `self.model_type=dist`
- Micro-Net: `mode='other'` , `self.model_type=micronet`
- DCAN: `mode='other'` , `self.model_type=dcan`
- SegNet: `mode='other'` , `self.model_type=segnet`
- U-Net: `mode='other'` , `self.model_type=unet`
- FCN8: `mode='other'` , `self.model_type=fcn8`

## Modifying Hyperparameters

To modify hyperparameters, refer to `opt/`. For HoVer-Net, modify the script `opt/hover.py`, else modify `opt/other.py`. 

## Augmentation

To modify the augmentation pipeline, refer to `get_train_augmentors()` in `config.py`. Refer to [this webpage](https://tensorpack.readthedocs.io/modules/dataflow.imgaug.html)        for information on how to modify the augmentation parameters.

## Data Format

For instance segmentation, please store patches in a 4 dimensional numpy array with channels [RGB, inst]. Here, inst is the instance segmentation ground truth. I.e pixels range from 0 to N, where 0 is background and N is the number of nuclear instances for that particular image. <br/>
For simultaneous instance segmentation and classification, please store patches in a 5 dimensional numpy array with channels [RGB, inst, type]. Here, type is the ground truth of the nuclear type. I.e every pixel ranges from 0-K, where 0 is background and K is the number of classes.

## Current Repository Setup

The repository is currently set up to run simultaneous segmentation and classification. To perform nuclear classificaion, as well as segmentation, `self.type_classification` is set as `True`. Note, for this the nuclear type labels **must** be available in the 5th dimension of the input patches. Otherwise, set `self.type_classification = False`. 

## Training

To train the network, the command is: <br/>

`python train.py --gpu='gpu_ids'` <br/>
where gpu_id denotes which GPU will be used for training. For example, if we are using GPU number 0 and 1, the command is: <br/>
`python train.py --gpu='0,1'` <br/>

Before training, set in `config.py`:
- path to pretrained weights Preact-ResNet50. Download the weights [here](https://drive.google.com/open?id=187C9pGjlVmlqz-PlKW1K8AYfxDONrB0n).
- path to the data directories
- path where checkpoints will be saved

## Inference

To generate the network predictions, the command is: <br/>
`python infer.py --gpu='gpu_id'` <br/>
similar to the above. However, the code only support 1 GPU for inference. To run inference with GPU number 0, the command is
`python infer.py --gpu='0'` <br/>

Before running inference, set in `config.py`:
- path where the output will be saved
- path to data root directories
- path to model checkpoint

Download the HoVer-Net instance segmentation checkpoints trained on: [Kumar](https://drive.google.com/open?id=13S7VPu-4uRUQlgA5r-FO5NcA4q3nyif7), [CoNSeP](https://drive.google.com/open?id=1Yk62MtSOfopDSZT5g0ZaoaeAoRTeWUj4), [CPM-17](https://drive.google.com/open?id=1YdEfxhSt57gNL5sgWiXOZNLIkhMrnE_v)

Download the HoVer-Net instance segmentation and classification checkpoints, trained on: [CoNSeP](https://drive.google.com/open?id=1cM_iBtkUdpiblNx6Kc5Te2EKLC3hyS4b)

To obtain the final instance segmentation, use the command: <br/>
`python process.py` <br/>
for post-processing the network predictions.


