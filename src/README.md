# XY-Net Training and Inference 

## Structure
- `model/` contains scripts that define the architecture of the model. Refer to `/model/graph.py` to understand the pipeline. 
- `loader/`contains scripts for data loading and self implemented augmentation functions.
- `metrics/`contains evaluation code. Refer to this repository for more information. To run the evaluation scripts, use `compute_stats.py`. 
- `misc/`contains util scripts. 
- `train.py` and `infer.py` are the training and inference scripts respectively.
- `process.py` is the post processing script for obtaining the final instances. 
- `extract_patches.py` is the patch extraction script. 
- `config.py` is the configuration file. Paths need to be changed accordingly.

## Training

To train the network, the command is: <br/>
`python train.py --gpu='gpu_ids'` <br/>
where gpu_id denote which GPU will be used for training. For example, if we are using GPU number 0 and 1, the command is: <br/>
`python train.py --gpu='0,1'` <br/>

Before training, set in `config.py`:
- path to pretrained weight Preact-ResNet50
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

To obtain final nuclei instance segmentation, use the command: <br/>
`python process.py` <br/>
for post-processing the network predictions.

## Modifying the Model

The model can be modified within `config.py`. For example, initial learning rate, batch size, number of epochs etc. We make use of the imgaug tensorpack library for augmentation. The augmentation pipeline can be modified here.


