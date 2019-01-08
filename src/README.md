# XY-Net Training and Inference 

## Structure
- `model/` contains scripts that define the architecture of the model. Refer to `/model/graph.py` to understand the pipeline. 
- `loader/`contains scripts for data loading and augmentation. We make use of the imgaug tensorpack library for augmentation. - `aug.py` contains self implemented augmentation functions. 
- `metrics/`contains evaluation code. Refer to this repository for more information. To run the evaluation scripts, use `compute_stats.py`. 
- `misc/`contains util scripts. 
- `train.py` and `infer.py` are the training and inference scripts respectively.
- `process.py` is the post processing script for obtaining the final instances. 
- `extract_patches.py` is the patch extraction script. 
- `config.py` is the configuration file, where the directories should be modified accordingly.

## Training

To train the network, the command is: <br/>
`python train.py --gpu='x,y'` <br/>
where x and y denote the GPU numbers to use. For example, if we are using GPU number 0 and 1, the command is: <br/>
`python train.py --gpu='0,1'` <br/>

