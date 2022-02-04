# HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images

A multiple branch network that performs nuclear instance segmentation and classification within a single network. The network leverages the horizontal and vertical distances of nuclear pixels to their centres of mass to separate clustered cells. A dedicated up-sampling branch is used to classify the nuclear type for each segmented instance. <br />

[Link](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045?via%3Dihub) to Medical Image Analysis paper. <br />

This is the official PyTorch implementation of HoVer-Net. For the original TensorFlow version of this code, please refer to [this branch](https://github.com/vqdang/hover_net/tree/tensorflow-final). The repository can be used for training HoVer-Net and to process image tiles or whole-slide images. As part of this repository, we supply model weights trained on the following datasets:

- [CoNSeP](https://www.sciencedirect.com/science/article/pii/S1361841519301045)
- [PanNuke](https://arxiv.org/abs/2003.10778)
- [MoNuSAC](https://ieeexplore.ieee.org/abstract/document/8880654)
- [Kumar](https://ieeexplore.ieee.org/abstract/document/7872382)
- [CPM17](https://www.frontiersin.org/articles/10.3389/fbioe.2019.00053/full)

Links to the checkpoints can be found in the inference description below.

![](docs/diagram.png)

## Set Up Environment

```
conda env create -f environment.yml
conda activate hovernet
pip install torch==1.6.0 torchvision==0.7.0
```

Above, we install PyTorch version 1.6 with CUDA 10.2. 

## Repository Structure

Below are the main directories in the repository: 

- `dataloader/`: the data loader and augmentation pipeline
- `docs/`: figures/GIFs used in the repo
- `metrics/`: scripts for metric calculation
- `misc/`: utils that are
- `models/`: model definition, along with the main run step and hyperparameter settings  
- `run_utils/`: defines the train/validation loop and callbacks 

Below are the main executable scripts in the repository:

- `config.py`: configuration file
- `dataset.py`: defines the dataset classes 
- `extract_patches.py`: extracts patches from original images
- `compute_stats.py`: main metric computation script
- `run_train.py`: main training script
- `run_infer.py`: main inference script for tile and WSI processing
- `convert_chkpt_tf2pytorch`: convert tensorflow `.npz` model trained in original repository to pytorch supported `.tar` format.

# Running the Code

## Training

### Data Format
For training, patches must be extracted using `extract_patches.py`. For instance segmentation, patches are stored as a 4 dimensional numpy array with channels [RGB, inst]. Here, inst is the instance segmentation ground truth. I.e pixels range from 0 to N, where 0 is background and N is the number of nuclear instances for that particular image. 

For simultaneous instance segmentation and classification, patches are stored as a 5 dimensional numpy array with channels [RGB, inst, type]. Here, type is the ground truth of the nuclear type. I.e every pixel ranges from 0-K, where 0 is background and K is the number of classes.

Before training:

- Set path to the data directories in `config.py`
- Set path where checkpoints will be saved  in `config.py`
- Set path to pretrained Preact-ResNet50 weights in `models/hovernet/opt.py`. Download the weights [here](https://drive.google.com/file/d/1KntZge40tAHgyXmHYVqZZ5d2p_4Qr2l5/view?usp=sharing).
- Modify hyperparameters, including number of epochs and learning rate in `models/hovernet/opt.py`.

### Usage and Options
 
Usage: <br />
```
  python run_train.py [--gpu=<id>] [--view=<dset>]
  python run_train.py (-h | --help)
  python run_train.py --version
```

Options:
```
  -h --help       Show this string.
  --version       Show version.
  --gpu=<id>      Comma separated GPU list.  
  --view=<dset>   Visualise images after augmentation. Choose 'train' or 'valid'.
```

Examples:

To visualise the training dataset as a sanity check before training use:
```
python run_train.py --view='train'
```

To initialise the training script with GPUs 0 and 1, the command is:
```
python run_train.py --gpu='0,1' 
```

## Inference

### Data Format
Input: <br />
- Standard images files, including `png`, `jpg` and `tiff`.
- WSIs supported by [OpenSlide](https://openslide.org/), including `svs`, `tif`, `ndpi` and `mrxs`.

Output: <br />
- Both image tiles and whole-slide images output a `json` file with keys:
    - 'bbox': bounding box coordinates for each nucleus
    - 'centroid': centroid coordinates for each nucleus
    - 'contour': contour coordinates for each nucleus 
    - 'type_prob': per class probabilities for each nucleus (default configuration doesn't output this)
    - 'type': prediction of category for each nucleus
- Image tiles output a `mat` file, with keys:
    - 'raw': raw output of network (default configuration doesn't output this)
    - 'inst_map': instance map containing values from 0 to N, where N is the number of nuclei
    - 'inst_type': list of length N containing predictions for each nucleus
 - Image tiles output a `png` overlay of nuclear boundaries on top of original RGB image
  
### Model Weights

Model weights obtained from training HoVer-Net as a result of the above instructions can be supplied to process input images / WSIs. Alternatively, any of the below pre-trained model weights can be used to process the data. These checkpoints were initially trained using TensorFlow and were converted using `convert_chkpt_tf2pytorch.py`. Provided checkpoints either are either trained for segmentation alone or for simultaneous segmentation and classification. Note, we do not provide a segmentation and classification model for CPM17 and Kumar because classification labels aren't available.

**IMPORTANT:** CoNSeP, Kumar and CPM17 checkpoints use the original model mode, whereas PanNuke and MoNuSAC use the fast model mode. Refer to the inference instructions below for more information. 

Segmentation and Classification:
- [CoNSeP checkpoint](https://drive.google.com/file/d/1FtoTDDnuZShZmQujjaFSLVJLD5sAh2_P/view?usp=sharing)
- [PanNuke checkpoint](https://drive.google.com/file/d/1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR/view?usp=sharing)
- [MoNuSAC checkpoint](https://drive.google.com/file/d/13qkxDqv7CUqxN-l5CpeFVmc24mDw6CeV/view?usp=sharing)

Segmentation Only:
- [CoNSeP checkpoint](https://drive.google.com/file/d/1BF0GIgNGYpfyqEyU0jMsA6MqcUpVQx0b/view?usp=sharing)
- [Kumar checkpoint](https://drive.google.com/file/d/1NUnO4oQRGL-b0fyzlT8LKZzo6KJD0_6X/view?usp=sharing) 
- [CPM17 checkpoint](https://drive.google.com/file/d/1lR7yJbEwnF6qP8zu4lrmRPukylw9g-Ms/view?usp=sharing) 

Access the entire checkpoint directory, along with a README on the filename description [here](https://drive.google.com/drive/folders/17IBOqdImvZ7Phe0ZdC5U1vwPFJFkttWp?usp=sharing).

If any of the above checkpoints are used, please ensure to cite the corresponding paper.

### Usage and Options

Usage: <br />
```
  run_infer.py [options] [--help] <command> [<args>...]
  run_infer.py --version
  run_infer.py (-h | --help)
```

Options:
```
  -h --help                   Show this string.
  --version                   Show version.

  --gpu=<id>                  GPU list. [default: 0]
  --nr_types=<n>              Number of nuclei types to predict. [default: 0]
  --type_info_path=<path>     Path to a json define mapping between type id, type name, 
                              and expected overlay color. [default: '']

  --model_path=<path>         Path to saved checkpoint.
  --model_mode=<mode>         Original HoVer-Net or the reduced version used in PanNuke / MoNuSAC, 'original' or 'fast'. [default: fast]
  --nr_inference_workers=<n>  Number of workers during inference. [default: 8]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 16]
  --batch_size=<n>            Batch size. [default: 128]
```

Tile Processing Options: <br />
```
   --input_dir=<path>     Path to input data directory. Assumes the files are not nested within directory.
   --output_dir=<path>    Path to output directory..

   --draw_dot             To draw nuclei centroid on overlay. [default: False]
   --save_qupath          To optionally output QuPath v0.2.3 compatible format. [default: False]
   --save_raw_map         To save raw prediction or not. [default: False]
```

WSI Processing Options: <br />
```
    --input_dir=<path>      Path to input data directory. Assumes the files are not nested within directory.
    --output_dir=<path>     Path to output directory.
    --cache_path=<path>     Path for cache. Should be placed on SSD with at least 100GB. [default: cache]
    --mask_dir=<path>       Path to directory containing tissue masks. 
                            Should have the same name as corresponding WSIs. [default: '']

    --proc_mag=<n>          Magnification level (objective power) used for WSI processing. [default: 40]
    --ambiguous_size=<int>  Define ambiguous region along tiling grid to perform re-post processing. [default: 128]
    --chunk_shape=<n>       Shape of chunk for processing. [default: 10000]
    --tile_shape=<n>        Shape of tiles for processing. [default: 2048]
    --save_thumb            To save thumb. [default: False]
    --save_mask             To save mask. [default: False]
```

The above command can be used from the command line or via an executable script. We supply two example executable scripts: one for tile processing and one for WSI processing. To run the scripts, first make them executable by using `chmod +x run_tile.sh` and `chmod +x run_tile.sh`. Then run by using `./run_tile.sh` and `./run_wsi.sh`.

Intermediate results are stored in cache. Therefore ensure that the specified cache location has enough space! Preferably ensure that the cache location is SSD.

Note, it is important to select the correct model mode when running inference. 'original' model mode refers to the method described in the original medical image analysis paper with a 270x270 patch input and 80x80 patch output. 'fast' model mode uses a 256x256 patch input and 164x164 patch output. Model checkpoints trained on Kumar, CPM17 and CoNSeP are from our original publication and therefore the 'original' mode **must** be used. For PanNuke and MoNuSAC, the 'fast' mode **must** be selected. The model mode for each checkpoint that we provide is given in the filename. Also, if using a model trained only for segmentation, `nr_types` must be set to 0.

`type_info.json` is used to specify what RGB colours are used in the overlay. Make sure to modify this for different datasets and if you would like to generally control overlay boundary colours.

As part of our tile processing implementation, we add an option to save the output in a form compatible with QuPath. 

Take a look on how to utilise the output in `examples/usage.ipynb`. 

## Overlaid Segmentation and Classification Prediction

<p float="left">
  <img src="docs/seg.gif" alt="Segmentation" width="870" />
</p>

Overlaid results of HoVer-Net trained on the CoNSeP dataset. The colour of the nuclear boundary denotes the type of nucleus. <br />
Blue: epithelial<br />
Red: inflammatory <br />
Green: spindle-shaped <br />
Cyan: miscellaneous

## Datasets

Download the CoNSeP dataset as used in our paper from [this link](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/). <br />
Download the Kumar, CPM-15, CPM-17 and TNBC datsets from [this link](https://drive.google.com/open?id=1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK).  <br />
Down

Ground truth files are in `.mat` format, refer to the README included with the datasets for further information. 

## Comparison to Original TensorFlow Implementation

Below we report the difference in segmentation results trained using this repository (PyTorch) and the results reported in the original manuscript (TensorFlow). 

Segmentation results on the Kumar dataset:
| Platform   | DICE       | PQ         | AJI       |
| -----------|----------- | -----------|-----------|
| TensorFlow | 0.8258     | 0.5971     | 0.6412    |
| PyTorch    | 0.8211     | 0.5904     | 0.6321    |

Segmentation results on CoNSeP dataset: 
| Platform   | DICE       | PQ         | AJI       |
| -----------|----------- | -----------|-----------|
| TensorFlow | 0.8525     | 0.5477     | 0.5995    |
| PyTorch    | 0.8504     | 0.5464     | 0.6009    |

Checkpoints to reproduce the above results can be found [here](https://drive.google.com/drive/folders/17IBOqdImvZ7Phe0ZdC5U1vwPFJFkttWp?usp=sharing).

Simultaneous Segmentation and Classification results on CoNSeP dataset: 
| Platform   | F1<sub>d</sub> | F1<sub>e</sub> | F1<sub>i</sub> | F1<sub>s</sub> | F1<sub>m</sub> |
| -----------|----------------| ---------------|----------------|----------------|----------------|
| TensorFlow | 0.748          | 0.635          | 0.631          | 0.566          | 0.426          |
| PyTorch    | 0.756          | 0.636          | 0.559          | 0.557          | 0.348          |


## Citation

If any part of this code is used, please give appropriate citation to our paper. <br />

BibTex entry: <br />
```
@article{graham2019hover,
  title={Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images},
  author={Graham, Simon and Vu, Quoc Dang and Raza, Shan E Ahmed and Azam, Ayesha and Tsang, Yee Wah and Kwak, Jin Tae and Rajpoot, Nasir},
  journal={Medical Image Analysis},
  pages={101563},
  year={2019},
  publisher={Elsevier}
}
```

## Authors

* [Quoc Dang Vu](https://github.com/vqdang)
* [Simon Graham](https://github.com/simongraham)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

Note that the PanNuke dataset is licensed under [Attribution-NonCommercial-ShareAlike 4.0 International](http://creativecommons.org/licenses/by-nc-sa/4.0/), therefore the derived weights for HoVer-Net are also shared under the same license. Please consider the implications of using the weights under this license on your work and it's licensing. 



