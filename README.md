# HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images

A multiple branch network that performs nuclear instance segmentation and classification within a single network. The network leverages the horizontal and vertical distances of nuclear pixels to their centres of mass to separate clustered cells. A dedicated up-sampling branch is used to classify the nuclear type for each segmented instance. <br />

[Link](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045?via%3Dihub) to Medical Image Analysis paper. <br />

This is the official PyTorch implementation of HoVer-Net. For a TensorFlow version of this code, please refer to the [original repository](https://github.com/vqdang/hover_net). The repository can be used for training HoVer-Net and to process image tiles or whole-slide images. As part of this repository, we supply model weights trained on the following datasets:

- CoNSeP
- PanNuke
- MoNuSAC
- Kumar
- CPM17

![](docs/diagram.png)

## Set Up Environment

```
conda create --name hovernet python=3.6
conda activate hovernet
pip install -r requirements.txt
```

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

# Running the Code

## Training

### Data Format
For training, patches must be extracted using `extract_patches.py`. For instance segmentation, patches are stored as a 4 dimensional numpy array with channels [RGB, inst]. Here, inst is the instance segmentation ground truth. I.e pixels range from 0 to N, where 0 is background and N is the number of nuclear instances for that particular image. 

For simultaneous instance segmentation and classification, patches are stored as a 5 dimensional numpy array with channels [RGB, inst, type]. Here, type is the ground truth of the nuclear type. I.e every pixel ranges from 0-K, where 0 is background and K is the number of classes.

Before training:

- Set path to the data directories in `config.py`
- Set path where checkpoints will be saved  in `config.py`
- Set path to pretrained weights Preact-ResNet50  in `models/hovernet/opt.py`. Download the weights [here](https://drive.google.com/open?id=187C9pGjlVmlqz-PlKW1K8AYfxDONrB0n).
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
- Standard image input
    - Output1
    - Output2
- WSI input
    - Output1
    - Output2
  
### Model Weights

Model weights obtained from training HoVer-Net as a result of the above instructions can be supplied to process input images / WSIs. Alternatively, any of the below pre-trained model weights can be used to process the data.

- [CoNSeP checkpoint](link)
- [PanNuke checkpoint](link)
- [MoNuSAC checkpoint](link)
- [Kumar checkpoint](link) (only instance segmentation)
- [CPM17 checkpoint](link) (only instance segmentation)

### Usage and Options

Usage: <br />
```
  run_infer.py [--gpu=<id>] [--model_mode=<mode>] [--run_mode=<mode>] [--nr_types=<n>] [--model_path=<path>] \
               [--nr_inference_workers=<n>] [--nr_post_proc_workers=<n>] [--batch_size=<n>] \
               [--ambiguous_size=<n>] [--chunk_shape=<n>] [--tile_shape=<n>] [--wsi_proc_mag=<n>] \
               [--cache_path=<path>] [--input_dir=<path>] [--input_msk_dir=<path>] \
               [--output_dir=<path>] [--patch_input_shape=<n>] [--patch_output_shape=<n>] \
               [--save_raw_map=<BOOL>]
  run_infer.py (-h | --help)
  run_infer.py --version
```

Options: <br />
```
  -h --help                   Show this string.
  --version                   Show version.
  --gpu=<id>                  GPU list. [default: 0]
  --run_mode=<mode>           Inference mode. 'tile' or 'wsi'. [default: tile]
  --nr_types=<n>              Number of nuclei types to predict. [default: 0]
  --model_path=<path>         Path to saved checkpoint.
  --model_mode=<mode>         Original HoVer-Net or the reduced version in Pannuke, 'None' or 'pannuke'. [default: pannuke]
  --nr_inference_workers=<n>  Number of workers during inference. [default: 8]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 16]
  --batch_size=<n>            Batch size. [default: 128]
  --ambiguous_size=<n>        Ambiguous size. [default: 128]
  --chunk_shape=<n>           Shape of chunk for processing. [default: 10000]
  --tile_shape=<n>            Shape of tiles for processing. [default: 2048]
  --save_raw_map=<BOOL>       For `run_mode`=`tile`. To save raw prediction or not. [default: False]
  --wsi_proc_mag=<n>          Magnification level used for WSI processing. [default: 40]
  --cache_path=<path>         Path for cache. Should be placed on SSD with at least 100GB. [default: cache]
  --input_dir=<path>          Path to input data directory. Assumes the files are not nested within directory.
  --input_msk_dir=<path>      Path to directory containing tissue masks. Should have the same name as corresponding WSIs. [default: '']
  --output_dir=<path>         Path to output data directory. Will create automtically if doesn't exist. [default: output/]
  --patch_input_shape=<n>     Shape of input patch to the network- Assume square shape. [default: 270]
  --patch_output_shape=<n>    Shape of network output- Assume square shape. [default: 80]
```
Examples:

Example one:
```
python run_infer.py
```

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

## Overlaid Segmentation and Classification Prediction

<p float="left">
  <img src="docs/seg.gif" alt="Segmentation" width="870" />
</p>

The colour of the nuclear boundary denotes the type of nucleus. <br />
Blue: epithelial<br />
Red: inflammatory <br />
Green: spindle-shaped <br />
Cyan: miscellaneous

## Datasets

Download the CoNSeP dataset as used in our paper from [this link](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet/). <br />
Download the Kumar, CPM-15, CPM-17 and TNBC datsets from [this link](https://drive.google.com/open?id=1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK).  <br />

Ground truth files are in `.mat` format, refer to the README included with the datasets for further information. 

## Authors

* [Quoc Dang Vu](https://github.com/vqdang)
* [Simon Graham](https://github.com/simongraham)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


