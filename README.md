# HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images

A multiple branch network that performs nuclear instance segmentation and classification within a single network. The network leverages the horizontal and vertical distances of nuclear pixels to their centres of mass to separate clustered cells. A dedicated up-sampling branch is used to classify the nuclear type for each segmented instance. <br />

This is an extended version of our previous work: XY-Net. <br />

[Link to paper](https://arxiv.org/abs/1812.06499v4)

## Repository Structure

- `src/` contains executable files used to run the model. Further information on running the code can be found in the corresponding directory.
- `loader/`contains scripts for data loading and self implemented augmentation functions.
- `metrics/`contains evaluation code. 
- `misc/`contains util scripts. 
- `model/` contains scripts that define the architecture of the segmentation models. 
- `opt/` contains scripts that define the model hyperparameters. 
- `postproc/` contains post processing utils. 
- `config.py` is the configuration file. Paths need to be changed accordingly.
- `train.py` and `infer.py` are the training and inference scripts respectively.
- `process.py` is the post processing script for obtaining the final instances. 
- `extract_patches.py` is the patch extraction script. 

## HoVer-Net

![](network.png)


## Citation

If any part of this code is used, please give appropriate citation to our paper: <br />

**Graham, Simon, Quoc Dang Vu, Shan E. Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak, and Nasir Rajpoot. "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images." arXiv preprint arXiv:1812.06499v4 (2019).** <br />

BibTex entry: <br />
```
@article{graham2019hover,
  title={HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images},
  author={Simon Graham, Quoc Dang Vu, Shan E Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak, Nasir Rajpoot},
  journal={arXiv preprint arXiv:1812.06499v4},
  year={2019}
}
```

## Overlaid Segmentation and Classification Prediction

<p float="left">
  <img src="/seg.gif" alt="Segmentation" width="870" />
</p>

The colour of the nuclear boundary denotes the type of nucleus. <br />
Blue: epithelial<br />
Red: inflammatory <br />
Green: spindle-shaped <br />
Cyan: miscellaneous

## Authors

* [Quoc Dang Vu](https://github.com/vqdang)
* [Simon Graham](https://github.com/simongraham)

## Additional Implementations available 
 
* [FCN8](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
* [U-Net](https://arxiv.org/pdf/1505.04597.pdf)
* [SegNet](https://arxiv.org/pdf/1511.00561.pdf)
* [DCAN](https://www.sciencedirect.com/science/article/abs/pii/S1361841516302043) 
* [Micro-Net](https://www.sciencedirect.com/science/article/abs/pii/S1361841518300628)
* [DIST](https://ieeexplore.ieee.org/document/8438559)

## Getting Started

Install the required libraries before using this code. Please refer to `requirements.txt`

## Results

All comparative results on the CoNSeP, Kumar and CPM-17 datasets can be found [here](https://drive.google.com/drive/folders/1WTkleeaE6ne8qxuYzptv2bKwMdZVBpzr?usp=sharing). 

## Extra

The cell profiler pipeline that we used in our comparative experiments can be found [here](https://drive.google.com/file/d/1E5UII9fsYT2N2KBUNLS89OV9AstYDLlZ/view?usp=sharing).

## Companion Sites
The same version of this repository is officially available on the following sites for collection/affiliation purpose

* [Tissue Image Analytics Lab](https://github.com/TIA-Lab)
* [Quantitative Imaging and Informatics Laboratory](https://github.com/QuIIL)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


