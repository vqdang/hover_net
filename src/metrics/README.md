
# Statistical Measurements

## Description

In this directory, the script `stats_utils.py` contains the code for each statistical measure. In order of appearance, the available measurementsare:

- get_fast_aji()
- get_fast_dice_2()
- get_fast_panoptic_quality()
- get_dice_1()
- get_dice_2()
- get_aji()

`get_aji()`: aggregated jaccard index as used in **[1]**. Ported from the authors' matlab code. The original matlab code is also given in the directory. <br/>
`get_fast_aji()`: aji optimised for speed. Set `mode=plus` to run AJI+, as used within our paper. <br/>
`get_dice_1()` and `get_dice_2()`: standard dice and ensemble dice (DICE2) **[2]** measures respectively. <br/> `get_fast_panoptic_quality()`: panoptic quality as used in **[3]**.

## References
**[1]**  Kumar, Neeraj, Ruchika Verma, Sanuj Sharma, Surabhi Bhargava, Abhishek Vahadane, and Amit Sethi. "A dataset and a technique for generalized nuclear segmentation for computational pathology." IEEE transactions on medical imaging 36, no. 7 (2017): 1550-1560. <br/>
**[2]** Vu, Quoc Dang, Simon Graham, Minh Nguyen Nhat To, Muhammad Shaban, Talha Qaiser, Navid Alemi Koohbanani, Syed Ali Khurram et al. "Methods for Segmentation and Classification of Digital Microscopy Tissue Images." arXiv preprint arXiv:1810.13230 (2018).  <br/>
**[3]** Kirillov, Alexander, Kaiming He, Ross Girshick, Carsten Rother, and Piotr Dollár. "Panoptic Segmentation." arXiv preprint arXiv:1801.00868 (2018).







