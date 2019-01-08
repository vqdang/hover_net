
# Statistical Measurements

## Description

In this directory, the script `stats_utils.py` contains the code for each statistical measure. In order of appearance, the measurements present are:

- get_fast_aji()
- get_fast_dice_2()
- get_fast_panoptic_quality()
- get_dice_1()
- get_dice_2()
- get_aji()

`get_aji()` reports the aggregated jaccard index as used in [1] and is ported from the authors' matlab code. The original matlab code is also given in the directory. `get_fast_aji()` is optimised for speed. Set `mode=plus` to run AJI+, as used within our paper. `get_dice_1()` and `get_dice_2()` refer to standard dice and ensemble dice (DICE2) [2] measures respectively. `get_fast_panoptic_quality()` reports the statistical measure as used in [3].





