# Nuclear Segmentation Data

## Kumar Dataset

Please refer to this [link](https://drive.google.com/open?id=1HKNOed4n0IV5frKFs_-kMtfiqqzBKMX5) to download the data that we used for the Kumar dataset. 

### File Structure

```
-- Images/
	 -- train/
	 -- test/
    	       -- same_tissue/
    	       -- diff_tissue/
-- Labels/
-- Overlay/
```
          
`Images` contains the original images, that are split into `train` and `test` directories. <br/> 

We split `test` into `same_tissue` and `diff_tissue` test sets. `same_tissue` contains tissue types that have been seen during training, whereas `diff_tissue` contains tissue types that are not represented in the training set. A very similar strategy was used in **[1]**, but the exact data split was not given. We recommend people using this dataset to follow the exact data split to encourage reproducibility.

`Labels` contains a numpy file for each image, that contains the instance-level ground truth. The background is labelled as 0 and each nucleus is assigned a unique integer. 

`Overlay` shows the ground truth overlaid on top of the original images. 

### Using the Data

The code within the repository uses patches in `.npy` format. Each patch contains 4 channels:
- Channels 1-3: Original RGB image
- Channel 4: Instance-level label
Use `/src/extract_patches.py` to generate patches from the original RGB image and label (the code automatically stacks the original image and the ground truth label before extraction).

## References
**[1]**  Kumar, Neeraj, Ruchika Verma, Sanuj Sharma, Surabhi Bhargava, Abhishek Vahadane, and Amit Sethi. "A dataset and a technique for generalized nuclear segmentation for computational pathology." IEEE transactions on medical imaging 36, no. 7 (2017): 1550-1560. <br/>

