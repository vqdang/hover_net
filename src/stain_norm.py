import os
import cv2
import glob
import staintools

from misc.utils import rm_n_mkdir

norm_brightness = False
imgs_dir = '../../../data/NUC_UHCW/No_SN/valid/' 
save_dir = '../../../data/NUC_UHCW/SN_5784/valid/' 

file_list = glob.glob(imgs_dir + '*.png')
file_list.sort() # ensure same order [1]

if norm_brightness:
    standardizer = staintools.BrightnessStandardizer()
stain_normalizer = staintools.StainNormalizer(method='vahadane')

# dict of paths to target image and dir code to make output folder
stain_norm_target = {
    '../data/TCGA-21-5784-01Z-00-DX1.tif' : '5784'
}

for target_path, target_code in stain_norm_target.items():
    target_img = cv2.imread(target_path)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    if norm_brightness:
        target_img = standardizer.transform(target_img)
    stain_normalizer.fit(target_img)
    
    norm_dir = "%s/%s/" % (save_dir, target_code)
    rm_n_mkdir(norm_dir)
    
    for img_path in file_list:
        filename = os.path.basename(img_path)
        basename = filename.split('.')[0]
        print(basename)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if norm_brightness:
            img = standardizer.transform(img)
        img = stain_normalizer.transform(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("%s/%s.png" % (norm_dir, basename), img)
