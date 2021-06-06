import os
import shutil
import pandas as pd

import torch
import PIL.Image as Image
import torchvision.transforms as transforms
import time

t = transforms.Compose([transforms.Resize((224,224))])

data_dir = '../../data'
image_dir = os.path.join(data_dir, 'isic/Images')

def main(csv_filename, include_sonic):
    if include_sonic:
        new_image_dir = image_dir.replace('Images','ImagesSmallerWithSonic')
        p = pd.read_csv(os.path.join(data_dir,csv_filename))
    else:
        new_image_dir = image_dir.replace('Images','ImagesSmaller')
        p = pd.read_csv(os.path.join(data_dir,csv_filename))

    image_names =  p['image_name'].values

    if not os.path.exists(new_image_dir):
        print('making ',new_image_dir)
        os.mkdir(new_image_dir)

    t1 = time.time()
    print('resizing images')
    for i,image_name in enumerate(image_names):
        if i % 1000 == 0:
            t2 = time.time()
            print(i, t2-t1)

        original = os.path.join(image_dir, image_name)
        target = os.path.join(new_image_dir, image_name)
        #shutil.copyfile(original, target)

        #print(image_name)
        img = Image.open(os.path.join(image_dir,image_name))
        # tranform
        img_t = t(img).convert("RGB")
        img_t.save(os.path.join(new_image_dir,image_name),"JPEG")
        
if __name__ == '__main__':
    main(csv_filename='isic/df_with_sonic_age_over_50_id.csv',include_sonic=False)