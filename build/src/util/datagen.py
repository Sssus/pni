from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import numpy as np
import cv2 as cv
from albumentations import *


class clf_generator(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, zip_path_list):
        self.batch_size = batch_size
        self.img_size = img_size
        self.zip_path_list = zip_path_list

    def __len__(self):
        return len(self.zip_path_list) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.zip_path_list[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + (1,), dtype="uint8")
        for j, path in enumerate(batch_input_img_paths):
            img = cv.imread(path[0])
            img = img.astype(np.float32)/255.0
            x[j] = img
        
            patch_group = path[1].split('.')[0][-1]; label=0
            if patch_group=='2' or patch_group=='3':
                label=1
            elif patch_group=='1' or patch_group=='4':
                label=0
            y[j] = np.expand_dims(label, 1)
            
        return x, y

    
    
class generator(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, zip_path_list,is_train):
        self.batch_size = batch_size
        self.img_size = img_size
        self.zip_path_list = zip_path_list
        self.augment = train_aug() if is_train==True else test_aug()
        self.augmentor = ImageDataGenerator(
            rotation_range=90,
            horizontal_flip=True,
            vertical_flip=True
        )
    
    def __len__(self):
        return len(self.zip_path_list) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_patch_pairs = self.zip_path_list[i : i + self.batch_size]
        
        batch_x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        batch_y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_patch_pairs):
            img_path = path[0]; mask_path = path[1]
            #img = load_img(path, target_size=self.img_size)
            img = cv.imread(img_path)
            #img = img.astype(np.float32)/255.0
            batch_x[j] = img
            if '_p3' in mask_path or '_p4' in mask_path:
                msk_img = np.zeros(self.img_size)
            else:
                msk_img = np.squeeze(cv.imread(mask_path,0))
            batch_y[j] = np.expand_dims(msk_img, 2)
            batch_aug = self.augment(image=batch_x[j],mask=batch_y[j])
            batch_x[j] = batch_aug['image']; batch_y[j] = batch_aug['mask']
        return batch_x,batch_y
    
class tumor_generator(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, zip_path_list,is_train):
        self.batch_size = batch_size
        self.img_size = img_size
        self.zip_path_list = zip_path_list
        self.augment = train_aug() if is_train==True else test_aug()
        
    def __len__(self):
        return len(self.zip_path_list) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_patch_pairs = self.zip_path_list[i : i + self.batch_size]
        
        batch_x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        batch_y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_patch_pairs):
            img_path = path[0]; mask_path = path[1]
            img = cv.imread(img_path)
            batch_x[j] = img
            if '_p1' in mask_path or '_p4' in mask_path:
                msk_img = np.zeros(self.img_size)
            else:
                msk_img = np.squeeze(cv.imread(mask_path,0))
            batch_y[j] = np.expand_dims(msk_img, 2)
            batch_aug = self.augment(image=batch_x[j],mask=batch_y[j])
            batch_x[j] = batch_aug['image']; batch_y[j] = batch_aug['mask']
        return batch_x,batch_y

###############################
## Augmentation Func
###############################
def train_aug():
    ret = Compose([
        HorizontalFlip(),
        VerticalFlip(),
        Rotate(),
        ElasticTransform(),
        #RandomContrast(limit=0.2, p=0.5),
        #GaussNoise(),
        #MotionBlur(),
        #RandomGamma(gamma_limit=(80, 120), p=0.5),
        #RandomBrightness(limit=0.2, p=0.5),
        #HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=.9),
        ToFloat(max_value=255) # Scaling to [0.0 ,1.0]
    ])
    return ret

def test_aug():
    ret = Compose([
        #HorizontalFlip(p=0)
        ToFloat(max_value=255) # Scaling to [0.0 ,1.0]
    ])
    return ret