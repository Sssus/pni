from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import numpy as np
import cv2 as cv
from albumentations import *
import tensorflow as tf

class generator(tf.data.Dataset):
    
    # -----------------------------------------------
    # file_gen 은 img, mask path를 return하는 iterator
    # -----------------------------------------------
    def file_gen(zip_path_list):
        for zip_path in zip_path_list:
            img_path,mask_path = zip_path
            yield (img_path,mask_path)
    
    # ------------------------------------------------
    # map_func은 path로 부터 img를 load하는 map function
    # ------------------------------------------------
    def map_func(paths,zero_labels,is_train,binary):
        img_path = str(paths[0],'utf-8'); msk_path = str(paths[1],'utf-8')
        img = cv.imread(img_path)
        
        if binary==True:
            msk = cv.imread(msk_path,0)
            for z in zero_labels:
                msk[msk==z]=0
            msk = np.clip(msk,0,1).astype(np.float32)
            
        else:
            msk  = np.zeros(img.shape).astype(np.float32)
            lbl_msk = cv.imread(msk_path,0)
            msk[...,0] = lbl_msk==0*1
            msk[...,1] = lbl_msk==1*1
            msk[...,2] = np.logical_or((lbl_msk==2),(lbl_msk==3))*1
            msk = msk.astype(np.float32)
        ## Augmentation
        if is_train==True:
            aug = Compose([
                HorizontalFlip(),
                HueSaturationValue(hue_shift_limit=5,sat_shift_limit=20,val_shift_limit=10,p=.9),
                VerticalFlip(),
                RandomRotate90(),
                ElasticTransform(),
                ToFloat(max_value=255,always_apply=True,p=1.0)
            ])
        else:
            aug = Compose([
                ToFloat(max_value=255,always_apply=True,p=1.0)
            ])
        
        augmented = aug(image=img,mask=msk)

        return augmented['image'], augmented['mask']

    # ------------------------------------------------
    # class 호출하면 __new__로 dataset class 생성
    # ------------------------------------------------    
    def __new__(cls,zip_path_list,batch_size,zero_labels,is_train,binary):
        ret = tf.data.Dataset.from_generator(
            cls.file_gen,
            tf.string,
            output_shapes = tf.TensorShape([None]),
            args = (zip_path_list,)
        )
        ret = ret.map(
            lambda x : tf.numpy_function(
                cls.map_func,
                [x,zero_labels,is_train,binary],
                [tf.float32,tf.float32]
            ),num_parallel_calls = tf.data.experimental.AUTOTUNE
        )
        ret = ret.batch(batch_size,drop_remainder = True)
        ret = ret.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
        
        return ret



    
    



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

    
    
class keras_generator(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, zip_path_list,is_train,zero_labels):
        '''
        zero_labels : zero mask를 생성하는 라벨들의 list, format은 numerical list
        '''
        self.batch_size = batch_size
        self.img_size = img_size
        self.zip_path_list = zip_path_list
        self.zero_labels = ['_p'+str(s) for s in zero_labels]
        self.augmentor = train_aug() if is_train==True else test_aug()
        
    
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
            if np.any([x in mask_path for x in self.zero_labels]):
                msk_img = np.zeros(self.img_size)
            else:
                msk_img = np.squeeze(cv.imread(mask_path,0))
            msk_img = np.expand_dims(msk_img, 2)
            batch_aug = self.augmentor(image=img,mask=msk_img)
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
        #ChannelShuffle(),
        #Blur(),
        #RGBShift(),
        HueSaturationValue(hue_shift_limit=5,sat_shift_limit=20,val_shift_limit=10,p=.9),
        VerticalFlip(),
        RandomRotate90(),
        ElasticTransform(),
        #RandomContrast(limit=0.2, p=0.5),
        #GaussNoise(),
        #MotionBlur(),
        #RandomGamma(gamma_limit=(80, 120), p=0.5),
        #RandomBrightness(limit=0.2, p=0.5),
        # Scaling to [0.0 ,1.0]
        ToFloat(max_value=255,always_apply=True,p=1.0),

    ])
    return ret

def test_aug():
    ret = Compose([
        #HorizontalFlip(p=0)
        ToFloat(max_value=255,always_apply=True,p=1.0) # Scaling to [0.0 ,1.0]
    ])
    return ret