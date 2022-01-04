import numpy as np
import cv2 as cv
from albumentations import *
import tensorflow as tf

# ---------------------------------------------------------------
# 
# Tensorflow Data Generator
#
# ---------------------------------------------------------------

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
            tumor_mask = lbl_msk==1
            nerve_mask = np.logical_or(lbl_msk==2 , lbl_msk==3)
            tum_ner = np.stack([tumor_mask,nerve_mask],axis=-1).astype(np.float32)
            bg_mask = 1-tum_ner.sum(axis=-1,keepdims=True)
            msk = np.concatenate((bg_mask,tum_ner),axis=-1)
            
        ## Augmentation
        if is_train==True:
            aug = Compose([
                HorizontalFlip(),
                HueSaturationValue(hue_shift_limit=5,sat_shift_limit=20,val_shift_limit=10,p=.9),
                VerticalFlip(),
                RandomRotate90(),
                #GaussNoise(var_limit=25),
                #Blur(blur_limit=5),
                ElasticTransform(),
                ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, 
                    rotate_limit=15, border_mode=cv.BORDER_REFLECT_101, p=0.7
                ), 
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