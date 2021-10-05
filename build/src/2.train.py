from util.datagen import generator
from util.model import *
from cfg import *

import os, glob, datetime, gc, argparse, sys
import numpy as np
import tensorflow as tf
import cv2 as cv
import segmentation_models as sm
sm.set_framework('tf.keras')
import tensorflow.keras.backend as k
from albumentations import *

# ------------------------------------------
#  Parameter parsing
# ------------------------------------------
def parse_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('MODEL', type = str, help = 'used model')
    parser.add_argument('BACKBONE', type = str, help = 'backbone paired model')
    parser.add_argument('WEIGHT', type = str, help = 'weight paired model and backbone')
    #parser.add_argument('LOSS', type = str, help = 'loss binary_focal , binary_ce , jacard , dice ')

    args = parser.parse_args()
    MODEL = args.MODEL
    BACKBONE = args.BACKBONE
    WEIGHT = args.WEIGHT
    #LOSS = args.LOSS
    TODAY = datetime.datetime.now().strftime('%m%d')
    MODEL_PATH = os.path.join('./model/',(CLASS_NAME+'_'+str(MAGNIFICATION)+'_'+MODEL+'_'+BACKBONE+'_'+WEIGHT+'_'+TODAY+'.hdf5'))
    return MODEL,BACKBONE,WEIGHT,TODAY,MODEL_PATH

if __name__=='__main__':
    # -------------------------------------------------
    # Setting gpu device
    # -------------------------------------------------
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    # -------------------------------------------------
    # Parameter Setting
    # -------------------------------------------------
    MODEL,BACKBONE,WEIGHT,TODAY,MODEL_PATH = parse_func()
    if CLASS_NAME=='tumor' or CLASS_NAME=='nerve':
        loss = sm.losses.DiceLoss() + sm.losses.BinaryFocalLoss(alpha = 0.25, gamma = 6.0)
        activation = 'sigmoid'
        binary=True
        n_classes=1
    else:
        loss = sm.losses.DiceLoss(np.array([1,0.5,1])) + sm.losses.CategoricalFocalLoss(alpha = 0.25, gamma = 6.0)
        activation = 'softmax'
        binary=False
        n_classes=3
    
    optim = Adam(INITIAL_LEARNING_RATE)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), sm.metrics.Precision(), sm.metrics.Recall() ]
    # -------------------------------------------------
    # Define Data
    # -------------------------------------------------
    
    
    # -------------------------------------------------
    # Datagenerator
    # -------------------------------------------------
    train_gen = generator(
        TRAIN_ZIP,
        BATCH_SIZE,
        ZERO_LABELS,
        is_train=True,
        binary=binary
    )
    valid_gen = generator(
        VALID_ZIP,
        BATCH_SIZE,
        ZERO_LABELS,
        is_train=False,
        binary=binary
    )
    
    # -------------------------------------------------
    # Model Build and Compile with distributed strategy
    # -------------------------------------------------
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = build_seg_model(
            model=MODEL,
            backbone=BACKBONE,
            weight=WEIGHT,
            input_shape=INPUT_SHAPE,
            activation=activation,
            n_classes = n_classes
            
        )
        model.compile(optim,loss,metrics)
    
    # -------------------------------------------------
    # Model Train    
    # -------------------------------------------------
    callback_list = build_callback(MODEL_PATH,PATIENCE)
    model.fit(
        train_gen, 
        epochs = EPOCHS,
        validation_data=valid_gen,
        callbacks=callback_list,
        max_queue_size=36,
        workers=12
    )
    sys.exit()
