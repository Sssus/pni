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
    parser.add_argument('CLASS_NAME' , type = str)
    parser.add_argument('MODEL', type = str, help = 'used model')
    parser.add_argument('BACKBONE', type = str, help = 'backbone paired model')
    parser.add_argument('WEIGHT', type = str, help = 'weight paired model and backbone')
    parser.add_argument('LOSS', type = str, help = 'focal or ce or dice')

    args = parser.parse_args()
    CLASS_NAME = args.CLASS_NAME
    MODEL = args.MODEL
    BACKBONE = args.BACKBONE
    WEIGHT = args.WEIGHT
    LOSS = args.LOSS
    TODAY = datetime.datetime.now().strftime('%m%d')
    MODEL_PATH = os.path.join('./model',(CLASS_NAME+'_'+str(MAGNIFICATION)+'_'+MODEL+'_'+BACKBONE+'_'+WEIGHT+'_'+TODAY+'_'+LOSS+'.hdf5'))
    return CLASS_NAME,MODEL,BACKBONE,WEIGHT,TODAY,MODEL_PATH

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
    CLASS_NAME, MODEL, BACKBONE, WEIGHT, TODAY, MODEL_PATH = parse_func()
    if CLASS_NAME=='tumor' or CLASS_NAME=='nerve':
        if LOSS=='focal':
            loss = sm.losses.DiceLoss(smooth=1e+0,) + sm.losses.BinaryFocalLoss(alpha = 0.25, gamma = 6.0)
        elif LOSS=='ce':
            loss = sm.losses.DiceLoss(smooth=1e+0,) + sm.losses.BinaryCELoss()
        else:
            loss = sm.losses.DiceLoss(smooth=1e+0,)
        activation = 'sigmoid'
        binary=True
        n_classes=1
    else: ## Class Weight is computed for 5 batches. and Computed by sklearn . then log1p for cw
        if LOSS=='focal':
       	    loss = sm.losses.DiceLoss(smooth=1e+0,class_weights=np.array([0.4,0.7,3.0])) + sm.losses.CategoricalFocalLoss(alpha = 0.25, gamma = 6.0)
        elif LOSS=='ce':
            loss = sm.losses.DiceLoss(smooth=1e+0,class_weights=np.array([0.4,0.7,3.0])) + sm.losses.CategoricalCELoss() # Back, Tumor, Nerve
        else:
            loss = sm.losses.DiceLoss(smooth=1e+0,class_weights=np.array([0.4,0.7,3.0])) 
        activation = 'softmax'
        binary=False
        n_classes=3
    # -------------------------------------------------
    # Define Data
    # -------------------------------------------------    
    if CLASS_NAME=='nerve':
        TRUE_LABEL_LIST = [
            {'label':2,'restrict':0},
            {'label':3,'restrict':0}
        ]
        ZERO_LABEL_LIST = [
            {'label':1,'restrict':5},
            {'label':4,'restrict':10}
        ]
    elif CLASS_NAME=='tumor':
        TRUE_LABEL_LIST = [
        {'label':1,'restrict':0}
        ]
        ZERO_LABEL_LIST = [
            {'label':2,'restrict':50},
            {'label':3,'restrict':50},
            {'label':4,'restrict':100}
        ]
    else:
        TRUE_LABEL_LIST = [
            {'label':1,'restrict':100},
            {'label':2,'restrict':100},
            {'label':3,'restrict':100},
            {'label':4,'restrict':100}
        ]
        ZERO_LABEL_LIST = []   

    ALL_ZIP = []
    for path in ALL_SLIDE_PATHS:
        slide_name = path.split('/')[-1].replace('.tiff','').replace('.svs','')
        patient_path = '/'.join(path.split('/')[:-1]) + '/'
        for lbl in TRUE_LABEL_LIST :
            target_list = glob.glob(patient_path+f'{PATCH_NAME}/mask/{slide_name}*_p{lbl["label"]}.png')
            if lbl['restrict']==0:
                ALL_ZIP.extend(target_list) 
            else:
                ALL_ZIP.extend(shuffle(target_list,random_state = 311)[:lbl["restrict"]])
        for lbl in ZERO_LABEL_LIST :
            target_list = glob.glob(patient_path+f'{PATCH_NAME}/mask/{slide_name}*_p{lbl["label"]}.png')
            ALL_ZIP.extend(shuffle(target_list,random_state = 311)[:lbl["restrict"]])
    ALL_ZIP = shuffle([('/'.join(x.split('/')[:-2])+'/image/'+re.sub('_p[0-9]','',x.split('/')[-1]),x) for x in ALL_ZIP],random_state=311)

    TRAIN_ZIP  = ALL_ZIP[:int(0.66*len(ALL_ZIP))]
    VALID_ZIP  = ALL_ZIP[int(0.66*len(ALL_ZIP)):int(0.88*len(ALL_ZIP))]
    TEST_ZIP  = ALL_ZIP[int(0.88*len(ALL_ZIP)):]
    print(f'ALL PATCH : {len(ALL_ZIP)}, TRAIN PATCH : {len(TRAIN_ZIP)}, VALID PATCH : {len(VALID_ZIP)}, TEST PATCH : {len(TEST_ZIP)}')
    ZERO_LABELS = [x['label'] for x in ZERO_LABEL_LIST]
    print(ZERO_LABELS)

    print(f'Train Length : {len(TRAIN_ZIP)} Valid Length : {len(VALID_ZIP)}')
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
    optim = Adam(INITIAL_LEARNING_RATE)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), sm.metrics.Precision(), sm.metrics.Recall() ]        
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    print(f'model : {MODEL}, backbone : {BACKBONE}, n_class : {n_classes}, class : {CLASS_NAME}, activation : {activation}')
    with strategy.scope():
        model = build_seg_model(
            model = MODEL,
            backbone = BACKBONE,
            weight = WEIGHT,
            input_shape = INPUT_SHAPE,
            n_classes = n_classes,
            activation = activation
        )
        model.compile(optim, loss, metrics)
    
    # -------------------------------------------------
    # Model Train    
    # -------------------------------------------------
    callback_list = build_callback(MODEL_PATH,PATIENCE)
    try:

        model.fit(
            train_gen, 
            epochs = EPOCHS,
            validation_data=valid_gen,
            callbacks=callback_list,
            max_queue_size=36,
            workers=20,
        )
    except Exception as e:
        print(e)
        pass
    sys.exit()
