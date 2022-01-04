import segmentation_models as sm
import tensorflow.keras.backend as k
import gc
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

sm.set_framework('tf.keras')
# ---------------------------------------------------------------
# 
# Direct Segmentation model Builder
# Deeplabv3+ is added to segmentation_models package
#
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# build model structure
# ---------------------------------------------------------------
def build_seg_model(
    model,
    backbone = 'xception',
    weight = 'pascal_voc',
    input_shape = (512,512,3),
    activation='sigmoid',
    n_classes = 1,
    decoder='upsampling',
    freeze=True
):
    '''
    ---- Possible ( Models , Backbone, Weight ) ----
    (['Deeplab'] ,  ['xception','mobilenetv2'], ['pascal_voc','city_scape'] ),
    (['FPN','Unet','Linknet'] ,
    ['inceptionresnetv2','inceptionv3','vgg16','vgg19',
    'resnet50','densenet121','mobilenetv2','efficientnetb[0-7]'] ,
    ['none','imagenet'])
    '''
    if model == 'Deeplab' or model=='deeplab':
        ret = sm.Deeplabv3(weights = weight,input_shape=input_shape,classes = n_classes,activation=activation,backbone=backbone)
    elif model == 'FPN' or model=='fpn':
        ret = sm.FPN(backbone, input_shape = input_shape,classes=n_classes, activation=activation,encoder_weights=weight,encoder_freeze=freeze)
    elif model == 'Unet' or model=='unet':
        ret = sm.Unet(backbone, input_shape = input_shape,classes=n_classes, activation=activation,encoder_weights=weight,decoder_block_type=decoder, encoder_freeze=freeze)
    elif model == 'Linknet' or model=='linknet':
        ret = sm.Linknet(backbone, input_shape = input_shape,classes=n_classes, activation=activation,encoder_weights=weight,encoder_freeze=freeze)
    
    return ret
# ---------------------------------------------------------------
# build train callback 
# ---------------------------------------------------------------
def build_callback(model_path,patience):
    model_chkpt = ModelCheckpoint(filepath = model_path, monitor = 'val_loss', verbose = 1, save_best_only = True)
    early_stopping = EarlyStopping(monitor = 'val_loss',patience = patience)
    lr_plan = ReduceLROnPlateau(monitor='val_loss',factor = 0.5, patience = patience//2)
    class ClearMemory(Callback):
        def on_epoch_end(self,epoch,logs = None):
            gc.collect()
            k.clear_session()
    return [model_chkpt,early_stopping,lr_plan,ClearMemory()]

