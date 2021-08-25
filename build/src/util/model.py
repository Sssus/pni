from tensorflow.keras import layers ,Model, Input
import segmentation_models as sm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as k
import gc
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam, RMSprop



sm.set_framework('tf.keras')

def build_seg_model(model,backbone,weight,input_shape,n_classes,loss,init_lr,optimizer,activation='sigmoid',is_train=False):
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
        ret = sm.FPN(backbone, input_shape = input_shape,classes=n_classes, activation=activation,encoder_weights=weight)
    elif model == 'Unet' or model=='unet':
        ret = sm.Unet(backbone, input_shape = input_shape,classes=n_classes, activation=activation,encoder_weights=weight)
    elif model == 'Linknet' or model=='linknet':
        ret = sm.Linknet(backbone, input_shape = input_shape,classes=n_classes, activation=activation,encoder_weights=weight)
    
    if is_train==False:
        return ret
    else:
        if n_classes==1:
            loss = build_binary_loss(loss)
        else:
            loss = build_multi_loss(loss)
        optim = Adam(init_lr) if optimizer=='adam' else SGD(init_lr)
        metrics = [sm.metrics.IOUScore(0.5),sm.metrics.FScore(0.5)]    
        ret.compile(optim,loss,metrics)
        return ret

def build_callback(model_path,patience):
    model_chkpt = ModelCheckpoint(filepath = model_path, monitor = 'val_loss', verbose = 1, save_best_only = True)
    early_stopping = EarlyStopping(monitor = 'val_loss',patience = patience)
    lr_plan = ReduceLROnPlateau(monitor='val_loss',factor = 0.5, patience = patience//2)
    class ClearMemory(Callback):
        def on_epoch_end(self,epoch,logs = None):
            gc.collect()
            k.clear_session()
    return [model_chkpt,early_stopping,lr_plan,ClearMemory()]

def build_binary_loss(loss):
    '''
    ---- Possible Losses ---- 
    'focal_dice', 'focal_jacard', 'ce_dice', 'ce_jacard', 'ce', 'focal', 'jacard', 'dice'
    '''
    if loss == 'focal_dice':
        ret = sm.losses.binary_focal_dice_loss
    elif loss == 'focal_jacard':
        ret = sm.losses.binary_focal_jacard_loss
    elif loss == 'ce_dice':
        ret = sm.losses.bce_dice_loss
    elif loss == 'ce_jacard':
        ret = sm.losses.bce_jacard_loss
    elif loss == 'ce':
        ret = sm.losses.binary_crossentropy
    elif loss == 'focal':
        ret = sm.losses.binary_focal_loss
    elif loss == 'jacard':
        ret = sm.losses.jacard_loss
    elif loss == 'dice':
        ret = sm.losses.dice_loss




def build_unet(img_size, num_classes):
    inputs = Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = Model(inputs, outputs)
    return model

def build_clf_model(num_class):
    
    base_model = InceptionV3(weights=None, include_top=False)
    #base_model = ResNet50(weights='imagenet', include_top=False)

    base_model.trainable = False

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    #x = Dense(512, activation='softmax')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_class, activation='sigmoid')(x)


    # this is the model we will train
    model = Model(base_model.input, predictions)
    return model
# Free up RAM in case the model definition cells were run multiple times
'''
keras.backend.clear_session()

# Build model
model = get_model(img_size, num_classes)
model.summary()
'''