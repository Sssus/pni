import segmentation_models as sm
from cfg import train_config
# -----------------------------------------
# Data Set
# -----------------------------------------

'''
각 환자(Train , Valid, Test)에서 패치 종류에 따라 datagen에 feeding할 image path와 mask path를 선정합니다.
Ex) Tumor Task의 경우 p5패치가 Tumor p1 , p4패치를 Non Tumor 패치로 할 경우. zero_labels에 [1,4]를 넣어줍니다.
=> p1,p4 패치는 zero mask가 datagen에서 생성됩니다. 
Bounding Box안에서 패치를 추출할 경우, 모든 패치를 넣으면 되지만 Bounding Box가 없을 경우. Non Class( Not Tumor) 패치의 갯수를 제한해 줍니다.
이 때 갯수는 1-Class 의 갯수와 비슷한 수준 (20%를 넘지 않게) 으로 합니다.
'''

def build_model(model,input_shape,activation,backbone,weight,n_classes,loss,metric):
    if model=='Deeplab':
        ret = sm.Deeplab()
    elif model=='FPN':
        ret = sm.FPN()
    elif model=='Linknet':
        ret = sm.Linknet()
    else:
        ret = sm.Unet()
    
    
    
def build_loss():
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss(alpha =0.25, gamma = 6.0)
    return dice_loss + focal_loss
    
def build_metric():
    return [sm.metrics.IOUScore(threshold=0.5),sm.metrics.FScore(threshold=0.5)]

def build_optim(lr):
    return Adam(lr)

def build_callback(model_path,patience):
    class ClearMemory(Callback):
        def on_epoch_end(self,epoch,logs=None):
            gc.collect()
            k.clear_session()
    checkpt = ModelCheckpoint(filepath=model_path,monitor='val_loss',verbose=1,save_best_only=True,mode='auto')
    stopping = EarlyStopping(monitor='val_loss',patience=patience)
    learning_plan = ReduceLROnPlateau( monitor='val_loss', factor = 0.5, patience = patience//2)
    return [checkpt,stopping,learning_plan,ClearMemory()]


if __name__=='__main__':
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_model(
            train_config['model'],
            train_config['input_shape'],
            train_config['backbone'],
            train_config['weight'],
            train_config['n_classes'],
            train_config['loss'],
        )
        model.compile(
            build_optim(train_config['init_lr']),
            build_loss(),
            build_metric()
        )
    model.fit(
        train_gen,
        epochs = train_config['epoch'],
        validation_data = valid_gen,
        callbacks = build_callback(train_config['model_path'],train_config['patience']),
        max_queue_size = 20,
        workers=10,
        verbose=1
    )
    model.
    