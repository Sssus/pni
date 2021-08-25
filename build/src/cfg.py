import glob, os, re
from sklearn.utils import shuffle

# ------------------
# Data Config
# ------------------

ORGAN = ['colon','pancreas','prostate'] # 슬라이드 조직
DATA_SOURCE = ['PAIP']# 슬라이드 출처
PATIENT_SPLIT = [0.7,0.15,0.15] # train valid test 환자의 분리

# ------------------
# Patch Config
# ------------------

PATCH_TYPE = 'patches' # 패치를 추출하여 저장할 directory name
PATCH_SIZE = 512 # 추출할 패치의 사이즈
OVERLAP = 0.5 # 패치의 오버랩 추출 (0이면 overlap이 없음)
XML_NAME = '_nerve.xml' # svs파일과 대응되는 xml파일의 형식 (abcd.svs => abcd_nerve.xml)
PATCH_DICT = {
    'p1':'nerve_without_tumor',
    'p2':'nerve_with_tumor',
    'p3':'tumor_without_nerve',
    'p4':'non_tumor_non_nerve',
    'p5':'tumor',
    'p6':'tumor_bbox'
}

# ------------------
# Datagen Config 
# ------------------
INPUT_SIZE = (512,512)

BATCH_SIZE = 32
NUM_CLASSES = 1
TRUE_LABEL_LIST = [
    {'label':2,'restrict':0}
]
ZERO_LABEL_LIST = [
    {'label':3,'restrict':30},
    {'label':4,'restrict':5}
]

# Binary Segmentation 기준 true mask에 사용할 label과 빈 mask에 사용할 label 
# restrict는 각각의 label에 해당하는 패치의 제한을 주는 것 0이면 제한 없음

# ------------------
# Train Config 
# ------------------
INPUT_SHAPE = (512,512,3)
EPOCHS = 200
PATIENCE = 10
N_CLASSES = 1
MODEL_PATH = './model/0824_deeplabv3.hdf5'
MODEL = 'deeplab'
BACKBONE = 'xception'
WEIGHT = 'pascal_voc'
ACTIVATION = 'sigmoid'
INITIAL_LEARNING_RATE=0.005
LOSS = 'dice+focal'
OPTIMIZER = 'adam'
METRIC = ['iou','f1-score']



ALL_PATIENT_PATHS = []; [ALL_PATIENT_PATHS.extend(glob.glob(f'/data/{d}/{e}/*/')) for d in ORGAN for e in DATA_SOURCE]
TRAIN_PATIENT_PATHS = shuffle(ALL_PATIENT_PATHS,random_state=311)[:int(PATIENT_SPLIT[0]*len(ALL_PATIENT_PATHS))]
VALID_PATIENT_PATHS = shuffle(ALL_PATIENT_PATHS,random_state=311)[int(PATIENT_SPLIT[0]*len(ALL_PATIENT_PATHS)):
                                                                  -int(PATIENT_SPLIT[2]*len(ALL_PATIENT_PATHS))]
TEST_PATIENT_PATHS = shuffle(ALL_PATIENT_PATHS,random_state=311)[-int(PATIENT_SPLIT[2]*len(ALL_PATIENT_PATHS)):]
ALL_SLIDE_PATHS = [glob.glob(path+'*.svs') for path in ALL_PATIENT_PATHS]

TRAIN_ZIP = []
for path in TRAIN_PATIENT_PATHS:
    for lbl in TRUE_LABEL_LIST :TRAIN_ZIP.extend(glob.glob(path+f'{PATCH_TYPE}/mask/*_p{lbl["label"]}.png')) 
    for lbl in ZERO_LABEL_LIST :TRAIN_ZIP.extend(glob.glob(path+f'{PATCH_TYPE}/mask/*_p{lbl["label"]}.png')[:lbl["restrict"]])
TRAIN_ZIP = shuffle([('/'.join(x.split('/')[:-2])+'/image/'+re.sub('_p[0-9]','',x.split('/')[-1]),x) for x in TRAIN_ZIP],random_state=311)
VALID_ZIP = []
for path in VALID_PATIENT_PATHS:
    for lbl in TRUE_LABEL_LIST :VALID_ZIP.extend(glob.glob(path+f'{PATCH_TYPE}/mask/*_p{lbl["label"]}.png')) 
    for lbl in ZERO_LABEL_LIST :VALID_ZIP.extend(glob.glob(path+f'{PATCH_TYPE}/mask/*_p{lbl["label"]}.png')[:lbl["restrict"]])
VALID_ZIP = shuffle([('/'.join(x.split('/')[:-2])+'/image/'+re.sub('_p[0-9]','',x.split('/')[-1]),x) for x in VALID_ZIP],random_state=311)
TEST_ZIP = []
for path in TEST_PATIENT_PATHS:
    for lbl in TRUE_LABEL_LIST :TEST_ZIP.extend(glob.glob(path+f'{PATCH_TYPE}/mask/*_p{lbl["label"]}.png')) 
    for lbl in ZERO_LABEL_LIST :TEST_ZIP.extend(glob.glob(path+f'{PATCH_TYPE}/mask/*_p{lbl["label"]}.png')[:lbl["restrict"]])
TEST_ZIP = shuffle([('/'.join(x.split('/')[:-2])+'/image/'+re.sub('_p[0-9]','',x.split('/')[-1]),x) for x in TEST_ZIP],random_state=311)
ZERO_LABELS = [x['label'] for x in ZERO_LABEL_LIST]












init_params = {
    'level':2,
    'overlap':0.5,
    'patch_size':512,
    'patch_name':'patches'
}


GLOB_PATH='../../new_data/paip/'

model_cfg = {
    'clf_model_path': '/home/centos/jupyter/pathology/data/model/clf_0706.hdf5',
    'seg_model_path':'/home/centos/jupyter/pathology/data/model/seg_nerve_0721_inceptionresnetv2.hdf5',
    'nerve_backbone':'inceptionresnetv2',
    #'seg_tumor_model_path':'/home/centos/jupyter/pathology/data/model/seg_tumor_tuner_inceptionv3.hdf5',
    #'tumor_backbone':'inceptionv3'
    'seg_tumor_model_path':'/home/centos/jupyter/pathology/data/model/seg_tumor_0721_inceptionresnetv2.hdf5',
    'tumor_backbone':'inceptionresnetv2'
}

BASE_PATCH_PATH = glob.glob('/home/centos/jupyter/pathology/data/img/patch_overlap_global_512//*/*')
TRAIN_PATIENT_PATH = shuffle(BASE_PATCH_PATH,random_state=311)[:int(0.60*len(BASE_PATCH_PATH))]
VALID_PATIENT_PATH = shuffle(BASE_PATCH_PATH,random_state=311)[int(0.60*len(BASE_PATCH_PATH)):int(0.80*len(BASE_PATCH_PATH))]
TEST_PATIENT_PATH = shuffle(BASE_PATCH_PATH,random_state=311)[int(0.80*len(BASE_PATCH_PATH)):]