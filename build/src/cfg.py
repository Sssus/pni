import glob, os, re
from sklearn.utils import shuffle

# ------------------
# Data Config
# ------------------

ORGAN = ['colon'] #['colon','pancreas','prostate'] # 슬라이드 조직
DATA_SOURCE = ['IS']# 슬라이드 출처 ex) ['IS','PAIP','HGH','TCGA']
DATA_SPLIT = [0.6,0.2,0.2] # train valid test 환자의 분리

# ------------------
# Patch Config
# ------------------


MAGNIFICATION=100      # **************** Change ****************** #
PATCH_SIZE = 512 # 추출할 패치의 사이즈  
OVERLAP = 0.5 # 패치의 오버랩 추출 (0이면 overlap이 없음)  # **************** Change ****************** #
LEVEL = 2    # **************** Change ****************** #
SLIDE_FORMAT='.tiff'
XML_NAME = '.xml' # svs파일과 대응되는 xml파일의 형식 (abcd.svs => abcd_nerve.xml)
PATCH_NAME = 'patch100' # 패치를 추출하여 저장할 directory name  # **************** Change ****************** #
PATCH_NAME = f'patch_{MAGNIFICATION}_{LEVEL}_{str(OVERLAP)}'


# ------------------
# Datagen Config 
# ------------------
CLASS_NAME = 'nerve' # **************** Change ****************** #
INPUT_SIZE = (PATCH_SIZE,PATCH_SIZE)

BATCH_SIZE = 32   # **************** Change ****************** #
NUM_CLASSES = 1

if CLASS_NAME=='nerve':
    TRUE_LABEL_LIST = [
        {'label':2,'restrict':0},
        {'label':3,'restrict':0}
    ]
    ZERO_LABEL_LIST = [
        {'label':1,'restrict':0},
        {'label':4,'restrict':0}
    ]
elif CLASS_NAME=='tumor':
    TRUE_LABEL_LIST = [
    {'label':1,'restrict':0}
    ]
    ZERO_LABEL_LIST = [
        {'label':2,'restrict':0},
        {'label':3,'restrict':0},
        {'label':4,'restrict':0}
    ]
else:
    TRUE_LABEL_LIST = [
        {'label':1,'restrict':100},
        {'label':2,'restrict':100},
        {'label':3,'restrict':100},
        {'label':4,'restrict':100}
    ]
    ZERO_LABEL_LIST = []


# Binary Segmentation 기준 true mask에 사용할 label과 빈 mask에 사용할 label 
# restrict는 각각의 label에 해당하는 패치의 제한을 주는 것 0이면 제한 없음

# ------------------
# Train Config 
# ------------------
INPUT_SHAPE = (PATCH_SIZE,PATCH_SIZE,3)
EPOCHS = 200
PATIENCE = 12
N_CLASSES = 1
MODEL_PATH = './model/0824_deeplabv3.hdf5'
MODEL = 'deeplab'
BACKBONE = 'xception'
WEIGHT = 'pascal_voc'
ACTIVATION = 'sigmoid'
INITIAL_LEARNING_RATE=0.01
LOSS = 'dice+focal'
OPTIMIZER = 'adam'
METRIC = ['iou','f1-score']


# --------- Patient 기준 ---------
ALL_PATIENT_PATHS = []; [ALL_PATIENT_PATHS.extend(glob.glob(f'/data/{d}/{e}/*/')) for d in ORGAN for e in DATA_SOURCE]
TRAIN_PATIENT_PATHS = shuffle(ALL_PATIENT_PATHS,random_state=311)[:int(DATA_SPLIT[0]*len(ALL_PATIENT_PATHS))]
VALID_PATIENT_PATHS = shuffle(ALL_PATIENT_PATHS,random_state=311)[int(DATA_SPLIT[0]*len(ALL_PATIENT_PATHS)):
                                                                  -int(DATA_SPLIT[2]*len(ALL_PATIENT_PATHS))]
TEST_PATIENT_PATHS = shuffle(ALL_PATIENT_PATHS,random_state=311)[-int(DATA_SPLIT[2]*len(ALL_PATIENT_PATHS)):]

TRAIN_ZIP = []
for path in TRAIN_PATIENT_PATHS:
    for lbl in TRUE_LABEL_LIST :TRAIN_ZIP.extend(glob.glob(path+f'{PATCH_NAME}/mask/*_p{lbl["label"]}.png')) 
    for lbl in ZERO_LABEL_LIST :TRAIN_ZIP.extend(glob.glob(path+f'{PATCH_NAME}/mask/*_p{lbl["label"]}.png')[:lbl["restrict"]])
TRAIN_ZIP = shuffle([('/'.join(x.split('/')[:-2])+'/image/'+re.sub('_p[0-9]','',x.split('/')[-1]),x) for x in TRAIN_ZIP],random_state=311)
VALID_ZIP = []
for path in VALID_PATIENT_PATHS:
    for lbl in TRUE_LABEL_LIST :VALID_ZIP.extend(glob.glob(path+f'{PATCH_NAME}/mask/*_p{lbl["label"]}.png')) 
    for lbl in ZERO_LABEL_LIST :VALID_ZIP.extend(glob.glob(path+f'{PATCH_NAME}/mask/*_p{lbl["label"]}.png')[:lbl["restrict"]])
VALID_ZIP = shuffle([('/'.join(x.split('/')[:-2])+'/image/'+re.sub('_p[0-9]','',x.split('/')[-1]),x) for x in VALID_ZIP],random_state=311)
TEST_ZIP = []
for path in TEST_PATIENT_PATHS:
    for lbl in TRUE_LABEL_LIST :TEST_ZIP.extend(glob.glob(path+f'{PATCH_NAME}/mask/*_p{lbl["label"]}.png')) 
    for lbl in ZERO_LABEL_LIST :TEST_ZIP.extend(glob.glob(path+f'{PATCH_NAME}/mask/*_p{lbl["label"]}.png')[:lbl["restrict"]])
TEST_ZIP = shuffle([('/'.join(x.split('/')[:-2])+'/image/'+re.sub('_p[0-9]','',x.split('/')[-1]),x) for x in TEST_ZIP],random_state=311)
ZERO_LABELS = [x['label'] for x in ZERO_LABEL_LIST]

# --------- Slide 기준 ----------
ALL_SLIDE_PATHS = []; [ALL_SLIDE_PATHS.extend(glob.glob(path+'*.tiff')) for path in ALL_PATIENT_PATHS]
TRAIN_SLIDE_PATHS = shuffle(ALL_SLIDE_PATHS,random_state = 311)[:int(DATA_SPLIT[0]*len(ALL_SLIDE_PATHS))]
VALID_SLIDE_PATHS = shuffle(ALL_SLIDE_PATHS,random_state=311)[int(DATA_SPLIT[0]*len(ALL_SLIDE_PATHS)):-int(DATA_SPLIT[2]*len(ALL_SLIDE_PATHS))]
TEST_SLIDE_PATHS = shuffle(ALL_SLIDE_PATHS,random_state=311)[-int(DATA_SPLIT[2]*len(ALL_SLIDE_PATHS)):]

TRAIN_ZIP = []
for path in TRAIN_SLIDE_PATHS:
    slide_name = path.split('/')[-1].replace('.tiff','').replace('.svs','')
    patient_path = '/'.join(path.split('/')[:-1]) + '/'
    for lbl in TRUE_LABEL_LIST :
        target_list = glob.glob(patient_path+f'{PATCH_NAME}/mask/{slide_name}*_p{lbl["label"]}.png')
        TRAIN_ZIP.extend(target_list) 
    for lbl in ZERO_LABEL_LIST :
        target_list = glob.glob(patient_path+f'{PATCH_NAME}/mask/{slide_name}*_p{lbl["label"]}.png')
        TRAIN_ZIP.extend(shuffle(target_list,random_state = 311)[:lbl["restrict"]])
TRAIN_ZIP = shuffle([('/'.join(x.split('/')[:-2])+'/image/'+re.sub('_p[0-9]','',x.split('/')[-1]),x) for x in TRAIN_ZIP],random_state=311)

VALID_ZIP = []
for path in VALID_SLIDE_PATHS:
    slide_name = path.split('/')[-1].replace('.tiff','').replace('.svs','')
    patient_path = '/'.join(path.split('/')[:-1]) + '/'
    for lbl in TRUE_LABEL_LIST :
        target_list = glob.glob(patient_path+f'{PATCH_NAME}/mask/{slide_name}*_p{lbl["label"]}.png')
        VALID_ZIP.extend(target_list) 
    for lbl in ZERO_LABEL_LIST :
        target_list = glob.glob(patient_path+f'{PATCH_NAME}/mask/{slide_name}*_p{lbl["label"]}.png')
        VALID_ZIP.extend(shuffle(target_list,random_state = 311)[:lbl["restrict"]])
VALID_ZIP = shuffle([('/'.join(x.split('/')[:-2])+'/image/'+re.sub('_p[0-9]','',x.split('/')[-1]),x) for x in VALID_ZIP],random_state=311)

TEST_ZIP = []
for path in TEST_SLIDE_PATHS:
    slide_name = path.split('/')[-1].replace('.tiff','').replace('.svs','')
    patient_path = '/'.join(path.split('/')[:-1]) + '/'
    
    for lbl in TRUE_LABEL_LIST :
        target_list = glob.glob(patient_path+f'{PATCH_NAME}/mask/{slide_name}*_p{lbl["label"]}.png')
        TEST_ZIP.extend(target_list) 
    
    for lbl in ZERO_LABEL_LIST :
        target_list = glob.glob(patient_path+f'{PATCH_NAME}/mask/{slide_name}*_p{lbl["label"]}.png')
        TEST_ZIP.extend(shuffle(target_list,random_state = 311)[:lbl["restrict"]])
        
TEST_ZIP = shuffle([('/'.join(x.split('/')[:-2])+'/image/'+re.sub('_p[0-9]','',x.split('/')[-1]),x) for x in TEST_ZIP],random_state=311)

ZERO_LABELS = [x['label'] for x in ZERO_LABEL_LIST]


# ------------------------------
# Prediction Config
# ------------------------------
NERVE_MODEL = 'deeplab'
TUMOR_MODEL = 'deeplab'

NERVE_MODEL_PATH = '/tf/model/0820_deeplabv3.hdf5'
TUMOR_MODEL_PATH = '/tf/model/0901_deeplab_xception_ish_tumor.hdf5'









init_params = {
    'level':LEVEL,
    'patch_size':PATCH_SIZE,
    'patch_name':PATCH_NAME
}

