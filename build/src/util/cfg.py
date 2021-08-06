import glob
from sklearn.utils import shuffle

init_params = {
    'level':2,
    'overlap':0.5,
    'patch_size':512,
    'patch_dir':'./patch_overlap_global_512_stain/'
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