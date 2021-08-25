from util.cfg import *
from util.processor import slide_processor
import glob, os
import multiprocessing

def extract(init_params):
    slide = slide_processor(init_params)
    slide.get_patch(save=True,tool='asap',feature='global')

if __name__=='__main__':
    for patient_path in glob.glob('/data/*/PAIP/???'):
        patch_exist = os.path.exists(os.path.join(patient_path,'tumor_patches'))
        if patch_exist==False:
            svs_path = glob.glob(os.path.join(patient_path,'*.svs'))[0]
            xml_path = svs_path[:-4] + '_tumor.xml'
            init_params.update({
                'patient_path':patient_path,
                'svs_path':svs_path,
                'xml_path':xml_path,
                'patch_name':'tumor_patches'
            })
            p = multiprocessing.Process(target = extract,args = (init_params,))
            p.start()
        else:
            continue
    

