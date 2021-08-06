from cfg import *
from util import processor
import glob, os
import multiprocessing

def extract(init_params):
    slide = processor(init_params)
    slide.get_patch(save=True,tool='asap',feature='global')

if __name__=='__main__':
    for slide_path in glob.glob(GLOB_PATH+'*/*.svs'):
        tissue=slide_path.split('/')[-2] ; num = slide_path[-8:-4]
        t = os.path.exists('./patch_overlap_global_512_stain/' + tissue + '/' + num)
        if t==False:
            init_params.update({'svs_path':slide_path,'xml_path':slide_path[:-3]+'xml'})
            p = multiprocessing.Process(target=extract,args=(init_params,))
            p.start()
        else:
            continue
    

