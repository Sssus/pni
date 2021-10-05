from cfg import *
from util.processor import slide_processor
import glob, os
import multiprocessing

def extract_patient(patient_path,init_params): 
    '''
    multi slide의 patient의 경우 multi processing에 extract_slide를 넣으면 같은 patch directory에 patch를 넣는 과정에서 충돌이 날 수 있어서 patient단위로 multiprocessing에 태움
    '''
    slide_paths = glob.glob(os.path.join(patient_path,f'*{SLIDE_FORMAT}'))
    cnt = 0
    for slide_path in slide_paths:
        xml_path = '.'.join(slide_path.split('.')[:-1]) + XML_NAME
        xml_exist = os.path.exists(xml_path)

        if xml_exist==True: # xml pair가 존재할 경우 patch extraction 단계로
            patch_exist = os.path.exists(os.path.join(patient_path,PATCH_NAME))

            if patch_exist==False: # multi slide일 경우 patch를 그다음순번부터 시작
                patch_cnt = 0
            else:
                patch_cnt = len(glob.glob(os.path.join(patient_path,PATCH_NAME,'image') + '/*'))

            init_params.update({
                'slide_path':slide_path,
                'xml_path':xml_path
            })
            extract_slide(init_params,patch_cnt)
        else:
            cnt+=1
            continue
            
        cnt+=1
        print(f'{cnt/len(slide_paths)} Done')
            

def extract_slide(init_params,patch_cnt):
    slide = slide_processor(init_params)
    slide.get_patch_bbox(
        save=True,
        tool='asap',
        mag=MAGNIFICATION,
        overlap = OVERLAP,
        patch_count=patch_cnt,
        patch_ratio=0.1
    )

if __name__=='__main__':
    init_params = {
        'level':LEVEL,
        'patch_size':PATCH_SIZE,
        'patch_name':PATCH_NAME
    }
    for patient_path in ALL_PATIENT_PATHS:
        p = multiprocessing.Process(target=extract_patient,args = (patient_path,init_params))
        p.start()
        
                
        

