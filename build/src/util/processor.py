import openslide, os, random, shutil, math, glob, tqdm
from skimage import io, morphology, measure, filters
from lxml import etree
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class slide_processor(object):
    def __init__(self,init_params):
        self.slide_path = init_params['slide_path']
        self.annotation = '' if init_params['xml_path']=='' else etree.parse(init_params['xml_path'])
        self.level = init_params['level']
        self.magnification = 100 if 'magnification' not in init_params.keys() else init_parmas['magnification']
        
        self.patient_path = '/'.join(self.slide_path.split('/')[:-1])
        self.patch_size = init_params['patch_size'] 
        self.patch_name = init_params['patch_name'] 
        
        self.slide = openslide.OpenSlide(self.slide_path)
        self.src_w, self.src_h = self.slide.level_dimensions[0]
        self.dest_w, self.dest_h = self.slide.level_dimensions[self.level]
        self.multiple = self.src_w//self.dest_w
        self.mpp = float(self.slide.properties.get('openslide.mpp-x'))
    # ------------------------------------------
    # Annotation File Parsing 
    # ------------------------------------------
    def get_annotation_dict(self):
        trees = self.annotation.getroot()[0]
        w_ratio = self.src_w / self.dest_w; h_ratio = self.src_h / self.dest_h
        bbox_dict = {}; coord_dict = {}
        
        for tree in trees:
            if tree.get('Type')=='Rectangle':
                group = tree.get('PartOfGroup')
                patch_group = 'p' + str(int(group.split('_')[0])) 
                is_test_region = group.split('_')[1]
                if is_test_region=='test':
                    patch_group = patch_group+ '_' + is_test_region
                bboxes = tree.findall('Coordinates')
                for bbox in bboxes:
                    pts = list()
                    coords = bbox.findall('Coordinate')
                    for coord in coords:
                        x = round(float(coord.get('X')))
                        y = round(float(coord.get('Y')))
                        x = np.clip(round(x/w_ratio),0,self.dest_w)
                        y = np.clip(round(y/h_ratio),0,self.dest_h)
                        pts.append((x,y))
                if patch_group in bbox_dict.keys():
                    bbox_dict[patch_group].append(pts)
                else:
                    bbox_dict[patch_group] = [pts]
            else:
                group = tree.get('PartOfGroup')
                patch_group = 'p' + str(int(group.split('_')[0]))
                regions = tree.findall('Coordinates')
                for region in regions:
                    pts = []
                    coords = region.findall('Coordinate')
                    for coord in coords:
                        x = round(float(coord.get('X')))
                        y = round(float(coord.get('Y')))
                        x = np.clip(round(x/w_ratio),0,self.dest_w)
                        y = np.clip(round(y/h_ratio),0,self.dest_h)
                        pts.append((x,y))
                if patch_group in coord_dict.keys():
                    coord_dict[patch_group].append(pts)
                else:
                    coord_dict[patch_group] = [pts]
        return {'bboxes':bbox_dict,'coords':coord_dict}
    # ------------------------------------------
    # Tissue Mask by remove_small_hole 
    # ------------------------------------------        
    def get_tissue_mask(self,area_thr=10000,RGB_min=0,show=False):
        hsv = cv.cvtColor(self.arr,cv.COLOR_RGB2HSV)
        background_R = self.arr[:, :, 0] > filters.threshold_otsu(self.arr[:, :, 0])
        background_G = self.arr[:, :, 1] > filters.threshold_otsu(self.arr[:, :, 1])
        background_B = self.arr[:, :, 2] > filters.threshold_otsu(self.arr[:, :, 2])
        
        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        tissue_S = hsv[:, :, 1] > filters.threshold_otsu(hsv[:, :, 1])
        
        min_R = self.arr[:, :, 0] > RGB_min
        min_G = self.arr[:, :, 1] > RGB_min
        min_B = self.arr[:, :, 2] > RGB_min

        mask = tissue_S & (tissue_RGB + min_R + min_G + min_B)
        ret = morphology.remove_small_holes(mask,area_threshold=area_thr)
        ret = np.array(ret).astype(np.uint8)
        if show==True:
            plt.figure(figsize = (14,10)); plt.imshow(ret)
        return cv.morphologyEx(ret*255,cv.MORPH_OPEN,cv.getStructuringElement(cv.MORPH_RECT,(5,5)))
    
    # --------------------------------
    # Annotation Mask를 key별로 구하기
    # --------------------------------
    def get_anno_mask(self,tool='etc',show=False):
        '''
        annotation dict 의 형태 : {'bboxes': , 'coords': } 
        ----
        bbox를 기준으로 
        bbox의 key ( class ) 에 해당하는 bbox를 bboxes에 append하고 
        동일한 key의 annotation이 있다면 coords에 append하고 없다면 bbox자체를 annotation에 넣는다...
        '''
        #prt = self.arr
        if tool=='asap':
            annotation_dict = self.get_annotation_dict()
        else:
            annotation_dict = self.get_annotation()
        i = 0
        mask = np.zeros((self.dest_h,self.dest_w))
        for key in annotation_dict['bboxes'].keys():
            lbl = 1 if '1' in key else 2 if '2' in key else 3 if '3' in key else 4 # tumor는 1, nerve는 2, pni(nerve)는 3, normal은 4에 mapping
            # Bounding Box안에 Annotation이 있다면 Annotation region 사용
            if key in annotation_dict['coords'].keys():
                regions = annotation_dict['coords'][key]
            # Bounding Box안에 Annotation이 없다면 Bbox Region 사용
            else:
                regions = annotation_dict['bboxes'][key]
                
            for region in regions:
                pts = [np.array(region,dtype=np.int32)]
                mask = cv.fillPoly(mask,pts,lbl)
                #prt = cv.drawContours(prt,pts,-1,(100*i,255-100*i,100*i),3)
        ret = mask
        if show==True:
            plt.figure(figsize = (14,10)); plt.imshow(prt)
        return ret
    # ------------------------------------------
    # Get Iteration Range
    # ------------------------------------------    
    def get_seq_range(self,slide_width, slide_height, multiple,img_size,s):
        y_seq = tqdm.trange(math.ceil(((slide_height*multiple) - int(img_size)) / int(s) + 1))
        x_seq = range(math.ceil(((slide_width*multiple) - int(img_size)) / int(s) + 1))
        return y_seq, x_seq
    # ------------------------------------------
    # Cell Ratio which is not background 
    # ------------------------------------------
    def get_ratio_mask(self,patch):
        h_,w_ = patch.shape[0], patch.shape[1]
        n_total = h_*w_
        n_cell  = np.count_nonzero(patch)
        if (n_cell != 0) :
            return n_cell*1.0/n_total*1.0
        else :
            return 0
    # ------------------------------------------
    # Save Patch in Filesystem
    # ------------------------------------------            
    def save_patch(self,dir_path,file_name,img):
        os.makedirs(dir_path,exist_ok=True)
        if np.max(img)!=0:
            cv.imwrite(os.path.join(dir_path,file_name),img)
    # ------------------------------------------
    # Plot & Save Patch when window is on. 
    # ------------------------------------------
    def execute_patch(self,patch_img,mask_img,patch_count,save_dir,slide_name,patch_name='p1',show=False,save=False):
        '''
        slide_name : patch가 어떤 slide로 부터 extract되었는지 tracking하기 위함 ( 동일 환자가 multi slide일 경우 환자 안에 패치 dir은 하나 )
        '''
        resize_image = cv.resize(patch_img,(self.patch_size,self.patch_size),cv.INTER_CUBIC)
        resize_mask = cv.resize(mask_img,(self.patch_size,self.patch_size),cv.INTER_CUBIC)
        #norm_image = stain_norm(resize_image) # patch에 대한 염색성 정규화
        
        if show==True:
            plt.figure();
            plt.subplot(1,2,1); plt.title('patch'); plt.imshow(resize_image)
            plt.subplot(1,2,2); plt.title(f'{patch_name}'); plt.imshow(resize_mask,vmin=0,vmax=255)
        if save==True:
            self.save_patch(save_dir+'/image',f'{slide_name}_{patch_count}.png',resize_image)
            self.save_patch(save_dir+'/mask',f'{slide_name}_{patch_count}_{patch_name}.png',np.clip(resize_mask,0,4))        
    
    # ------------------------------------------
    # Patch Extraction with Bounding box 
    # ------------------------------------------
    def get_patch_bbox(
        self,
        tool='asap',
        mag=100,
        patch_ratio = 0.3,
        patch_count = 0,
        overlap = 0.5,
        show=False,
        save=False
    ):
        '''
        tool : annotation tool ['asap','etc']
        mag : magnification for get 100x patch => mag = 100
        patch_ratio : annotation cell ratio for patch cell 
        patch_count : for multi slide per one patient.
        '''
        slide_name = self.slide_path.split('/')[-1].replace('.tiff','').replace('.svs','')
        patch_size_lv0 = round( round(self.patch_size/self.mpp) * (100/mag) )
        patch_size_lv2 = patch_size_lv0//self.multiple
        step = 1-overlap
        s = int(patch_size_lv0*step)
        save_path = os.path.join(self.patient_path,self.patch_name)
        print(f'SIZE_PATCH_0 : {patch_size_lv0} SIZE_PATCH_2 : {patch_size_lv2}')
        if patch_count!=0:
            print(patch_count)
        anno_mask = self.get_anno_mask(tool=tool)
        bbox_dict = self.get_annotation_dict()['bboxes']

        patch_count = patch_count
        for bbox_key in bbox_dict:
            # Test 용 Region은 Patch를 추출하지 않습니다.
            if 'test' in bbox_key:
                continue
            for bbox in bbox_dict[bbox_key]:
                patch_key = bbox_key
                min_x,min_y = np.min(bbox,axis = 0)
                max_x,max_y = np.max(bbox,axis = 0)
                slide_w = max_x-min_x; slide_h = max_y-min_y
                y_seq,x_seq = self.get_seq_range(slide_w,slide_h,self.multiple,patch_size_lv0,s)

                for y in y_seq:
                    for x in x_seq:
                        start_x = int(min_x + (s*x/self.multiple)); end_x = int(min_x + (s*(x+int(1/step))/self.multiple))
                        start_y = int(min_y + (s*y/self.multiple)); end_y = int(min_y + (s*(y+int(1/step))/self.multiple))
                        
                        img_patch = np.array(self.slide.read_region(
                            location = (int((min_x*self.multiple)+(s*x)),
                                       int((min_y*self.multiple)+(s*y))),
                            level=0,
                            size=(patch_size_lv0,patch_size_lv0)
                        ).convert('RGB')).astype(np.uint8)[...,:3]

                        patch_mask = anno_mask[start_y:end_y,start_x:end_x]
                        over_limit = 1.01 if patch_key=='p4' else 0.75 # 'p4'는 normal patch( no poly anno. )
                        if (self.get_ratio_mask(patch_mask)>patch_ratio) and (self.get_ratio_mask(patch_mask)<over_limit):
                            self.execute_patch(
                                img_patch,patch_mask,patch_count,save_path,slide_name=slide_name,patch_name=patch_key,show=show,save=save
                            )
                            patch_count+=1
        print(patch_count)
    
