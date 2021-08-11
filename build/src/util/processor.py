import openslide
import numpy as np
import os, random, shutil, math, glob
import scipy.misc, scipy.ndimage
from skimage import io, morphology, measure, filters
from lxml import etree
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import tqdm
from util.stain_normalization import *

class slide_processor(object):
    def __init__(self,init_params):
        self.slide_path = init_params['svs_path']
        self.annotation = '' if init_params['xml_path']=='' else etree.parse(init_params['xml_path'])
        self.level = init_params['level']
        self.overlap = init_params['overlap'] # patch overlap rate
        self.patch_size = init_params['patch_size']
        self.patient_path = init_params['patient_path']
        self.patch_name = init_params['patch_name']
        
        self.slide = openslide.OpenSlide(self.slide_path)
        self.src_w, self.src_h = self.slide.level_dimensions[0]
        self.dest_w, self.dest_h = self.slide.level_dimensions[self.level]
        self.multiple = self.src_w//self.dest_w
        self.mpp = float(self.slide.properties.get('openslide.mpp-x'))
        self.arr = np.array(self.slide.read_region((0,0),self.level,size = (self.dest_w,self.dest_h)).convert('RGB'))
        

    def get_annotation_asap(self):
        ret = dict()
        trees = self.annotation.getroot()[0]
        w_ratio = self.src_w / self.dest_w; h_ratio = self.src_h / self.dest_h

        for tree in trees:
            try:
                group = 'p_'+str(int(tree.get('PartOfGroup').split('_')[-1])+1)
            except Exception as e:
                #group = 'p_5'
                print(e)
            if tree.get('Type') == 'Rectangle':
                group = 'p_6'
                regions = tree.findall('Coordinates')
                for region in regions:
                    coordinates = region.findall('Coordinate')
                    pts = list()
                    for coord in coordinates:
                        x = round(float(coord.get('X')))
                        y = round(float(coord.get('Y')))
                        x = np.clip(round(x/w_ratio),0,self.dest_w)
                        y = np.clip(round(y/h_ratio),0,self.dest_h)
                        pts.append((x,y))
                    if group in ret.keys():
                        ret[group].append(pts)
                    else:
                        ret[group] = [pts]
            else:
                regions = tree.findall('Coordinates')
                for region in regions:
                    coordinates = region.findall('Coordinate')
                    pts = list()
                    for coord in coordinates:
                        x = round(float(coord.get('X')))
                        y = round(float(coord.get('Y')))
                        x = np.clip(round(x/w_ratio),0,self.dest_w)
                        y = np.clip(round(y/h_ratio),0,self.dest_h)
                        pts.append((x,y))
                    if group in ret.keys():
                        ret[group].append(pts)
                    else:
                        ret[group] = [pts]
        return ret
    
    def get_annotation(self):
        ret = dict()
        w_ratio = self.src_w / self.dest_w; h_ratio = self.src_h / self.dest_h
        trees = self.annotation.getroot()
        
        ## Tree Parsing ( It can be modified for different annotation structure )
        for tree in trees:
            label = int(tree.get('Id'))
            group = tree.get('Name')
            if group=='nerve without tumor':
                group='p_1'
            elif group=='perineural invasion junction':
                group='p_2'
            elif group=='tumor without nerve':
                group='p_3'
            else:
                group='p_4'
            regions = tree.findall('Regions')[0]
            for region in regions.findall('Region'):
                pts = list()
                vertices =region.findall('Vertices')[0]
                for vertex in vertices.findall('Vertex'):
                    x = round(float(vertex.get('X')))
                    y = round(float(vertex.get('Y')))
                    x = np.clip(round(x/w_ratio),0,self.dest_w)
                    y = np.clip(round(y/h_ratio),0,self.dest_h)
                    pts.append((x,y))
                if group in ret.keys():
                    ret[group].append(pts)
                else:
                    ret[group] = [pts]
        return ret
    
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
    
    def get_anno_mask(self,tool='etc',show=False):
        ret = dict()
        if tool=='asap':
            annotation_dict = self.get_annotation_asap()
        else:
            annotation_dict = self.get_annotation()
        for key in annotation_dict.keys():
            mask = np.zeros((self.dest_h,self.dest_w))
            regions = annotation_dict[key]
            for region in regions:
                pts = [np.array(region,dtype=np.int32)]
                if len(region)!=4 and key=='p_2':
                    mask = cv.fillPoly(mask,pts,255)
                elif len(region)!=4:
                    mask = cv.fillPoly(mask,pts,255)
                elif len(region)==4 and key!='p_1' :
                    mask = cv.fillPoly(mask,pts,255)
                else:
                    pass
            if show==True:
                plt.figure(figsize = (14,10)); plt.title(key); plt.imshow(mask)
            ret[key] = mask
        return ret
    
    def get_seq_range(self,slide_width, slide_height, multiple,img_size,s):
        y_seq = tqdm.trange(int(((slide_height*multiple) - int(img_size)) // int(s) + 1))
        x_seq = range(int(((slide_width*multiple) - int(img_size)) // int(s) + 1))
        return y_seq, x_seq

    def get_ratio_mask(self,patch):
        h_,w_ = patch.shape[0], patch.shape[1]
        n_total = h_*w_
        n_cell  = np.count_nonzero(patch)
        if (n_cell != 0) :
            return n_cell*1.0/n_total*1.0
        else :
            return 0
            
    def save_patch(self,dir_path,file_name,img):
        os.makedirs(dir_path,exist_ok=True)
        if np.max(img)!=0:
            cv.imwrite(os.path.join(dir_path,file_name),img)

    def execute_patch(self,patch_img,mask_img,patch_count,save_dir,name='p1',show=False,save=False):
        '''
        patch_img inputted is RGB image
        '''
        resize_image = cv.resize(patch_img,(self.patch_size,self.patch_size),cv.INTER_CUBIC)
        resize_mask = cv.resize(mask_img,(self.patch_size,self.patch_size),cv.INTER_CUBIC)
        #norm_image = stain_norm(resize_image)
        
        if show==True:
            plt.figure(); plt.title('patch'); plt.imshow(resize_image)
            plt.figure(); plt.title(f'{name}'); plt.imshow(resize_mask,vmin=0,vmax=255)
        if save==True:
            self.save_patch(save_dir+'/image',f'{patch_count}.png',resize_image)
            self.save_patch(save_dir+'/mask',f'{patch_count}_{name}.png',np.clip(resize_mask,0,1))        
    
    def get_patch(self, tool='etc', feature = 'global', show=False, save=False):
        
        # Patch Size for lv0, lv2 
        # patch_lv0 must be bigger than patch_size to avoid vanishing resolution 
        if feature=='global':
            patch_size_lv0 = self.patch_size * 2
        else:
            patch_size_lv0 = self.patch_size
        patch_size_lv2 = patch_size_lv0//self.multiple    
        patch_size_origin = patch_size_lv0*self.mpp
        
        save_path = os.path.join(self.patient_path,self.patch_name)
        
        tissue_mask = self.get_tissue_mask(area_thr=100000,show=False)
        anno_mask = self.get_anno_mask(tool=tool,show=False)

        print(f'SIZE_PATCH_0 : {patch_size_lv0} SIZE_PATCH_2 : {patch_size_lv2} SIZE_PATCH_ORIGIN : {patch_size_origin}')

        step = 1-self.overlap 
        s = int(patch_size_lv0*step)
        patch_count=0
        print(anno_mask.keys())
        min_y , min_x = self.arr.shape[:2]
        max_x , max_y = (0,0)
        
        for key in anno_mask.keys():
            coord = np.where(anno_mask[key]>0)
            if np.min(coord[1])<min_x: min_x = np.min(coord[1])
            if np.max(coord[1])>max_x: max_x = np.max(coord[1])
            if np.min(coord[0])<min_y: min_y = np.min(coord[0])
            if np.max(coord[0])>max_y: max_y = np.max(coord[0])
        
        slide_w = max_x-min_x; slide_h = max_y-min_y
        print(f'Min x : {min_x} Min y : {min_y} Max x : {max_x} Max y : {max_y}')
        print(f'Slide Width : {slide_w} Slide Height : {slide_h} Sliding : {s}')
        y_seq,x_seq = self.get_seq_range(slide_w,slide_h,self.multiple,patch_size_lv0,s)
        
        for y in y_seq:
            for x in x_seq:
                start_x = int(min_x+(s*x/self.multiple)); end_x = int(min_x+((s*(x+int(1/step)))/self.multiple))
                start_y = int(min_y+(s*y/self.multiple)); end_y = int(min_y+((s*(y+int(1/step)))/self.multiple))
                # this img patch is 'RGB' image
                img_patch = np.array(self.slide.read_region(
                    location=(int((min_x*self.multiple)+(s*x)),
                             int((min_y*self.multiple)+(s*y))),
                    level=0, size = (patch_size_lv0,patch_size_lv0)
                ).convert('RGB')).astype(np.uint8)[...,:3]
                
                tissue_mask_patch = tissue_mask[start_y:end_y,start_x:end_x]
                if 'p_1' in anno_mask.keys(): 
                    p1_patch = anno_mask['p_1'][start_y:end_y,start_x:end_x] 
                    if (self.get_ratio_mask(p1_patch) > 0.05):
                        self.execute_patch(img_patch,p1_patch,patch_count,save_path,name='p1',show=show,save=save); patch_count+=1    
                if 'p_2' in anno_mask.keys(): 
                    p2_patch = anno_mask['p_2'][start_y:end_y,start_x:end_x]
                    if (self.get_ratio_mask(p2_patch) > 0.05):
                        self.execute_patch(img_patch,p2_patch,patch_count,save_path,name='p2',show=show,save=save); patch_count+=1    
                if 'p_3' in anno_mask.keys():
                    p3_patch = anno_mask['p_3'][start_y:end_y,start_x:end_x]
                    if (self.get_ratio_mask(p3_patch) == 1.0):
                        self.execute_patch(img_patch,p3_patch,patch_count,save_path,name='p3',show=show,save=save); patch_count+=1
                if 'p_4' in anno_mask.keys():
                    p4_patch = anno_mask['p_4'][start_y:end_y,start_x:end_x]
                    if (self.get_ratio_mask(p4_patch) == 1.0):
                        self.execute_patch(img_patch,p4_patch,patch_count,save_path,name='p4',show=show,save=save); patch_count+=1
                if 'p_5' in anno_mask.keys() and 'p_6' in anno_mask.keys():
                    p5_patch = anno_mask['p_5'][start_y:end_y,start_x:end_x]
                    p6_patch = anno_mask['p_6'][start_y:end_y,start_x:end_x]
                    if (self.get_ratio_mask(p5_patch) > 0.05 and self.get_ratio_mask(p6_patch)==1.0):
                        self.execute_patch(img_patch,p5_patch,patch_count,save_path,name='p5',show=show,save=save); patch_count+=1
    
        
                    
                
