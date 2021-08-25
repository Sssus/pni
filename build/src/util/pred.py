from util.model import *
import numpy as np
import cv2 as cv
from util.processor import slide_processor

class wsi_predictor(object):
    def __init__(self,init_params):
        self.nerve_mdl = build_seg_model(model='deeplab')
        self.tumor_mdl = build_seg_model(model='deeplab')
        self.nerve_mdl.load_weights(init_params['nerve_model_path'])
        self.tumor_mdl.load_weights(init_params['tumor_model_path'])
        
    # -------------------------------
    # Patch 수준의 Prediction
    # -------------------------------
    def predict_patch(self,patch,prob=0.5):
        '''
        prob : prob보다 작은 확률이면 0으로 mapping , 큰 확률이면 1로 mapping
        output : nerve predict mask와 tumor predict mask 둘다 return
        '''
        patch_input = np.expand_dims(patch,axis=0) # Model에 Feeding하기 위한 shape
        patch_input./=255 # Scaling
        pred_nerve = np.squeeze(self.nerve_mdl.predict(patch_input))
        pred_tumor = np.squeeze(self.tumor_mdl.predict(patch_input))
        pred_nerve = np.where(pred_nerve>prob,pred_nerve,0)
        pred_tumor = np.where(pred_tumor>prob,pred_tumor,0)
        
        return pred_nerve, pred_tumor
        
        
    # -------------------------------
    # Slide 수준의 Prediction
    # -------------------------------
    
    def predict_slide(slide_path,overlap=0,patch_size = 512,):
        '''
        '''
        # Slide
        init_params.update({'slide_path':slide_path,})
        slide = slide_processor(init_params)
        
        ret_mask = np.zeros((slide.dest_h,slide.dest_w)) # 최종 return될 mask
        nerve_mask = np.zeros((slide.dest_h,slide.dest_w)) # slide 수준의 nerve pred mask
        tumor_mask = np.zeros((slide.dest_h,slide.dest_w)) # slide 수준의 tumor pred mask
        tissue_mask = slide.get_tissue_mask()
        tissue_mask = tissue_mask = cv.morphologyEx(tissue_mask,cv.MORPH_OPEN,cv.getStructuringElement(cv.MORPH_RECT,(5,5)),iterations=1)
        
        min_y , min_x = slide.arr.shape[:2]
        max_x , max_y = (0,0)
        coord = np.where(tissue_mask>0) # bbox sliding이 없다면 tissue에서 sliding
        if np.min(coord[1])<min_x: min_x = np.min(coord[1])
        if np.max(coord[1])>max_x: max_x = np.max(coord[1])
        if np.min(coord[0])<min_y: min_y = np.min(coord[0])
        if np.max(coord[0])>max_y: max_y = np.max(coord[0])
        
        ## Prepare Sliding
        slide_w = max_x-min_x
        slide_h = max_y-min_y
        step = 1-overlap
        multiple = slide.src_h//slide.dest_h
        patch_size = 512; 
        patch_size_lv0 = 2*patch_size; 
        patch_size_lv2 = patch_size_lv0//multiple
        
        s = int(patch_size_lv0*step)       
        y_seq,x_seq = slide.get_seq_range(slide_w,slide_h,multiple,patch_size_lv0,s)

        ## Sliding
        for y in y_seq:
            for x in x_seq:
                start_x = int(min_x+(s*x/multiple))
                start_y = int(min_y+(s*y/multiple))
                end_x = int(min_x+((s*(x+int(1/step)))/multiple))
                end_y = int(min_y+((s*(y+int(1/step)))/multiple))   
                
                img_patch = np.array(slide.slide.read_region(
                    location = (int((min_x*multiple) + (s*x)), int((min_y*multiple) + (s*y))),
                    level = 0,
                    size = (patch_size_lv0, patch_size_lv0)
                ).convert('RGB')).astype(np.uint8)[...,:3]
                
                

class predictor(object):
    def __init__(self,init_params):
        self.clf_model = build_clf_model(1)
        self.seg_model = build_seg_model(512,3,1,backbone=init_params['nerve_backbone'])
        self.seg_tumor_model = build_seg_model(512,3,1,backbone=init_params['tumor_backbone'])
        
        self.clf_model.load_weights(init_params['clf_model_path'])
        self.seg_model.load_weights(init_params['seg_model_path'])
        self.seg_tumor_model.load_weights(init_params['seg_tumor_model_path'])
        
    def segment_nerve(self,patch,prob=0.5):
        patch_input = np.expand_dims(patch,axis=0)
        pred_mask = np.squeeze(self.seg_model.predict(patch_input/255.0))
        pred_mask_filter = np.where(pred_mask>prob,255,0)
        return pred_mask_filter
    
    def segment_tumor(self,patch,prob=0.5):
        patch_input = np.expand_dims(patch,axis=0)
        pred_mask = np.squeeze(self.seg_tumor_model.predict(patch_input/255.0))
        pred_mask_filter = np.where(pred_mask>prob,255,0)
        return pred_mask_filter
    
    def classify_tumor(self,patch):
        patch_input = np.expand_dims(patch,axis=0)
        pred_label = self.clf_model.predict(patch_input.astype(np.float32)/255.0)
        return pred_label

    def pred_patch(self,patch,dest_h,dest_w,mixed=False):
        '''
        input : image_rgb 
        output : boundary_mask
        '''
        ret = np.zeros((dest_h,dest_w))
        ratio = patch.shape[0]//dest_h
        pred_label = self.classify_tumor(patch)
        if pred_label>0.5:  ## Patch Cell is Tumor Cell
            pred_nerve = self.segment_nerve(patch)
            nerve_cnts = cv.findContours(pred_nerve.astype(np.uint8), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[0]
            if len(nerve_cnts)>0:
                ## nerve boundary draw
                for c in nerve_cnts:
                    ret = cv.drawContours(ret,[c//ratio],-1,1,1)
                ret[0,:] = 0; ret[-1,:] = 0; ret[:,0] = 0 ; ret[:,-1] = 0 
                
                ## tumor dilation
                if mixed==True:
                    pred_tumor = self.segment_tumor(patch)
                    dilated_tumor = cv.dilate(pred_tumor.astype(np.uint8),cv.getStructuringElement(cv.MORPH_RECT,(12,12)),iterations=8)
                    dilated_tumor = cv.resize(dilated_tumor,(dest_w,dest_h))
                    ret=np.logical_and(ret, dilated_tumor)
            else:
                pass
        else:
            pass
        return ret

