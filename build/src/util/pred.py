from model import *
import numpy as np
import cv2 as cv

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

