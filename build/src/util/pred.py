from util.model import *
import numpy as np
import cv2 as cv
from util.processor import slide_processor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class multi_predictor(object):
    def __init__(self,init_params):
        self.mdl = build_seg_model(
            model = init_params['MODEL'],
            backbone = init_params['BACKBONE'],
            weight = init_params['WEIGHT'],
            activation = 'softmax',
            n_classes = 3
        )
        self.mdl.load_weights(init_params['MODEL_PATH'])
        self.mag = init_params['MAG']

    def predict_patch(self,patch,prob=0.5):
        patch_input = np.expand_dims(patch,axis=0)
        patch_input = patch_input.astype(np.float32)/255.0

        pred_prob = self.mdl.predict(patch_input)
        pred_nerve = pred_prob[...,2].squeeze()
        pred_tumor = pred_prob[...,1].squeeze() 

        pred_nerve = np.where(pred_nerve>prob,1,0)
        pred_tumor = np.where(pred_tumor>prob,1,0)

        return pred_nerve.astype(np.uint8), pred_tumor.astype(np.uint8)
    # -------------------------------------------------
    # Predict Region
    # -------------------------------------------------
    def predict_regions(self,slide_path,xml_path,kernel_size = 12, dilate_iter = 5, level = 5, overlap = 0, patch_size = 512,show=False):
        # Slide Initialization
        init_params = {'level':level,'patch_size':512,'patch_name':''}
        init_params.update({
            'slide_path':slide_path,
            'xml_path':xml_path
        })
        slide = slide_processor(init_params)
        prt = slide.arr # vis용
        ret_mask = np.zeros((slide.dest_h,slide.dest_w)) # 최종 return될 mask
        nerve_mask = np.zeros((slide.dest_h,slide.dest_w)) # slide 수준의 nerve pred mask
        tumor_mask = np.zeros((slide.dest_h,slide.dest_w)) # slide 수준의 tumor pred mask
        
        # Region Sliding
        anno_dict = slide.get_annotation_dict()
        bbox_dict = anno_dict['bboxes']
        coord_dict = anno_dict['coords']
        
        multiple = slide.src_h//slide.dest_h
        patch_size_lv0 = round( round(patch_size/slide.mpp) * (100/slide.magnification) )
        patch_size_lv2 = patch_size_lv0//multiple
        step = 1-overlap
        s = int(patch_size_lv0*step)       
        
        # Setting Legend
        pni_legend = mpatches.Patch(color=(0,1,1), label="Pred PNI")
        nerve_legend = mpatches.Patch(color=(1,0,1), label="Pred Nerve")
        tumor_legend = mpatches.Patch(color=(1,1,0), label="Pred Tumor")
        
        for key in coord_dict:
            for i in range(len(coord_dict[key])):
                cnt = np.reshape(np.array(coord_dict[key][i]),(-1,len(coord_dict[key][i]),2))
                if key=='p1':
                    prt = cv.drawContours(prt,cnt,-1,(255,0,0),1)
                elif key=='p2':
                    prt = cv.drawContours(prt,cnt,-1,(0,0,255),1)
                else:
                    prt = cv.drawContours(prt,cnt,-1,(0,255,0),1)
        
        for key in bbox_dict:
            prt = cv.drawContours(prt,np.array(bbox_dict[key]),-1,(0,0,0),2) # draw bbox
            for region in bbox_dict[key]:
                min_x, min_y = np.min(region,axis = 0)
                max_x, max_y = np.max(region,axis = 0)
                slide_w = max_x-min_x ; slide_h = max_y-min_y
                
                slide_w = patch_size_lv0//multiple if slide_w*multiple<patch_size_lv0 else slide_w
                slide_h = patch_size_lv0//multiple if slide_h*multiple<patch_size_lv0 else slide_h
                y_seq,x_seq = slide.get_seq_range(slide_w,slide_h,slide.multiple,patch_size_lv0,s)
                for y in y_seq:
                    for x in x_seq:
                        start_x = int(min_x+(s*x/multiple))
                        start_y = int(min_y+(s*y/multiple))
                        end_x = int(min_x+((s*(x+int(1/step)))/multiple))
                        end_y = int(min_y+((s*(y+int(1/step)))/multiple))   
                        
                        patch_box = np.expand_dims(np.array([[start_x,start_y],[end_x,start_y],[end_x,end_y],[start_x,end_y]]),0)
                        prt = cv.drawContours(prt,patch_box,-1,(100,100,100),1) # plot patch box

                        img_patch_0 = np.array(slide.slide.read_region(
                            location = (int((min_x*multiple) + (s*x)), int((min_y*multiple) + (s*y))),
                            level = 0,
                            size = (patch_size_lv0, patch_size_lv0)
                        ).convert('RGB')).astype(np.uint8)[...,:3]
                        ##
                        img_patch = cv.resize(img_patch_0,(patch_size,patch_size),cv.INTER_CUBIC)
                        pred_nerve_p, pred_tumor_p = self.predict_patch(img_patch)
                        nerve_mask[start_y:end_y,start_x:end_x] = np.logical_or(cv.resize(pred_nerve_p,(end_x-start_x,end_y-start_y)),nerve_mask[start_y:end_y,start_x:end_x])
                        tumor_mask[start_y:end_y,start_x:end_x] = np.logical_or(cv.resize(pred_tumor_p,(end_x-start_x,end_y-start_y)),tumor_mask[start_y:end_y,start_x:end_x])
        
        # 
        nerve_mask = nerve_mask.astype(np.uint8)
        tumor_mask = tumor_mask.astype(np.uint8)
        ## Merge Two Mask
        dilated_nerve = cv.dilate(
            nerve_mask.astype(np.uint8),
            cv.getStructuringElement(cv.MORPH_RECT,(kernel_size,kernel_size)),
            iterations=dilate_iter)
        nerve_cnts = cv.findContours(nerve_mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[0] 
        for c in nerve_cnts:
            single_space = np.zeros_like(nerve_mask)
            single_obj = cv.fillPoly(single_space,[c],1,1)
            single_obj_dilated = cv.dilate(
                single_obj,
                cv.getStructuringElement(cv.MORPH_RECT,(kernel_size,kernel_size)),
                iterations=dilate_iter
            )
            is_inter = np.any(np.logical_and(single_obj_dilated,tumor_mask))
            if is_inter:
                ret_mask = cv.fillPoly(ret_mask,[c],1,1)
                prt = cv.drawContours(prt,[c],-1,(0,255,255),1)
            else:
                prt = cv.drawContours(prt,[c],-1,(255,0,255),1)
                
                
        tumor_cnts = cv.findContours(tumor_mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[0]
        for c in tumor_cnts:
            prt = cv.drawContours(prt,[c],-1,(255,255,0),1)
        
        if show==True:
            plt.figure(figsize = (14,10)); plt.imshow(prt)
            plt.legend(handles=[pni_legend,nerve_legend,tumor_legend], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

        return ret_mask, prt

    # -------------------------------
    # Region Level Evaluation
    # -------------------------------
    def eval_regions(self,slide_path,xml_path,kernel_size = 12, dilate_iter = 5, level = 5, overlap = 0, patch_size = 512):
        '''
        모든 바운딩 박스에 대해
        ground truth가 pni인 바운딩 박스 부분의 예측값이 pni(val>0) 라면 pni box를 Detecting하였다고 판단 -> True Positive
        ground truth가 pni인 바운딩 박스 부분의 예측값이 zero라면 -> False Negative
        ground truth가 pni가 아닌 바운딩 박스 부분의 예측값이 zero라면 -> True Negative
        ground truth가 pni가 아닌 바운딩 박스의 예측값이 pni(val>0) 라면 -> False Positive
        '''
        tp = 0; fn = 0; fp = 0; tn = 0
        # Slide Initialization
        init_params = {'level':level,'patch_size':512,'patch_name':''}
        init_params.update({
            'slide_path':slide_path,
            'xml_path':xml_path
        })
        slide = slide_processor(init_params)
        
        bbox_dicts = slide.get_annotation_dict()['bboxes']
        pred_pni = self.predict_regions(
            slide_path,
            xml_path,
            kernel_size = kernel_size,
            dilate_iter = dilate_iter, 
            level = level, 
            overlap = overlap, 
            patch_size = patch_size
        )[0]
        
        for key in bbox_dicts:
            for bbox in bbox_dicts[key]:
                min_x,min_y = np.min(bbox,axis = 0) 
                max_x,max_y = np.max(bbox,axis = 0)
                pred_pni_bbox = pred_pni[min_y:max_y,min_x:max_x]
                if key=='p3' and np.max(pred_pni_bbox) >0 :
                    tp +=1
                elif key=='p3' and np.max(pred_pni_bbox) ==0 :
                    fn +=1
                elif key!='p3' and np.max(pred_pni_bbox)>0 :
                    fp +=1
                else:
                    tn +=1
        return (tp,fp,fn,tn)


class wsi_predictor(object):
    def __init__(self,init_params):
        self.nerve_mdl = build_seg_model(
            model = init_params['NERVE_MODEL'],
            backbone = init_params['NERVE_BACKBONE'],
            weight = init_params['NERVE_WEIGHT']
        )
        self.tumor_mdl = build_seg_model(
            model = init_params['TUMOR_MODEL'],
            backbone = init_params['TUMOR_BACKBONE'],
            weight = init_params['TUMOR_WEIGHT']
        )
        self.nerve_mdl.load_weights(init_params['NERVE_MODEL_PATH'])
        self.tumor_mdl.load_weights(init_params['TUMOR_MODEL_PATH'])
        self.nerve_mag = init_params['NERVE_MAG']
        self.tumor_mag = init_params['TUMOR_MAG']
        
    # -------------------------------
    # Patch 수준의 Prediction
    # -------------------------------
    def predict_patch(self,patch,prob=0.5):
        '''
        prob : prob보다 작은 확률이면 0으로 mapping , 큰 확률이면 1로 mapping
        output : nerve predict mask와 tumor predict mask 둘다 return
        '''
        patch_input = np.expand_dims(patch,axis=0) # Model에 Feeding하기 위한 shape
        patch_input = patch_input.astype(np.float32)/255.0 # Scaling
        #patch_input = patch.astype(np.float32)/255.0 # Scaling
        pred_nerve = np.squeeze(self.nerve_mdl.predict(patch_input))
        pred_tumor = np.squeeze(self.tumor_mdl.predict(patch_input))
        pred_nerve = np.where(pred_nerve>prob,1,0)
        pred_tumor = np.where(pred_tumor>prob,1,0)
        
        return pred_nerve.astype(np.uint8), pred_tumor.astype(np.uint8)
        
    # -------------------------------
    # Region level의 Prediction
    # -------------------------------
    def predict_regions(self,slide_path,xml_path,kernel_size = 12, dilate_iter = 5, level = 5, overlap = 0, patch_size = 512,show=False):
        # Slide Initialization
        init_params = {'level':level,'patch_size':512,'patch_name':''}
        init_params.update({
            'slide_path':slide_path,
            'xml_path':xml_path
        })
        slide = slide_processor(init_params)
        prt = slide.arr # vis용
        ret_mask = np.zeros((slide.dest_h,slide.dest_w)) # 최종 return될 mask
        nerve_mask = np.zeros((slide.dest_h,slide.dest_w)) # slide 수준의 nerve pred mask
        tumor_mask = np.zeros((slide.dest_h,slide.dest_w)) # slide 수준의 tumor pred mask
        
        # Region Sliding
        anno_dict = slide.get_annotation_dict()
        bbox_dict = anno_dict['bboxes']
        coord_dict = anno_dict['coords']
        
        multiple = slide.src_h//slide.dest_h
        patch_size_lv0 = round( round(patch_size/slide.mpp) * (100/slide.magnification) )
        patch_size_lv2 = patch_size_lv0//multiple
        step = 1-overlap
        s = int(patch_size_lv0*step)       
        
        # Setting Legend
        pni_legend = mpatches.Patch(color=(0,1,1), label="Pred PNI")
        nerve_legend = mpatches.Patch(color=(1,0,1), label="Pred Nerve")
        tumor_legend = mpatches.Patch(color=(1,1,0), label="Pred Tumor")
        
        for key in coord_dict:
            for i in range(len(coord_dict[key])):
                cnt = np.reshape(np.array(coord_dict[key][i]),(-1,len(coord_dict[key][i]),2))
                if key=='p1':
                    prt = cv.drawContours(prt,cnt,-1,(255,0,0),1)
                elif key=='p2':
                    prt = cv.drawContours(prt,cnt,-1,(0,0,255),1)
                else:
                    prt = cv.drawContours(prt,cnt,-1,(0,255,0),1)
        
        for key in bbox_dict:
            prt = cv.drawContours(prt,np.array(bbox_dict[key]),-1,(0,0,0),2) # draw bbox
            for region in bbox_dict[key]:
                min_x, min_y = np.min(region,axis = 0)
                max_x, max_y = np.max(region,axis = 0)
                slide_w = max_x-min_x ; slide_h = max_y-min_y
                if self.nerve_mag!=self.tumor_mag:
                    # Nerve Sliding
                    patch_size_lv0 = round( round(patch_size/slide.mpp) * (100/self.nerve_mag) )
                    s = int(patch_size_lv0*step)       
                    slide_w = patch_size_lv0//multiple if slide_w*multiple<patch_size_lv0 else slide_w
                    slide_h = patch_size_lv0//multiple if slide_h*multiple<patch_size_lv0 else slide_h
                    #print(f'Nerve Sliding Size in level0 : {patch_size_lv0} {slide_w} {slide_h}')
                    y_seq,x_seq = slide.get_seq_range(slide_w,slide_h,multiple,patch_size_lv0,s)
                    for y in y_seq:
                        for x in x_seq:
                            start_x = int(min_x+(s*x/multiple))
                            start_y = int(min_y+(s*y/multiple))
                            end_x = int(min_x+((s*(x+int(1/step)))/multiple))
                            end_y = int(min_y+((s*(y+int(1/step)))/multiple))   
                            #print(start_x,start_y,end_x,end_y)
                            patch_box = np.expand_dims(np.array([[start_x,start_y],[end_x,start_y],[end_x,end_y],[start_x,end_y]]),0)
                            prt = cv.drawContours(prt,patch_box,-1,(200,200,200),1) # plot patch box
                            
                            img_patch_0 = np.array(slide.slide.read_region(
                                location = (int((min_x*multiple) + (s*x)), int((min_y*multiple) + (s*y))),
                                level = 0,
                                size = (patch_size_lv0, patch_size_lv0)
                            ).convert('RGB')).astype(np.uint8)[...,:3]
                            ##
                            img_patch = cv.resize(img_patch_0,(patch_size,patch_size),cv.INTER_CUBIC)
                            pred_nerve_p, pred_tumor_p = self.predict_patch(img_patch)
                            nerve_mask[start_y:end_y,start_x:end_x] = np.logical_or(cv.resize(pred_nerve_p,(end_x-start_x,end_y-start_y)),nerve_mask[start_y:end_y,start_x:end_x])
                    
                    # Tumor Sliding
                    patch_size_lv0 = round( round(patch_size/slide.mpp) * (100/self.tumor_mag) )
                    s = int(patch_size_lv0*step)       
                    slide_w = patch_size_lv0//multiple if slide_w*multiple<patch_size_lv0 else slide_w
                    slide_h = patch_size_lv0//multiple if slide_h*multiple<patch_size_lv0 else slide_h
                    #print(f'Tumor Sliding Size in level0 : {patch_size_lv0}')
                    y_seq,x_seq = slide.get_seq_range(slide_w,slide_h,slide.multiple,patch_size_lv0,s)
                    for y in y_seq:
                        for x in x_seq:
                            start_x = int(min_x+(s*x/multiple))
                            start_y = int(min_y+(s*y/multiple))
                            end_x = int(min_x+((s*(x+int(1/step)))/multiple))
                            end_y = int(min_y+((s*(y+int(1/step)))/multiple))   
                            #print(start_x,start_y,end_x,end_y)
                            patch_box = np.expand_dims(np.array([[start_x,start_y],[end_x,start_y],[end_x,end_y],[start_x,end_y]]),0)
                            prt = cv.drawContours(prt,patch_box,-1,(100,100,100),1) # plot patch box
                            
                            img_patch_0 = np.array(slide.slide.read_region(
                                location = (int((min_x*multiple) + (s*x)), int((min_y*multiple) + (s*y))),
                                level = 0,
                                size = (patch_size_lv0, patch_size_lv0)
                            ).convert('RGB')).astype(np.uint8)[...,:3]
                            ##
                            img_patch = cv.resize(img_patch_0,(patch_size,patch_size),cv.INTER_CUBIC)
                            pred_nerve_p, pred_tumor_p = self.predict_patch(img_patch)
                            tumor_mask[start_y:end_y,start_x:end_x] = np.logical_or(cv.resize(pred_tumor_p,(end_x-start_x,end_y-start_y)),tumor_mask[start_y:end_y,start_x:end_x])

                else:
                    slide_w = patch_size_lv0//multiple if slide_w*multiple<patch_size_lv0 else slide_w
                    slide_h = patch_size_lv0//multiple if slide_h*multiple<patch_size_lv0 else slide_h
                    y_seq,x_seq = slide.get_seq_range(slide_w,slide_h,slide.multiple,patch_size_lv0,s)
                    for y in y_seq:
                        for x in x_seq:
                            start_x = int(min_x+(s*x/multiple))
                            start_y = int(min_y+(s*y/multiple))
                            end_x = int(min_x+((s*(x+int(1/step)))/multiple))
                            end_y = int(min_y+((s*(y+int(1/step)))/multiple))   
                            
                            patch_box = np.expand_dims(np.array([[start_x,start_y],[end_x,start_y],[end_x,end_y],[start_x,end_y]]),0)
                            prt = cv.drawContours(prt,patch_box,-1,(100,100,100),1) # plot patch box

                            img_patch_0 = np.array(slide.slide.read_region(
                                location = (int((min_x*multiple) + (s*x)), int((min_y*multiple) + (s*y))),
                                level = 0,
                                size = (patch_size_lv0, patch_size_lv0)
                            ).convert('RGB')).astype(np.uint8)[...,:3]
                            ##
                            img_patch = cv.resize(img_patch_0,(patch_size,patch_size),cv.INTER_CUBIC)
                            pred_nerve_p, pred_tumor_p = self.predict_patch(img_patch)
                            nerve_mask[start_y:end_y,start_x:end_x] = np.logical_or(cv.resize(pred_nerve_p,(end_x-start_x,end_y-start_y)),nerve_mask[start_y:end_y,start_x:end_x])
                            tumor_mask[start_y:end_y,start_x:end_x] = np.logical_or(cv.resize(pred_tumor_p,(end_x-start_x,end_y-start_y)),tumor_mask[start_y:end_y,start_x:end_x])
        
        # 
        nerve_mask = nerve_mask.astype(np.uint8)
        tumor_mask = tumor_mask.astype(np.uint8)
        ## Merge Two Mask
        dilated_nerve = cv.dilate(
            nerve_mask.astype(np.uint8),
            cv.getStructuringElement(cv.MORPH_RECT,(kernel_size,kernel_size)),
            iterations=dilate_iter)
        nerve_cnts = cv.findContours(nerve_mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[0] 
        for c in nerve_cnts:
            single_space = np.zeros_like(nerve_mask)
            single_obj = cv.fillPoly(single_space,[c],1,1)
            single_obj_dilated = cv.dilate(
                single_obj,
                cv.getStructuringElement(cv.MORPH_RECT,(kernel_size,kernel_size)),
                iterations=dilate_iter
            )
            is_inter = np.any(np.logical_and(single_obj_dilated,tumor_mask))
            if is_inter:
                ret_mask = cv.fillPoly(ret_mask,[c],1,1)
                prt = cv.drawContours(prt,[c],-1,(0,255,255),1)
            else:
                prt = cv.drawContours(prt,[c],-1,(255,0,255),1)
                
                
        tumor_cnts = cv.findContours(tumor_mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[0]
        for c in tumor_cnts:
            prt = cv.drawContours(prt,[c],-1,(255,255,0),1)
        
        if show==True:
            plt.figure(figsize = (14,10)); plt.imshow(prt)
            plt.legend(handles=[pni_legend,nerve_legend,tumor_legend], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

        return ret_mask, prt
        
    
    
    # -------------------------------
    # Slide 수준의 Prediction
    # -------------------------------
    
    def predict_slide(self,slide_path,xml_path = '',kernel_size = 12, dilate_iter = 5, level = 5,overlap=0,patch_size = 512):
        '''
        '''
        # Slide
        init_params = {'level':level,'patch_size':512,'patch_name':''}
        init_params.update({
            'slide_path':slide_path,
            'xml_path':xml_path
        })
        slide = slide_processor(init_params)
        prt = slide.arr # print 용
        ret_mask = np.zeros((slide.dest_h,slide.dest_w)) # 최종 return될 mask
        nerve_mask = np.zeros((slide.dest_h,slide.dest_w)) # slide 수준의 nerve pred mask
        tumor_mask = np.zeros((slide.dest_h,slide.dest_w)) # slide 수준의 tumor pred mask
        
        ## Sliding Condition
        
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
        #patch_size_lv0 = 2*patch_size; 
        patch_size_lv0 = round( round(patch_size/slide.mpp) * (100/slide.magnification) )
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
                
                img_patch_0 = np.array(slide.slide.read_region(
                    location = (int((min_x*multiple) + (s*x)), int((min_y*multiple) + (s*y))),
                    level = 0,
                    size = (patch_size_lv0, patch_size_lv0)
                ).convert('RGB')).astype(np.uint8)[...,:3]
                ##
                img_patch = cv.resize(img_patch_0,(patch_size,patch_size),cv.INTER_CUBIC)
                pred_nerve_p, pred_tumor_p = self.predict_patch(img_patch)
                nerve_mask[start_y:end_y,start_x:end_x] = cv.resize(pred_nerve_p,(end_x-start_x,end_y-start_y))
                tumor_mask[start_y:end_y,start_x:end_x] = cv.resize(pred_tumor_p,(end_x-start_x,end_y-start_y))
        nerve_mask = nerve_mask.astype(np.uint8)
        tumor_mask = tumor_mask.astype(np.uint8)
        ## Merge Two Mask
        dilated_nerve = cv.dilate(
            nerve_mask.astype(np.uint8),
            cv.getStructuringElement(cv.MORPH_RECT,(kernel_size,kernel_size)),
            iterations=dilate_iter)
        cnts = cv.findContours(nerve_mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[0] 
        for c in cnts:
            single_space = np.zeros_like(nerve_mask)
            single_obj = cv.fillPoly(single_space,[c],1,1)
            single_obj_dilated = cv.dilate(
                single_obj,
                cv.getStructuringElement(cv.MORPH_RECT,(kernel_size,kernel_size)),
                iterations=dilate_iter
            )
            is_inter = np.any(np.logical_and(single_obj_dilated,tumor_mask))
            if is_inter:
                ret_mask = cv.fillPoly(ret_mask,[c],1,1)
            else:
                continue
        return ret_mask
    
    # -------------------------------
    # Region Level Evaluation
    # -------------------------------
    def eval_regions(self,slide_path,xml_path,kernel_size = 12, dilate_iter = 5, level = 5, overlap = 0, patch_size = 512):
        '''
        모든 바운딩 박스에 대해
        ground truth가 pni인 바운딩 박스 부분의 예측값이 pni(val>0) 라면 pni box를 Detecting하였다고 판단 -> True Positive
        ground truth가 pni인 바운딩 박스 부분의 예측값이 zero라면 -> False Negative
        ground truth가 pni가 아닌 바운딩 박스 부분의 예측값이 zero라면 -> True Negative
        ground truth가 pni가 아닌 바운딩 박스의 예측값이 pni(val>0) 라면 -> False Positive
        '''
        tp = 0; fn = 0; fp = 0; tn = 0
        # Slide Initialization
        init_params = {'level':level,'patch_size':512,'patch_name':''}
        init_params.update({
            'slide_path':slide_path,
            'xml_path':xml_path
        })
        slide = slide_processor(init_params)
        
        bbox_dicts = slide.get_annotation_dict()['bboxes']
        pred_pni = self.predict_regions(
            slide_path,
            xml_path,
            kernel_size = kernel_size,
            dilate_iter = dilate_iter, 
            level = level, 
            overlap = overlap, 
            patch_size = patch_size
        )[0]

        for key in bbox_dicts:
            for bbox in bbox_dicts[key]:
                min_x,min_y = np.min(bbox,axis = 0) 
                max_x,max_y = np.max(bbox,axis = 0)
                pred_pni_bbox = pred_pni[min_y:max_y,min_x:max_x]
                if key=='p3' and np.max(pred_pni_bbox) >0 :
                    tp +=1
                elif key=='p3' and np.max(pred_pni_bbox) ==0 :
                    fn +=1
                elif key!='p3' and np.max(pred_pni_bbox)>0 :
                    fp +=1
                else:
                    tn +=1
        return (tp,fp,fn,tn)
            
