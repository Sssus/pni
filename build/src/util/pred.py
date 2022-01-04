from util.model import *
import numpy as np
import cv2 as cv
from util.processor import slide_processor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, math

class wsi_predictor(object):
    def __init__(self,init_params):
        self.nerve_mdl = build_seg_model(
            model = init_params['NERVE_MODEL'],
            backbone = init_params['NERVE_BACKBONE'],
            weight = init_params['NERVE_WEIGHT'],
            decoder = 'upsampling' if 'NERVE_DECODER' not in init_params.keys() else init_params['NERVE_DECODER']
        ) if 'NERVE_MODEL' in init_params.keys() else ''
        self.tumor_mdl = build_seg_model(
            model = init_params['TUMOR_MODEL'],
            backbone = init_params['TUMOR_BACKBONE'],
            weight = init_params['TUMOR_WEIGHT'],
            decoder = 'upsampling' if 'TUMOR_DECODER' not in init_params.keys() else init_params['TUMOR_DECODER']
        ) if 'TUMOR_MODEL' in init_params.keys() else ''
        self.multi_mdl = build_seg_model(
            model = init_params['MULTI_MODEL'],
            backbone=init_params['MULTI_BACKBONE'],
            weight = init_params['MULTI_WEIGHT'],
            decoder = 'upsampling',
            activation = 'softmax',
            n_classes = 3
        ) if 'MULTI_MODEL' in init_params.keys() else ''
        if self.nerve_mdl!='' : self.nerve_mdl.load_weights(init_params['NERVE_MODEL_PATH']) ;self.nerve_mag = init_params['NERVE_MAG']
        if self.tumor_mdl!='' : self.tumor_mdl.load_weights(init_params['TUMOR_MODEL_PATH']) ;self.tumor_mag = init_params['TUMOR_MAG']
        if self.multi_mdl!='' : self.multi_mdl.load_weights(init_params['MULTI_MODEL_PATH']) ;self.multi_mag = init_params['MULTI_MAG']
        
    
    # -------------------------------
    # Post Processing Probability Map => Binary Map
    # -------------------------------
    def post_process(self,probability_map,method='otsu',prob=0.5,kernel_size=32):
        if method=='otsu':
            binary_map = cv.threshold((probability_map*255).astype(np.uint8),127,255,cv.THRESH_BINARY|cv.THRESH_OTSU)[1]
        else:
            binary_map = np.where(probability_map>prob,1,0)
        
        open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernel_size,kernel_size))
        binary_map = cv.morphologyEx(binary_map.astype(np.uint8),cv.MORPH_OPEN,open_kernel)
        return binary_map

    # -------------------------------
    # Patch 수준의 Prediction
    # -------------------------------

    def predict_patch(self,patch,method='otsu',prob=0.5,remove_kernel_size=32):
        '''
        prob : prob보다 작은 확률이면 0으로 mapping , 큰 확률이면 1로 mapping
        output : nerve predict mask와 tumor predict mask 둘다 return
        output : probability map
        '''
        patch_input = np.expand_dims(patch,axis=0) # Model에 Feeding하기 위한 shape
        patch_input = patch_input.astype(np.float32)/255.0 # Scaling
        
        if self.multi_mdl=='':
            pred_nerve = np.squeeze(self.nerve_mdl.predict(patch_input))
            pred_tumor = np.squeeze(self.tumor_mdl.predict(patch_input))
        else:
            pred_prob = self.multi_mdl.predict(patch_input)
            pred_nerve = pred_prob[...,2].squeeze()
            pred_tumor = pred_prob[...,1].squeeze() 
        #pred_nerve = self.post_process(pred_nerve,method=method,prob=prob,kernel_size=remove_kernel_size)
        #pred_tumor = self.post_process(pred_tumor,method=method,prob=prob,kernel_size=remove_kernel_size)
        #return pred_nerve.astype(np.uint8), pred_tumor.astype(np.uint8)
        return pred_nerve, pred_tumor
    # -------------------------------
    # Region level의 Prediction (Probability Map)
    # -------------------------------
    def predict_map(self,slide_path,xml_path,level=5,overlap=0,patch_size=512):
        # Slide Initialization
        init_params = {'level':level,'patch_size':512,'patch_name':''}
        init_params.update({
            'slide_path':slide_path,
            'xml_path':xml_path
        })
        slide = slide_processor(init_params)
        prt = np.array(slide.slide.read_region((0,0),slide.level,size = (slide.dest_w,slide.dest_h)).convert('RGB')) # vis용
        ret_mask = np.zeros((slide.dest_h,slide.dest_w)) # 최종 return될 mask
        nerve_map = np.zeros((slide.dest_h,slide.dest_w)) # slide 수준의 nerve pred mask
        tumor_map = np.zeros((slide.dest_h,slide.dest_w)) # slide 수준의 tumor pred mask
        overlay = prt.copy()
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
                    #prt = cv.drawContours(prt,cnt,-1,(255,0,0),1)
                    overlay = cv.fillPoly(overlay,cnt,(255,0,0),1)
                elif key=='p2':
                    #prt = cv.drawContours(prt,cnt,-1,(0,0,255),1)
                    overlay = cv.fillPoly(overlay,cnt,(255,127,0),1)
                else:
                    #prt = cv.drawContours(prt,cnt,-1,(0,255,0),1)
                    overlay = cv.fillPoly(overlay,cnt,(0,0,255),1)
        
        for key in bbox_dict:
            if 'test' not in key:
                continue
            #prt = cv.drawContours(prt,np.array(bbox_dict[key]),-1,(0,0,0),1) # draw bbox
            for region in bbox_dict[key]:
                min_x, min_y = np.min(region,axis = 0)
                max_x, max_y = np.max(region,axis = 0)
                slide_w = max_x-min_x ; slide_h = max_y-min_y
                
                print(slide_w,slide_h,slide.multiple,patch_size_lv0,s)
                slide_w = patch_size_lv0//multiple if slide_w*multiple<patch_size_lv0 else slide_w
                slide_h = patch_size_lv0//multiple if slide_h*multiple<patch_size_lv0 else slide_h
                print(slide_w,slide_h)
                y_seq,x_seq = slide.get_seq_range(slide_w,slide_h,slide.multiple,patch_size_lv0,s)
                for y in y_seq:
                    for x in x_seq:
                        start_x = int(min_x+(s*x/multiple))
                        start_y = int(min_y+(s*y/multiple))
                        end_x = int(min_x+((s*(x+int(1/step)))/multiple))
                        end_y = int(min_y+((s*(y+int(1/step)))/multiple))   

                        patch_box = np.expand_dims(np.array([[start_x,start_y],[end_x,start_y],[end_x,end_y],[start_x,end_y]]),0)
                        #prt = cv.drawContours(prt,patch_box,-1,(100,100,100),1) # plot patch box

                        img_patch_0 = np.array(slide.slide.read_region(
                            location = (int((min_x*multiple) + (s*x)), int((min_y*multiple) + (s*y))),
                            level = 0,
                            size = (patch_size_lv0, patch_size_lv0)
                        ).convert('RGB')).astype(np.uint8)[...,:3]
                        ##
                        img_patch = cv.resize(img_patch_0,(patch_size,patch_size),cv.INTER_CUBIC)
                        pred_nerve_p, pred_tumor_p = self.predict_patch(
                            img_patch,
                        ) ####################

                        tumor_map[start_y:end_y,start_x:end_x] = np.maximum(
                            cv.resize(pred_tumor_p,(end_x-start_x,end_y-start_y)),
                            tumor_map[start_y:end_y,start_x:end_x]
                        )
                        nerve_map[start_y:end_y,start_x:end_x] = np.maximum(
                            cv.resize(pred_nerve_p,(end_x-start_x,end_y-start_y)),
                            nerve_map[start_y:end_y,start_x:end_x]
                        ) #######  Output Probability Map 
        self.tumor_map=tumor_map
        self.nerve_map=nerve_map
        self.prt=prt
        self.gt =cv.addWeighted(prt,0.75,overlay,0.25,0)
        return tumor_map,nerve_map,prt
    # -------------------------------
    # Region level의 Prediction
    # -------------------------------
    def predict_regions(self,kernel_size = 4,remove_kernel_size=4, dilate_iter = 4, prob_thr=0.5,method='prob',show=False):
        #tumor_map, nerve_map, prt = self.predict_map(slide_path,xml_path,level=level,overlap=overlap,patch_size=patch_size)
        
        # Prob Map to Mask
        tumor_mask = self.post_process(self.tumor_map,method=method,prob=prob_thr,kernel_size=remove_kernel_size)
        nerve_mask = self.post_process(self.nerve_map,method=method,prob=prob_thr,kernel_size=remove_kernel_size)
        
        nerve_mask = nerve_mask.astype(np.uint8)
        tumor_mask = tumor_mask.astype(np.uint8)
        ret_mask = np.zeros_like(nerve_mask)
        
        prt = self.prt.copy()
        overlay = prt.copy()
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
            is_include = np.all(np.logical_and(single_obj_dilated,tumor_mask)==single_obj_dilated)
            is_inter = np.any(np.logical_and(single_obj_dilated,tumor_mask))
            if is_inter and not is_include:
                ret_mask = cv.fillPoly(ret_mask,[c],1,1)
                #prt = cv.drawContours(prt,[c],-1,(0,255,255),1)
                overlay = cv.fillPoly(overlay,[c],(0,0,255),1)
                #prt = cv.addWeighted(prt, 0.8, overlay, 0.2, 0)
            else:
                #prt = cv.drawContours(prt,[c],-1,(255,0,255),1)
                overlay = cv.fillPoly(overlay,[c],(255,127,0),1)
                #prt = cv.addWeighted(prt, 0.8, overlay, 0.2, 0)
            
        tumor_cnts = cv.findContours(tumor_mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[0]
        for c in tumor_cnts:
            #prt = cv.drawContours(prt,[c],-1,(255,255,0),1)
            overlay = cv.fillPoly(overlay,[c],(255,0,0),1)
            #prt = cv.addWeighted(prt,0.8,overlay,0.2,0)
        
        if show==True:
            plt.figure(figsize = (14,10)); plt.imshow(prt)
            plt.legend(handles=[pni_legend,nerve_legend,tumor_legend], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        prt = cv.addWeighted(prt,0.75,overlay,0.25,0)
        return ret_mask, prt
        
    # -------------------------------
    # figure Save
    # -------------------------------
    def save_fig(self,gt_box,pr_box,savepath='pr.png',figrate=1):
        # Legend
        nerve_c   = mpatches.Patch(edgecolor = (1,0.5,0,1),facecolor= (1,0.5,0,1), label = 'Nerve')
        tumor_c = mpatches.Patch(edgecolor = (1,0,0,1),facecolor= (1,0,0,1), label = 'Tumor')
        pni_c = mpatches.Patch(edgecolor = (0,0,1,1),facecolor= (0,0,1,1), label = 'PNI')
        legends = [pni_c,nerve_c,tumor_c]
        # GT
        plt.figure(figsize = (16,int(8*figrate)))
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams['savefig.facecolor'] = "None"
        plt.subplot(1,2,1)
        plt.title('Ground Truth',fontsize=18)
        plt.axis('off')
        plt.legend(handles=legends, loc='lower right', borderaxespad=1., fontsize=12 ,edgecolor='inherit')
        plt.imshow(gt_box)

        plt.subplot(1,2,2)
        plt.title('Predicted',fontsize=18)
        plt.axis('off')
        plt.legend(handles=legends, loc='lower right', borderaxespad=1., fontsize=12 ,edgecolor='inherit')
        plt.imshow(pr_box)
        com_box = cv.hconcat([gt_box,np.zeros_like(gt_box[:,:2,:]),pr_box])
        plt.imsave(savepath,com_box)
        #cv.imwrite(savepath,com_box)
        #plt.savefig(savepath,transparent=True,dpi=1000)
    # -------------------------------
    # Region Level Evaluation
    # -------------------------------
    def eval_regions(self,slide_path,xml_path,save_dir,kernel_size = 12,remove_kernel_size=12, dilate_iter = 5, level = 5, overlap = 0,prob_thr=0.5,method='otsu', patch_size = 512,save = False):
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
        slide_name = slide_path.split('/')[-1].replace('.tiff','').replace('.svs','')
        #patient_path = '/'.join(slide_path.split('/')[:-1])
        #save_dpath = os.path.join(patient_path,save_dir)
        org_path = '/'.join(slide_path.split('/')[:-2])
        inf_path = os.path.join(org_path,save_dir)
        
        bbox_dicts = slide.get_annotation_dict()['bboxes']
        
        pred_pni_mask,pred_pni_img = self.predict_regions(
            kernel_size = kernel_size,
            remove_kernel_size=remove_kernel_size,
            dilate_iter = dilate_iter, 
            method=method,
            prob_thr=prob_thr
        )
        
        for key in bbox_dicts:
            if 'test' not in key:
                continue
            for i,bbox in enumerate(bbox_dicts[key]):
                min_x,min_y = np.min(bbox,axis = 0) 
                max_x,max_y = np.max(bbox,axis = 0)
                pred_pni_bbox = pred_pni_mask[min_y-50:max_y+50,min_x-50:max_x+50]
                gt_pni_imgbox = self.gt[min_y-50:max_y+50,min_x-50:max_x+50]
                pred_pni_imgbox = pred_pni_img[min_y-50:max_y+50,min_x-50:max_x+50]
                wt = max_x-min_x; ht = max_y-min_y
                
                if key=='p3_test' and np.max(pred_pni_bbox) >0 :
                    save_dpath = os.path.join(inf_path,'tps'); os.makedirs(save_dpath,exist_ok=True)
                    tp_len = len(os.listdir(save_dpath))
                    tp +=1
                    save_fpath  = str(tp_len) + '.png'
                elif key=='p3_test' and np.max(pred_pni_bbox) ==0 :
                    save_dpath = os.path.join(inf_path,'fns'); os.makedirs(save_dpath,exist_ok=True)
                    fn_len = len(os.listdir(save_dpath))
                    fn +=1
                    save_fpath  = str(fn_len) + '.png'
                elif key!='p3_test' and np.max(pred_pni_bbox)>0 :
                    save_dpath = os.path.join(inf_path,'fps'); os.makedirs(save_dpath,exist_ok=True)
                    fp_len = len(os.listdir(save_dpath))
                    fp +=1
                    save_fpath  = str(fp_len) + '.png'
                else:
                    save_dpath = os.path.join(inf_path,'tns'); os.makedirs(save_dpath,exist_ok=True)
                    tn_len = len(os.listdir(save_dpath))
                    tn +=1
                    save_fpath  = str(tn_len) + '.png'
                if save==True:
                    #os.makedirs(save_dpath,exist_ok=True)
                    #cv.imwrite(os.path.join(save_dpath,save_fpath),pred_pni_imgbox)
                    self.save_fig(gt_pni_imgbox,pred_pni_imgbox,os.path.join(save_dpath,save_fpath),figrate=(ht/wt))
        return (tp,fp,fn,tn)
            
