from pred import *
from cfg import *
from util import *
import multiprocessing, glob

predictor = predictor(model_cfg)
def inference_slide(svs_path,save_dir,overlap = 0,tissue_ratio=0.7,patch_size = 512, mixed=False,in_train=False,save=True,show=False):
    ## config set
    xml_path = svs_path[:-3]+'xml' if in_train==True else ''
    init_params.update({'svs_path':svs_path,'xml_path':xml_path})
    filename = svs_path.split('/')[-1][:-4]
    save_path = save_dir + filename + '.tif'
    
    ##
    slide = processor(init_params)
    tissue_mask = slide.get_tissue_mask()
    mask = np.zeros((slide.dest_h,slide.dest_w))
    counter = np.zeros_like(mask)
    tissue_mask = cv.morphologyEx(tissue_mask,cv.MORPH_OPEN,cv.getStructuringElement(cv.MORPH_RECT,(5,5)),iterations=1)
    
    if in_train==True :
        anno_mask = slide.get_anno_mask()
    
    ##
    min_y , min_x = slide.arr.shape[:2]
    max_x, max_y = (0,0)
    coord = np.where(tissue_mask>0)
    if np.min(coord[1])<min_x: min_x = np.min(coord[1])
    if np.max(coord[1])>max_x: max_x = np.max(coord[1])
    if np.min(coord[0])<min_y: min_y = np.min(coord[0])
    if np.max(coord[0])>max_y: max_y = np.max(coord[0])
        
    ##
    slide_w = max_x-min_x; slide_h = max_y-min_y; step = 1-overlap; multiple = slide.src_h//slide.dest_h
    patch_size = 512; patch_size_lv0 = 2*patch_size;  patch_size_lv2 = patch_size_lv0//multiple
    print(f'MIN_X : {min_x} MIN_Y : {min_y} MAX_X : {max_x} MAX_Y : {max_y}')
    print(f'PATCH_0_SIZE : {patch_size_lv0} PATCH_2_SIZE : {patch_size_lv2} ')
    s = int(patch_size_lv0*step)
    y_seq,x_seq = slide.get_seq_range(slide_w,slide_h,multiple,patch_size_lv0,s)
    
    ##
    for y in y_seq:
        for x in x_seq:
            start_x = int(min_x+(s*x/multiple)); end_x = int(min_x+((s*(x+int(1/step)))/multiple))
            start_y = int(min_y+(s*y/multiple)); end_y = int(min_y+((s*(y+int(1/step)))/multiple))        
            img_patch = np.array(slide.slide.read_region(
                location=(int((min_x*multiple)+(s*x)),
                          int((min_y*multiple)+(s*y))),
                level=0, size = (patch_size_lv0,patch_size_lv0)   
            )).astype(np.uint8)[...,:3]
            img_patch_resized = cv.resize(img_patch,(patch_size,patch_size),cv.INTER_CUBIC)
            img_patch_rgb = cv.cvtColor(img_patch_resized,cv.COLOR_BGR2RGB)
            
            ##
            if in_train==True: perineural = anno_mask['p_2'][start_y:end_y,start_x:end_x]
            
            if slide.get_ratio_mask(tissue_mask[start_y:end_y,start_x:end_x])>tissue_ratio:
                pred_mask = predictor.pred_patch(img_patch_rgb,end_y-start_y,end_x-start_x,mixed=mixed)
                if show==True:
                    if in_train==True:
                        if  (np.max(perineural)>0) or (np.max(pred_mask)>0):
                            plt.figure()
                            plt.subplot(1,3,1);plt.title('image'); plt.imshow(img_patch_rgb)
                            plt.subplot(1,3,2);plt.title('answer'); plt.imshow(perineural)
                            plt.subplot(1,3,3);plt.title('predict'); plt.imshow(pred_mask)
                    else:
                        plt.figure()
                        plt.subplot(1,2,1);plt.title('image'); plt.imshow(img_patch_rgb)
                        plt.subplot(1,2,2);plt.title('predict'); plt.imshow(pred_mask)
            else:
                pred_mask = np.zeros((end_y-start_y,end_x-start_x))
            
            try:
                #mask[start_y:end_y,start_x:end_x] = cv.resize(pred_mask,(end_x-start_x,end_y-start_y),cv.INTER_CUBIC)
                mask[start_y:end_y,start_x:end_x] = pred_mask
            except:
                mask[start_y:end_y,start_x:end_x] = 0
    if save==True:
        cv.imwrite(save_path,mask.astype(np.uint8))
    
