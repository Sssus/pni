#!/bin/bash

#fpn,resnet50,imagenet
#fpn,efficientnetb3,imagenet
#unet,resnet50,imagenet
#unet,vgg16,imagenet,
#unet,inceptionv3,imagenet
#unet,inceptionresnetv2,imagenet
#unet,seresnext101,imagenet
#nerve,unet,efficientnetb0,imagenet

#nerve,deeplab,xception,pascal_voc
#nerve,fpn,efficientnetb0,imagenet
#tumor,unet,efficientnetb0,imagenet
#tumor,fpn,efficientnetb0,imagenet
#tumor,deeplab,xception,pascal_voc
export pairs="
multi,unet,efficientnetb0,imagenet
multi,fpn,efficientnetb0,imagenet
multi,deeplab,xception,pascal_voc
"

for i in $pairs;
do 
    IFS=',' read item1 item2 item3 item4 <<< "${i}"
	python 4.train_kfold.py ${item1} ${item2} ${item3} ${item4}
done
