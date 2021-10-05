#!/bin/bash

#fpn,resnet50,imagenet
#fpn,efficientnetb3,imagenet
#unet,resnet50,imagenet
#unet,vgg16,imagenet,
#unet,inceptionv3,imagenet
#unet,inceptionresnetv2,imagenet
#unet,seresnext101,imagenet
#nerve,deeplab,xception,pascal_voc
#tumor,deeplab,xception,pascal_voc
#multi,deeplab,xception,pascal_voc
export pairs="
nerve,unet,efficientnetb0,imagenet
tumor,unet,efficientnetb0,imagenet
multi,unet,efficientnetb0,imagenet

"

for i in $pairs;
do 
    IFS=',' read item1 item2 item3 item4 <<< "${i}"
	python 4.train_kfold.py ${item1} ${item2} ${item3} ${item4}
done
