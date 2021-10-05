#!/bin/bash

#fpn,resnet50,imagenet
#fpn,efficientnetb3,imagenet
#unet,resnet50,imagenet
#unet,vgg16,imagenet,
#unet,inceptionv3,imagenet
#unet,inceptionresnetv2,imagenet
#unet,seresnext101,imagenet
export pairs="
deeplab,xception,pascal_voc
unet,efficientnetb0,imagenet
"

for i in $pairs;
do 
    IFS=',' read item1 item2 item3 <<< "${i}"
        #echo "${item1}" and "${item2}" and "${item3}"
	python 2.train.py ${item1} ${item2} ${item3}
done
