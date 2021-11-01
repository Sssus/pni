#!/bin/bash


curl --location --request POST 'https://hooks.slack.com/services/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' \
--header 'Content-Type: application/json' \
--data-raw '{"text":"NIA GPU Allocated & Train Started"}'
#nerve,unet,efficientnetb0,imagenet,focal
#nerve,unet,efficientnetb0,imagenet,ce
#nerve,deeplab,xception,pascal_voc,focal
#nerve,deeplab,xception,pascal_voc,ce
#tumor,unet,efficientnetb0,imagenet,focal
#tumor,unet,efficientnetb0,imagenet,ce
#tumor,deeplab,xception,pascal_voc,focal
#tumor,deeplab,xception,pascal_voc,ce


export pairs="
multi,unet,efficientnetb0,imagenet,focal
multi,unet,efficientnetb0,imagenet,ce
multi,unet,efficientnetb0,imagenet,dice
multi,deeplab,xception,pascal_voc,focal
multi,deeplab,xception,pascal_voc,ce
multi,deeplab,xception,pascal_voc,dice
"

for i in $pairs;
do 
    IFS=',' read item1 item2 item3 item4 item5 <<< "${i}"
        #echo "${item1}" and "${item2}" and "${item3}"
	python 2.train.py ${item1} ${item2} ${item3} ${item4} ${item5}
done

curl --location --request POST 'https://hooks.slack.com/services/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' \
--header 'Content-Type: application/json' \
--data-raw '{"text":"NIA GPU Deallocated & Train Ended"}'
