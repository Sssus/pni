#!/bin/bash

slack_token=$(</sec/slacktoken)

curl --location --request POST "$slack_token" \
--header 'Content-Type: application/json' \
--data-raw '{"text":"V100 Allocated & Train Started"}'

export pairs="
multi,deeplab,xception,pascal_voc,focal
multi,unet,efficientnetb0,imagenet,focal
tumor,deeplab,xception,pascal_voc,ce
tumor,unet,efficientnetb0,imagenet,ce
nerve,deeplab,xception,pascal_voc,ce
nerve,unet,efficientnetb0,imagenet,ce
"

for i in $pairs;
do 
    IFS=',' read item1 item2 item3 item4 item5 <<< "${i}"
        #echo "${item1}" and "${item2}" and "${item3}"
	python 2.train.py ${item1} ${item2} ${item3} ${item4} ${item5} 
done

slack_token=$(</sec/slacktoken)

curl --location --request POST "$slack_token" \
--header 'Content-Type: application/json' \
--data-raw '{"text":"V100 Deallocated & Train Ended"}'
