1.patch_extraction.py 
- 1) config python script cfg.py에서 추출하고자 하는 parameter를 설정합니다.
배율, 패치사이즈, 오버랩, level을 바꾸어주고 패치를 저장할 directory 이름을 PATCH_NAME에 설정합니다. 패치는 환자 밑에 생성됩니다. 
- 2) Docker container run
- 3) python 1.patch_extraction.py

2. train.py
- 1) cfg.py에서 두가지 config section을 받아서 train
-- (datagen config) 
```
CLASS_NAME : binary의 경우 segmentation할 target의 class를 설정
BATCH_SIZE : datagen이 yield하는 batch_size 설정
NUM_CLASSES : binary의 경우 1 
```
-- (train config)
```
목적에 맞는 train config
```
- 2) train.py는 config에서 받는 config값과 별개로 argument를 받아서 script를 실행합니다. argument에는 model의 type과 backbone , 그리고 finetuning을 실행하므로 pretrained_weight를 받습니다.

3. train.sh
동일한 config안에 parameter만 바꾸어서 model을 train합니다.
pairs에 model_type, backbone, pretrained_weight를 쌍으로 넣어서 2.train.py를 호출하여 실행합니다.



