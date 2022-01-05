## Table of Contents <!-- omit in toc -->

- [Overview](#overview)
- [Data](#data)
- [Modeling](#modeling)
- [Development](#development)
- [References](#references)
## Overview
<b>Perineural Invasion</b>(pni) 
 means that cancer cells were seen surrounding along a nerve within tissue. When this is found, it means that there is a higher chance that the cancer has spread outside tissue. goal of research is to detect pni in colon cancer with DL.

## Data
<details>
 <summary > <b>Slide </b></summary> 
 All Slide is Colon Slide scanned by 40x Aperio from International Seongmo Hosp. and  labeled Polygonal Area with Bounding Box of Interest by Pathologist with ASAP which is open source to annotate large scale image.
</details>
<details>
 <summary><b>Patch</b></summary> 
 We used 10x (mpp : 1.0) and (512,512,3) sized patch image from slide by bounding box sliding.
</details>

## Modeling
<details>
 <summary><b>Pipeline</b></summary> 
 Experiments are proceeded as follows.<br>
Data Definition => Patch Extraction => Build DataGenerator with Patch acoording to Experiments Condition (e.g. Binary, Multiple ... ) => Train Segmentation Model => Patch Level Evaluation => Region Level Evaluation
</details>

## Development
<details>
 <summary><b>Code Explnation</b></summary> 
  <div markdown="1">

  - 1.patch_extraction.py <br>
    Extraction Patch from Slide with Target mpp (or Target Magnificiation) & Target Size.
    ``` 
    python 1.patch_extraction.py 
    ```

  - 2.train&#46;py<br>
    Train Segmentation Model with Given Parameters (train class, train model, model backbone, pretrained weight, loss function)
    ```
    python 2.train.py multi unet efficientnetb0 imagenet ce
    ```

  - 3.train&#46;sh<br>
    Bash Script for Various Train and Notification GPU Allocation to Slack
    ```
    chmod 755 3.train.sh
    ./3.train.sh
    ```

</div>
</details>
<details>
 <summary><b>Test Dashboard</summary> 
  <div markdown="1">
  <a href='http://pni.ssus.work'> Go Site </a>
  </div>
</details>

## References



