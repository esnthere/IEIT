# IEIT: Blind Image Quality Assessment for Authentic Distortions by Intermediary Enhancement and Iterative Training
This is the source code for [IEIT: Blind Image Quality Assessment for Authentic Distortions by Intermediary Enhancement and Iterative Training](https://ieeexplore.ieee.org/document/9786803).![IEIT Framework](https://github.com/esnthere/IEIT/blob/main/framework.png)

## Dependencies and Installation
Pytorch: 1.8.1  
timm: 0.3.2  
CUDA: 10.2  

## For test:
### 1. Data Prepareation  
   To ensure high speed, save images and lables of each dataset with 'mat/npz' files. Only need to run '**data_preparation_example_for_koniq.py**' once for each dataset. Enhanced images and files for slection probablility can be download from [Enhanced images](https://pan.baidu.com/s/1vSeiH61x5TD5VIRn8NKOVw?pwd=wz98)
   
### 2. Load pre-trained weight for test  
   The models pre-trained on KonIQ-10k with 5%, 10%, 25%, 80% samples are released. The dataset are randomly splitted several times during training, and each released model is obtained from the first split (numpy. random. seed(1)). The model file '**my_vision_transformer.py**' is modified from open accessed source code of [DEIT](https://github.com/facebookresearch/deit) and [TIMM](https://github.com/huggingface/pytorch-image-models/tree/main/timm). 
   
   The pre-trained models can be downloaded from: [Pre-trained models](https://pan.baidu.com/s/1IoGWFXKSi-ljYaWB6mOeaw?pwd=f4wv ). Please download these files and put them in the same folder of code and then run '**test_example_koniq_*n*percent.py**' to make intra/cross dataset test for models trained on *n%* samples.
   
   
## For train:  
The training code can be available at the 'training' folder.


## If you like this work, please cite:

{   
      author={Song, Tianshu and Li, Leida and Chen, Pengfei and Liu, Hantao and Qian, Jiansheng},
      journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
      title={Blind Image Quality Assessment for Authentic Distortions by Intermediary Enhancement and Iterative Training}, 
      year={2022},
      volume={32},
      number={11},
      pages={7592-7604},
      doi={10.1109/TCSVT.2022.3179744}
  }
  
## License
This repository is released under the Apache 2.0 license.  

