# Training Code for IEIT: Blind Image Quality Assessment for Authentic Distortions by Intermediary Enhancement and Iterative Training
This is the training example of IEIT on the LIVEW dataset, which is small enough to re-train. The trainning process is the same for other datasets:

## 1. Data Prepareation

   To ensure high speed, save training images and lables, enhanced images, probability for image selection into 'mat/npz' files. The preparation process please refer to the published paper [KG-IQA](https://ieeexplore.ieee.org/document/10003665). The necessary 'mat/npz' files can be downloaded from [Trainng files](https://pan.baidu.com/s/1EerM_rvNVo8Eevw74p3TNQ?pwd=z3oh). Please download these files and put them into the same folder of the training code.
   
## 2. Training the model

   Please 'run training_example_of_rbid_25percent.ipynb' to train the model. The pre-trained weight and model file '**my_vision_transformer.py**' is modified from open accessed source code of [DEIT](https://github.com/facebookresearch/deit) and [TIMM](https://github.com/huggingface/pytorch-image-models/tree/main/timm). 

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



