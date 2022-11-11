WANDA: Wifi Anomaly Detection Algorithm

General Info
This repository includes code for training a H-Score based Convolutional Neural Network (CNN) and an SVM to detect anomaly in Wifi signals.

Dataset Used: Variation of http://www.crawdad.org/owl/interference/20190212/index.html

This implementation uses pytorch library and pytorch lightning trainer

Requirements
Python3, Pytorch, Pytorch Lightning

Installation Commands
```
mamba install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch

mamba install -c conda-forge pytorch-lightning=1.6.0

mamba install pyod optuna scikit-learn-intelex -c conda-forge
```
Setup
To run this project install the requirements and make sure you have at least 14GB GPU:

`bash train.sh`