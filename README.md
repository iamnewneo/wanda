# WANDA: Wireless Anomaly Detection Algorithm

## General Info
This repository includes code for training a H-Score based Convolutional Neural Network (CNN) and an SVM to detect anomaly in Wifi signals.

## Dataset Used
Variation of http://www.crawdad.org/owl/interference/20190212/index.html

This implementation uses pytorch library and pytorch lightning trainer

## Requirements
Python3, Pytorch, Pytorch Lightning, Sklearn, Pyod

### Installation Commands
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch

conda install -c conda-forge pytorch-lightning

conda install pyod optuna scikit-learn-intelex -c conda-forge

conda install -c conda-forge scikit-learn=1.0.0
```

## Code Overview
WANDA has 8 main folders:
1. analyze:  This folder contains the code to analyze the prediction of each model. We plot things like AUC, ROC Curve from this folder.
2. data_loader: This folder contrains the code for pytorch data loaders to load data and feed it into the pytorch models.
3. infer: This folder has code for inferenec on the entire test and train set.
4. model: This folder has code for all the models that we use in this project, for e.g OC-SVM, HScore CNN, Isolation Forest etc.
5. preprocessing: This folder has code for preprocessing that is required to tranform the raw inputs to a form that is compatible with different models.
6. trainer: This folder has a script called trainer.py that we use to train our H-Score model.
7. utils: This folder has general util functions like resize, save and load models etc that we use in our project.
8. visualizer: This folder has the code to visualize the activation maps of our trained H-Score model

## Setup [For Training Models from Scratch]
To run this project install the requirements.
Execute `bash train.sh` in the root folder to start the training process
`train.sh` executes the `driver.py` file that is the central script to run all the training process. It brings together the entire framework and does all the steps like reading data, preprocessing, optimization, saving models, predictions etc, that are necessary to complete the entire pipeline of training models and saving these models after finding out the best hyperparametes. We use `optuna` to find the best hyperparameters, they were not manually tuned.

`driver.py` should be the first file you should look into if you want to understand the how the code flows under the hood.

## Setup [For using as API]
If we dont want to train the models from scratch and just want to use them for inference, there are two main files `api.py` and `main.py`. Both of them are very loosely coupled with `WANDA` and its dependence on the framework can be removed easily if we don't want to install all the packages and that are required for WANDA package.

`api.py` has the code for the `Predictor` class that loads all the models in memory that are required for inference on our input.

`main.py` has the code for `is_anomaly` function that should be used as the access point for inference. We can import this function into any other package and do the prediction on new images. It expects a single spectrogram as the input and returns a bool value signifying if the model found anomaly in the spectrogram or not. This function only takes a single image of spectrogram as the input, but it can be easily modified to receive batch input of spectrograms for batched inference.