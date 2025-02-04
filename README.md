# Detecting deepfake audio using VIT


This project is about detecting deepfake audio using attention algorithem.


idea source 
paper: https://ieeexplore.ieee.org/document/10197715</br>
by: Guzin Ulutas


## Model architecture

<img width="335" alt="image" src="https://github.com/user-attachments/assets/aa6ebf36-4cac-4281-9cc4-35a60446a044" />


[Fig1. ML Architecture]</br>
In this project I designed simple but effective structure. The main logic is 3 things.
- Converting audio data to CQT image had conducted in parallel programing.
- VIT model is mixed with LoRA model.
- Training model process has amp code.


------------------------------
## About Code


I used VIT as base model and added LoRA for accuracy.</br>
Important thing in here is you need to convert audio data into the CQT image. this process will be done at preprocessing layer.

|File name|description|
|------|---|
| /model | Every codes related to ML model |
| /model/Component | Codes that used inside the ML model.</br>Its a component of ML model like attention alg or FN layer. |
| handleConfig.py | load CONFIG.yaml and setup every parameters |
| handleDataset.py | By using PyTorch Dataloader, revise and format preprocessed data into PyTorch data structure |
| handlePreprocess | Load raw dataset and execure preprocessing |
| makeGraph | Just making graph to see the result. Nothing important in here. |
| handleMultiPs | Making the CQT file using multiprocessing. |

- main.py is the entry point of process. Just run main.py
- /model/Executor.py is the entry point of ML model. It is  a class to load params and define model train / test / run process.
- For the reduction of preprocessing time, parallel programin is used in here.
- There is the code about amp, and this reduced training time alot.
