import torch 
from torch.amp import autocast ## 2024-12-12 deprecated warning fixed (fixed torch.cuda.amp to torch.amp)
import time
from datetime import datetime
from torch import nn, optim, autograd
import os
import numpy as np
import torch.amp
from model.VIT_timm import Create_VIT
from model.LoRA import LoRA
from model.matrics import metric
#https://huggingface.co/docs/peft/main/en/task_guides/image_classification_lora
class AIExecutor:
    def __init__(self,_name, _options, _model, _critition):
        print("Step[Init Model] Name:", _name)
        self.__saveParamPath = "./SV_PARAM"
        self.__saveLoadPath = _options["savedLoadPath"]
        self.__modelName =_name
        
        self.batchSize = _options["batch_size"]
        self.epochSize = _options["train_epochs"]

        self.schedularStep = _options["schedular_step"]
        self.schedularGamma = _options["schedular_gamma"]
        

        self.__isEarlyEscape = _options["isEarlyEscape"]
        self.__usePreTrainedParams = _options["usePreTrainedParams"]
        self._exeMode = _options["executeMode"]
        self.__stepPrintProgress = _options["stepPrintProgress"]
        if torch.cuda.is_available():
            #cuda:0
            self.Device = torch.device("cuda")
            print("Executing model - Using GPU")
        else:
            self.Device = torch.device("cpu")
            print("Executing model - GPU is not available -> Using CPU")    
        self.Model = _model.to(self.Device)
        self.Optimizer = optim.AdamW(self.Model.parameters(), lr= _options["learning_rate"])#, weight_decay=0.1)
        self.Critition = _critition

    def Get_param_nums(self):
        print("\tㄴPrint param nums:")
        temp = 0
        for parameter in self.Model.parameters():
            if len(parameter.shape) == 2:
                temp+=parameter.shape[0] * parameter.shape[1]
            else:
                temp+=parameter.shape[0]
        print("\tㄴ",temp)

    def __save_params(self, _addTxt=""):
        now = datetime.now()
        now = now.strftime('date_%Y-%m-%d__time_%H-%M-%S') ## Filesave on Windows fixed
        if len(_addTxt) > 1:
            now += "_" + _addTxt
        torch.save(self.Model,"{}/{}_{}.pt".format(self.__saveParamPath,self.__modelName, now))
        torch.save(self.Model.state_dict(),"{}/{}_state_{}.pt".format(self.__saveParamPath,self.__modelName, now))
        print("\tㄴModel saved", now)
    
    def _run_model(self, _input, _target):
        _input = _input.to(self.Device)
        _target = _target.to(self.Device)
        return self.Model(_input), _target

    def __load_trained_params(self):
        if self.__usePreTrainedParams:
            loadFilePath = self.__saveLoadPath
            self.Model.load_state_dict(torch.load(loadFilePath, map_location=self.Device))
            print("\tㄴload model -",loadFilePath)


    def Train_model(self, _dataLoader):
        creterion = self.Critition().to(self.Device)
        print("Step[Train model]")
        start = time.time()
        scheduler = optim.lr_scheduler.StepLR(self.Optimizer, step_size=self.schedularStep, gamma=self.schedularGamma)
        #ampScaler = torch.amp.GradScaler("cuda") #2024-12-11 AMP scaler added to fix malfunction ## 2024-12-12 deprecated warning fixed
        self.__load_trained_params()
        autograd.set_detect_anomaly(True)
        for e in range(self.epochSize): #Epoch
            print("\tㄴ progress - Epoch: {}/{}".format(e, self.epochSize))
            self.Model.train()
            with torch.set_grad_enabled(True):
                i = 0
                trainLoss = []
                for inputs, targets in _dataLoader:
                    self.Optimizer.zero_grad()
                    inputs, targets = inputs.to(self.Device), targets.to(self.Device) ## 2024-12-04 added code for amp, send inputs and labels to GPU
                    with autocast("cpu"): ## 2024-12-04 added code for amp ## 2024-12-12 deprecated warning fixed
                        try:
                            outputs, targets = self._run_model(inputs, targets)
                        except Exception as e:
                            print(e)
                            os._exit(0)
                        loss = creterion(outputs, targets)
                    if i % self.__stepPrintProgress == 0:
                        print("\t\t\tㄴ{0}th Loss: {1:.7f}".format(i, loss.item()))
                    if np.isnan(loss.item()) or torch.isnan(loss): ## 2024-12-12 patch to pass if NaN happens
                        #print("\t\t\tㄴBreak!!! {0}th Loss is NaN {1}".format(i, loss.item()))
                        #break
                        print("\t\t\tㄴNaN value occured!!! Passing current process... {0}th Loss is NaN {1}".format(i, loss.item()))
                        continue
                    trainLoss.append(loss.item())
                    #2024-12-04 added code for amp (3 lines)
                    #ampScaler.scale(loss).backward()
                    #ampScaler.step(self.Optimizer)
                    #ampScaler.update()
                    #loss.backward()
                    #self.Optimizer.step()
                    
                    #Early break ========================
                    # if self.__isEarlyEscape and i == 45001:
                    #    break
                    i+=1
                print("\t\tEpoch {0}th Loss avg: {1:.7f}".format(e, np.average(trainLoss)))
                if e % 18 == 0: 
                    self.__save_params("epoch_{}".format(e))
                scheduler.step()
            #break ## For testing single epoch
            print("\tㄴ Epoch {}th End Time: {:.5f}(sec)".format(e, time.time() - start))
        autograd.set_detect_anomaly(False)
        end = time.time()
        print(f"\tㄴfinished: {end - start:.5f} sec")
        self.__save_params("final")
    
    def Test_model(self, _dataLoader):
        print("Step[Test model]")
        self.__load_trained_params()

        preds = []
        trues = []
        self.Model.eval()
        i = 0
        countCorrOut = 0
        with torch.no_grad():
            for inputs, targets in _dataLoader:
                outputs, targets = self._run_model(inputs, targets)
                #outputs = outputs.detach().cpu().numpy()
                #targets = targets.detach().cpu().numpy()
                # print(outputs.flatten().flatten().shape, targets.flatten().flatten().shape)
                # return
                preds.append(outputs.cpu().numpy())
                trues.append(targets.cpu().numpy())
                if outputs.item() == targets.item():
                    countCorrOut += 1
                if i % self.__stepPrintProgress == 0:
                    print("\t\t\tㄴ{0}th".format(i))
                i+=1

        preds = np.array(preds) # 2024-12-11 TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        #print(preds.flatten().shape, trues.flatten().shape)
        #print(preds[0][0], trues[0][0])
        #return preds, trues
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        #print('\tㄴTest result: mse={1:.7f}, mae={1:.7f}, rse={1:.7f}'.format(mse, mae, rse))
        print("Correctness: {}%".format(countCorrOut/i*100))
        return preds, trues

class RunVIT(AIExecutor):
    def __init__(self, _hyperparams): 
        #model = VIT(2) #lora 적용 전 vit
        model = LoRA(Create_VIT(), 128, 64, 2) #alpha = r * 4
        super().__init__("VIT", _hyperparams, model, _critition = nn.CrossEntropyLoss)
        self.predLen = _hyperparams["pred_len"]
        self.targetLen = _hyperparams["target_len"]#nn.MSELoss()
    
    def _run_model(self, _img, _label:torch.Tensor):
        _img = _img.to(self.Device)
        _label = _label.to(self.Device)

        output = self.Model(_img)
        if self._exeMode != "train":
            _, output = torch.max(output, 1)

        return output, _label #.floats