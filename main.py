#from handlePreprocess import Run_preprocess
from handleConfig import Get_config
from handleDataset import Make_dataloader
from model.Executor import RunVIT
import os
#import warnings
#warnings.filterwarnings("ignore")

def __process_model_test(_model, _dataLoader):
    _model.Test_model(_dataLoader)

def __process_model_train(_model, _dataLoader):
    _model.Train_model(_dataLoader)
    os._exit(0)

if __name__ == "__main__":
    print("start")
    paramSet = Get_config()
    __PROCESS_MODE = paramSet["executeMode"]
    print("Process Mode:", __PROCESS_MODE)
    # 데이터셋 생성 부분 (실행할 필요 XX) ======================
    #dataSets = Run_preprocess()
    # ===================================================
    dataLoader = Make_dataloader(paramSet["datasetOpts"])
    model = RunVIT(paramSet["hyperParams"])
    model.Get_param_nums()
    if __PROCESS_MODE == "train":
        __process_model_train(model, dataLoader[0])
    else:
        __process_model_test(model, dataLoader[1])

