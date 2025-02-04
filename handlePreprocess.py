from torch.utils.data import Dataset
import polars as pl
import os
from handleCQT import Conv_to_CQT_img
from handleMultiPs import Exec_in_parallel
from handleConfig import Get_config

class Preprocess:
    def __init__(self , _rawdPath, _metadPath, _outputPath,
                 _labelRealPath, _labelfakePath, _threadCnt = 10):
        
        self.__metadataPath = _metadPath
        self.__OUTPUT_DATA_DIR = _outputPath
        self.__resDataPath = self.__OUTPUT_DATA_DIR  +"img/"
        self.__srcDataPath = _rawdPath
        self.__labelRealPath = _labelRealPath
        self.__labelfakePath = _labelfakePath
        self._METADATA = None
        self._PS_CNT = _threadCnt
        os.makedirs(self.__resDataPath, exist_ok=True)
    
    #csv파일 전처리
    def Make_metadata(self):
        df = pl.read_csv(self.__metadataPath)
        self._METADATA = df
        print("\tcsv파일 전처리 - 총 {}행".format(self._METADATA.shape[0]))
    
    def __make_cqt_images(self, _inputList, _imgPath):
        threadCnt = self._PS_CNT
        jobStep = self._METADATA.shape[0] // threadCnt
        if self._METADATA.shape[0] % threadCnt != 0:
            threadCnt += 1
        temp = 0
        jobArgs = []
        for i in range(0, len(_inputList), jobStep):
            if i + jobStep < len(_inputList):
                subList = _inputList[i:i + jobStep]
            else:
                subList = _inputList[i:]
            temp += len(subList)
            jobArgs.append({
                "i": i,
                "ids": subList,
                "inputPath": self.__srcDataPath,
                "outputPath": _imgPath
            })
        _ = Exec_in_parallel(Conv_to_CQT_img, jobArgs, threadCnt)


    def Make_datasets(self):
        realAudioList = self._METADATA.filter(pl.col("label")==pl.lit("real")).select('id').to_series().to_list()
        fakeAudioList = self._METADATA.filter(pl.col("label")==pl.lit("fake")).select('id').to_series().to_list()
        print("\tCQT 변환 대상 수: Total {}, Real {}, Fake {}".format( self._METADATA.shape[0], len(realAudioList), len(fakeAudioList)))
        print("\t\tㄴconverting label==real image...")
        self.__make_cqt_images(realAudioList, self.__labelRealPath)
        print("\t\tㄴconverting label==fake image...")
        self.__make_cqt_images(fakeAudioList, self.__labelfakePath)
     
        return
    def Run(self):
        print("Step[Run preprocess]")
        self.Make_metadata()
        self.Make_datasets()
        print("\tPreprocess end")

def Run_preprocess():
    config = Get_config()
    paramSets = config["preprocess"]
    Preprocess(
        paramSets["inputPath"],
        paramSets["metadPath"],
        paramSets["outputPath"],
        paramSets["labelRealPath"],
        paramSets["labelFakePath"]
    ).Run()
