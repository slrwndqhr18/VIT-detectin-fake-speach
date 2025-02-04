from  torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import numpy as np

def _calc_img_norm(_dataset):
    # dataset의 axis=1, 2에 대한 평균 산출
    mean_ = np.array([np.mean(x.numpy(), axis=(1, 2)) for x, _ in _dataset])
    # r, g, b 채널에 대한 각각의 평균 산출
    mean_r = mean_[:, 0].mean()
    mean_g = mean_[:, 1].mean()
    mean_b = mean_[:, 2].mean()

    # dataset의 axis=1, 2에 대한 표준편차 산출
    std_ = np.array([np.std(x.numpy(), axis=(1, 2)) for x, _ in _dataset])
    # r, g, b 채널에 대한 각각의 표준편차 산출
    std_r = std_[:, 0].mean()
    std_g = std_[:, 1].mean()
    std_b = std_[:, 2].mean()
    
    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)

def _calculate_dataset_stat(_imgDirPath):
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    rawDataset = ImageFolder(_imgDirPath, transform=transform)
    print("\tㄴimageFolder class to IDX:",rawDataset.class_to_idx)
    sizeOfDSet = rawDataset.__len__()
    print("\tㄴTotal image num:",sizeOfDSet)
    mean, std = _calc_img_norm(rawDataset)
    print(f'평균(R,G,B): {mean}\n표준편차(R,G,B): {std}')

def Make_dataloader(_opts):
    #loader output형태는 image, label = loader
    # Image Transform 정의
    transform = transforms.Compose([
        transforms.CenterCrop(224), #사용시 mean=(0.33640617, 0.124915846, 0.25378224), std=(0.35208786, 0.16510043, 0.2284499)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.33806258, 0.12576567, 0.2540155], std=[0.35299265, 0.16608432, 0.2285928])#(mean=[0.567, 0.210, 0.427], std=[0.273, 0.166, 0.110])#(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    rawDataset = ImageFolder(_opts["outputPath"], transform=transform)
    print("\tㄴimageFolder class to IDX:",rawDataset.class_to_idx)
    sizeOfDSet = rawDataset.__len__()
    print("\tㄴTotal image num:",sizeOfDSet)
    trainDataset, testDataset = random_split(rawDataset, [52438,3000])

    #Validate data set ====================
    #_validate_dataset_items(trainDataset)
    #_validate_dataset_items(testDataset)
    # =====================================

    TrainSet = DataLoader(
        trainDataset,
        batch_size=_opts["batch_size"], #수정필요
        num_workers=_opts["loader_workers"],
        shuffle=False,
        drop_last=True
        )
    TestSet = DataLoader(
        testDataset,
        batch_size=1,#_opts["batch_size"]
        num_workers=_opts["loader_workers"],
        shuffle=False,
        drop_last=True
        )
    return TrainSet, TestSet

if __name__ == "__main__":
    print("이미지 데이터셋 통계수치 계산")
    from handleConfig import Get_config
    paramSet = Get_config()["preprocess"]
    _calculate_dataset_stat(paramSet["outputPath"])