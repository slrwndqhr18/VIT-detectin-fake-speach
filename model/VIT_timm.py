import timm
import torch.nn as nn

class VIT(nn.Module):
    def __init__(self, _n_labels):
        from peft import LoraConfig
        LoraConfig
        super().__init__()
        #self.__backbone = timm.models.vit_base_patch16_224(pretrained=True, num_classes=_n_labels)
        self.__backbone = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.__outputLayer = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.__backbone(x)
        x = self.__outputLayer(x)
        return x
    
def Create_VIT():
    return timm.create_model("vit_base_patch16_224", pretrained=True) 
