from math import sqrt
from torch import nn, eye
from safetensors import safe_open, torch
from timm.models.vision_transformer import VisionTransformer as timm_ViT
#https://github.com/JamesQFreeman/LoRA-ViT/blob/main/lora.py#L289
class _qkv(nn.Module):
    def __init__(
            self,
            _qkv,
            _linear_a_q,
            _linear_b_q,
            _linear_a_v,
            _linear_b_v,
            _rank,
            _alpha
    ):
        super().__init__()
        self.qkv = _qkv
        self.linear_a_q = _linear_a_q
        self.linear_b_q = _linear_b_q
        self.linear_a_v = _linear_a_v
        self.linear_b_v = _linear_b_v
        self.dim = _qkv.in_features
        self.w_identity = eye(_qkv.in_features)
        self.rank = _rank
        self.alpha = _alpha

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.dropout(x)
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += (self.alpha // self.rank) * new_q
        qkv[:, :, -self.dim :] += (self.alpha // self.rank) * new_v
        return qkv

class LoRA(nn.Module):
    def __init__(self, _model, _rank=16, _alpha=8, _n_labels=2):
        super().__init__()
        self.__layer_lora = list(range(len(_model.blocks)))
        self.__baseModel:timm_ViT = _model

        self.__weight_a = []
        self.__weight_b = []
        # The original params are frozen, and only LoRA params are trainable.
        self.__freeze_base_model_params()
        self.__unfreeze_lora_params(_rank, _alpha)
        self.__reset_params()

        self.proj_3d = nn.Linear(_n_labels * 30, _n_labels) #num_classes == 2
        self.__baseModel.reset_classifier(num_classes=_n_labels)

    def __freeze_base_model_params(self):
        for param in self.__baseModel.parameters():
            param.requires_grad = False

    def __unfreeze_lora_params(self, _rank, _alpha):
        for layer, blk in enumerate(self.__baseModel.blocks):
            if layer not in self.__layer_lora:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, _rank, bias=False)
            w_b_linear_q = nn.Linear(_rank, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, _rank, bias=False)
            w_b_linear_v = nn.Linear(_rank, self.dim, bias=False)
            self.__weight_a.append(w_a_linear_q)
            self.__weight_a.append(w_a_linear_v)
            self.__weight_b.append(w_b_linear_q)
            self.__weight_b.append(w_b_linear_v)
            blk.attn.qkv = _qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                _rank,
                _alpha
            )
    
    def __reset_params(self):
        for wA in self.__weight_a:
            nn.init.kaiming_uniform_(wA.weight, a=sqrt(5))
        for wB in self.__weight_b:
            nn.init.kaiming_uniform_(wB.weight)
    
    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        
        save both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        
        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        torch.save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\
            
        load both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = nn.parameter.Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = nn.parameter.Parameter(saved_tensor)
                
            _in = self.lora_vit.head.in_features
            _out = self.lora_vit.head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = nn.parameter.Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def forward(self, x):
        return self.__baseModel(x)