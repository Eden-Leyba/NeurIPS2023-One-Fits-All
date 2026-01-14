import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers import Qwen2Model, AutoModel, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, Qwen2Config
from einops import rearrange

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in 16-bit for speed
    bnb_4bit_use_double_quant=True,       # optimize memory further
    bnb_4bit_quant_type="nf4"             # "normalized float 4" (best for LLMs)
)

class QWEN4TS(nn.Module):
    
    def __init__(self, configs, device):
        super(QWEN4TS, self).__init__()
        self.is_gpt = configs.is_gpt            #true
        self.patch_size = configs.patch_size    #16
        self.pretrain = configs.pretrain        #true
        self.stride = configs.stride            #8           
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1 
        #(336-16)//8+1=41
        
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        if configs.is_gpt:
            if configs.pretrain:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    'Qwen/Qwen2.5-7B', 
                    trust_remote_code=True, 
                    device_map="auto", # Load on CPU first to avoid instant OOM
                    quantization_config=bnb_config,
                    dtype=torch.bfloat16
                ) 
            else:
                print("------------------no pretrain------------------")
                self.llm_model = AutoModel.from_config(AutoConfig.from_pretrained('Qwen/Qwen2.5-7B', trust_remote_code=True))

            self.d_model = self.llm_model.config.hidden_size
            print(f"Model Hidden Size: {self.d_model}")

            try:
                if hasattr(self.llm_model, 'layers'): 
                    print(f"Slicing to first {configs.gpt_layers} layers. 1")
                    self.llm_model.layers = self.llm_model.layers[:configs.gpt_layers]
                elif hasattr(self.llm_model.model, 'layers'): 
                    print(f"Slicing to first {configs.gpt_layers} layers. 2")
                    self.llm_model.model.layers = self.llm_model.model.layers[:configs.gpt_layers]
                else:
                    print("Warning: Could not find layer list to slice. Using full model.")
            except Exception as e:
                print(f"Layer slicing failed: {e}")

        self.in_layer = nn.Linear(configs.patch_size, self.d_model)
        self.out_layer = nn.Linear(self.d_model * self.patch_num, configs.pred_len)

        if configs.freeze and configs.pretrain:
            for name, param in self.llm_model.named_parameters():
                if 'norm' in name.lower() or 'ln' in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.in_layer.to(device=device)
        self.out_layer.to(device=device)
        self.padding_patch_layer.to(device=device)
        
        self.cnt = 0


    def forward(self, x, itr):
        B, L, M = x.shape
        
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        
        outputs = self.in_layer(x)
        outputs = outputs.to(dtype=self.llm_model.dtype)
        if self.is_gpt:
            if hasattr(self.llm_model, "model"): 
                # If using AutoModelForCausalLM, the base model is stored in .model
                outputs = self.llm_model.model(inputs_embeds=outputs).last_hidden_state
            else:
                # If using AutoModel (no head), just call it directly
                outputs = self.llm_model(inputs_embeds=outputs).last_hidden_state
        
        #llm model executes on bfloat16 for memory efficiency, convert back to float32 for prediction
        outputs = outputs.to(dtype=torch.float32)
        outputs = self.out_layer(outputs.reshape(B*M, -1))

        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        
        outputs = outputs * stdev
        outputs = outputs + means
        
        return outputs
