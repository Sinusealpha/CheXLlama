# xray_model/model_loader.py
import os
import re
import torch
import torch.nn as nn
import torchvision
from torchvision.models import DenseNet121_Weights

class DenseNet121(nn.Module):
    """Modified DenseNet121 with proper weight initialization"""
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.densenet121(x)
    
def load_chexnet_model(CKPT_PATH, N_CLASSES):

    # Initialize model
    model = DenseNet121(N_CLASSES)

    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH)
        state_dict = checkpoint['state_dict']
    
        new_state_dict = {}
        for key, value in state_dict.items():
            # Remove DataParallel module prefix
            new_key = key.replace('module.', '')
        
            # Fix layer numbering pattern (e.g., convert .1 to 1)
            new_key = re.sub(r'\.(\d+)', r'\1', new_key)
    
            new_key = re.sub(
                r'(densenet121\.classifier)(\d+)',  
                r'\1.\2',  #
                new_key
            )             
            new_state_dict[new_key] = value

        # Load with strict checking
        load_result = model.load_state_dict(new_state_dict, strict=False)
    
        if load_result.missing_keys:
            print(f"\n{len(load_result.missing_keys)} MISSING KEYS:")
            for k in load_result.missing_keys[:3]:  # Show first 3 examples
                print(f"- {k}")
            
        if load_result.unexpected_keys:
            print(f"\n{len(load_result.unexpected_keys)} UNEXPECTED KEYS:")
            for k in load_result.unexpected_keys[:3]:  # Show first 3 examples
                print(f"- {k}")
    
        if not load_result.missing_keys and not load_result.unexpected_keys:
            print("\nAll keys matched successfully!")
    
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")
    
    return model

