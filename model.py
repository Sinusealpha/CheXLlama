# encoding: utf-8

"""
The main CheXNet model implementation (CPU version) with single image prediction capability.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score
from torchvision.models import DenseNet121_Weights
from PIL import Image
import re

CKPT_PATH = 'D:\\projects\\vqa_chest\\chexnet\\model.pth.tar'
N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = 'D:\\projects\\vqa_chest\\chexnet\\ChestX-ray14\\images'
TEST_IMAGE_LIST = 'D:\\projects\\vqa_chest\\chexnet\\ChestX-ray14\\labels\\a.txt'
BATCH_SIZE = 1
SINGLE_TEST_IMAGE = 'D:\\projects\\vqa_chest\\chexnet\\ChestX-ray14\\images\\00000003_003.png'

# Threshold configuration
NO_FINDING_THRESHOLD = 0.75
CLASS_THRESHOLDS = {
    'Atelectasis': 0.7,
    'Cardiomegaly': 0.8,
    'Effusion': 0.7,
    'Infiltration': 0.9,
    'Mass': 0.7,
    'Nodule': 0.7,
    'Pneumonia': 0.8,
    'Pneumothorax': 0.7,
    'Consolidation': 0.8,
    'Edema': 0.8,
    'Emphysema': 0.85,
    'Fibrosis': 0.8,
    'Pleural_Thickening': 0.8,
    'Hernia': 0.85
}

# Normalization parameters
normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])

def ten_crop_to_tensor(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])

def normalize_crops(crops):
    return torch.stack([normalize(crop) for crop in crops])

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

def preprocess_image(image_path):
    """Process single image for model input"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(ten_crop_to_tensor),
        transforms.Lambda(normalize_crops)
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  

def predict_single_image(model, image_path):
    processed_image = preprocess_image(image_path)
    with torch.no_grad():
        bs, n_crops, c, h, w = processed_image.size()
        output = model(processed_image.view(-1, c, h, w))
        output_mean = output.view(bs, n_crops, -1).mean(1)
        return output_mean.squeeze().numpy()

def compute_AUCs(gt, pred):
    """Calculate AUROC scores for all classes"""
    aucs = []
    for i in range(N_CLASSES):
        unique_classes_gt = np.unique(gt[:, i].numpy())
        if len(unique_classes_gt) < 2:
            auc_val = np.nan
        else:
            auc_val = roc_auc_score(gt[:, i], pred[:, i])
        aucs.append(auc_val)
    return aucs

UNCERTAINTY_MARGIN = 0.1

def apply_thresholds(probabilities):

    predictions = [cls for cls, prob in zip(CLASS_NAMES, probabilities) 
                   if prob > CLASS_THRESHOLDS[cls]]
    
    warnings = []
    if probabilities.size > 0:  
        max_prob = probabilities.max()
        
        # Check uncertainty for all classes
        for cls, prob in zip(CLASS_NAMES, probabilities):
            threshold = CLASS_THRESHOLDS[cls]
            if abs(prob - threshold) <= UNCERTAINTY_MARGIN:
                warnings.append(cls)
        
        # Check No Finding uncertainty
        if abs(max_prob - NO_FINDING_THRESHOLD) <= UNCERTAINTY_MARGIN:
            warnings.append('No Finding')
    ###################### check for removing not predictions later
    if not predictions and max_prob < NO_FINDING_THRESHOLD:
        return ['No Finding'], warnings
    return predictions, warnings


def main():
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

    # Test dataset evaluation
    test_dataset = ChestXrayDataSet(
        data_dir=DATA_DIR,
        image_list_file=TEST_IMAGE_LIST,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(ten_crop_to_tensor),
            transforms.Lambda(normalize_crops)
        ])
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )

    # Full dataset evaluation
    model.eval()
    gt = torch.FloatTensor()
    pred = torch.FloatTensor()

    with torch.no_grad():
        for i, (inp, target) in enumerate(test_loader):
            gt = torch.cat((gt, target), 0)
            bs, n_crops, c, h, w = inp.size()
            output = model(inp.view(-1, c, h, w))
            output_mean = output.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, output_mean.data), 0)
            
    # Calculate metrics
    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print(f'\nAverage AUROC: {AUROC_avg:.3f}')
    for i in range(N_CLASSES):
        print(f'{CLASS_NAMES[i]}: {AUROCs[i]:.3f}')

    # Single image prediction
    if os.path.isfile(SINGLE_TEST_IMAGE):
        print("\n\nSingle Image Prediction Results:")
        probabilities = predict_single_image(model, SINGLE_TEST_IMAGE)
    
        print(f"Image: {os.path.basename(SINGLE_TEST_IMAGE)}")
        print("Pathology Probabilities:")
        for cls, prob in zip(CLASS_NAMES, probabilities):
            print(f"{cls + ':':<20} {prob:.4f}")

    # In main execution section
    print("Diagnosis Report:")
    preds, uncertain_classes = apply_thresholds(probabilities)

    # Print uncertainty warnings
    if uncertain_classes:
        print("\x1b[33m*** REQUIRES DOCTOR REVIEW ***\x1b[0m")
        for cls in uncertain_classes:
            # Handle No Finding special case
            if cls == 'No Finding':
                value = max(probabilities)
                threshold = NO_FINDING_THRESHOLD
            else:
                value = probabilities[CLASS_NAMES.index(cls)]
                threshold = CLASS_THRESHOLDS[cls]
            
            status = "above" if value > threshold else "below"
            print(f"\x1b[33m- {cls}: {value:.2f} ({status} {threshold:.2f}±{UNCERTAINTY_MARGIN})\x1b[0m")

    # Print final predictions
    print("Final Conclusions:")
    if preds:
        print(*preds, sep='-')
    else:
        print("No definitive diagnosis")


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    torch.set_default_dtype(torch.float)
    torch.set_default_device('cpu')
    main()
