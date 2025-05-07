
"""
The main CheXNet model implementation 
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
from openai import OpenAI


CKPT_PATH = 'D:\\projects\\vqa_chest\\chexnet\\model.pth.tar'
N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

# this directory contains our test images. download and put the test images into this directory.
DATA_DIR = 'D:\\projects\\vqa_chest\\chexnet\\ChestX-ray14\\images'
# this directory is not important for our single image predictions but to avoid error please write the address of this text file properly.
TEST_IMAGE_LIST = 'D:\\projects\\vqa_chest\\chexnet\\ChestX-ray14\\labels\\a.txt'
# batch size is equal to one to see the performance of model for single images.
BATCH_SIZE = 1
# this directory addresses the image you want to test and get the report from.
# write the name of your image correctly like :00000002_000.png
# but before anything make sure that this test image located in the DATA_DIR directory.
SINGLE_TEST_IMAGE = 'D:\\projects\\vqa_chest\\chexnet\\ChestX-ray14\\images\\00000003_001.png'

# Threshold configuration
NO_FINDING_THRESHOLD = 1
CLASS_THRESHOLDS = {
    'Atelectasis': 0.5,
    'Cardiomegaly': 0.6,
    'Effusion': 0.6,
    'Infiltration': 0.5,
    'Mass': 0.5,
    'Nodule': 0.5,
    'Pneumonia': 0.5,
    'Pneumothorax': 0.65,
    'Consolidation': 0.5,
    'Edema': 0.55,
    'Emphysema': 0.55,
    'Fibrosis': 0.62,
    'Pleural_Thickening': 0.55,
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

# Determining an interval for handling uncertainties for user:
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
    probabilities = None
    if os.path.isfile(SINGLE_TEST_IMAGE):
        print("\n\nSingle Image Prediction Results:")
        probabilities = predict_single_image(model, SINGLE_TEST_IMAGE)

        print(f"Image: {os.path.basename(SINGLE_TEST_IMAGE)}")
        print("Pathology Probabilities:")
        for cls, prob in zip(CLASS_NAMES, probabilities):
            print(f"{cls + ':':<20} {prob:.4f}")


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
    
    # Create dictionary of only predicted class probabilities
    predicted_probs = {cls: probabilities[CLASS_NAMES.index(cls)] for cls in preds}
    
    # Create dictionary of uncertain classes and their probabilities
    uncertain_probs = {cls: probabilities[CLASS_NAMES.index(cls)] for cls in uncertain_classes}
    
    # Separate into two lists: predicted classes and their probabilities
    preds = list(predicted_probs.keys())
    predicted_probs = list(predicted_probs.values())
    
    return preds,predicted_probs,uncertain_probs


#########################################################################################################
"""
our prompt builder for preparing model outputs for LM. 
"""

from abstraction_layer import (
    _validate_inputs,
    map_model_output,
    map_model_output_structured
)

# test_probs = [0.2, 0.7, 0.1]
# test_diseases = ["Pneumonia", "No finding", "Atelectasis"]
# print(map_model_output(test_probs, test_diseases))
# structured = map_model_output_structured(test_probs, test_diseases)
# print(structured)


def create_prompt(image_findings, patient_info, medical_context, instruction):
    """
    Creates a prompt for the language model by combining image findings, patient info, 
    medical context, and an instruction, optimized for clarity and specificity.
    
    Args:
        image_findings (str): Text describing the image model's prediction.
        patient_info (str): Patient data (e.g., age, sex, symptoms).
        medical_context (str): Relevant medical information.
        instruction (str): What the language model should do.
    
    Returns:
        str: The formatted prompt.
    """
    
    prompt = f"""You are a medical assistant tasked with supporting a radiologist. Below is the context and task for your response. Use clear, concise language and structure your answer to directly address the instruction.

**Context**:
- **Role**: Medical assistant providing insights to a radiologist.
- **Image Findings**: {image_findings}
- **Patient Information**: {patient_info}
- **Medical Context**: {medical_context}

**Task**:
{instruction}

**Guidelines**:
- Ensure explanations are accurate and tailored to the radiologist's expertise.
- If suggesting actions, prioritize evidence-based recommendations.
- Avoid speculative or irrelevant details.
- Ensure your response meets the following criteria:   
- Be concise and clear.   

"""
    return prompt.strip()

# Example usage
image_findings = "The chest X-ray shows a high probability of pneumonia (0.85)."
patient_info = "Patient is a 45-year-old male with a cough for 2 weeks."
medical_context = "Pneumonia is an infection that inflames the air sacs in one or both lungs."
instruction = "Explain the chest X-ray findings in simple terms and suggest possible next steps for the patient."

#########################################################################################################
"""
preparing our API to retrieve responses from the Language Model.
"""

def API (prompt):
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
    )

    messages = []
    user_input = prompt
    while True:
        
        if user_input.lower() in ["quit", "exit"]:
            break
    
        messages.append({"role": "user", "content": user_input})
    
        # Keep only last 5 messages
        if len(messages) > 5:
            messages = messages[-5:]
    
        response = client.chat.completions.create(
            model="nvidia/llama-3.3-nemotron-super-49b-v1:free",
            messages=messages
        )
    
        bot_response = response.choices[0].message.content
        print("Bot:", bot_response)
        
        user_input=input("---------------------------------------------------\nPlease feel free to share any follow-up questions\nI’m here to provide answers.\n")
    
        messages.append({"role": "assistant", "content": bot_response})
    

##################################################################################

if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    torch.set_default_dtype(torch.float)
    torch.set_default_device('cpu')
    preds,predicted_probs,uncertain_probs=main()
    
    # Converting all the probabilities from <class 'numpy.float32'> to float
    predicted_probs = [float(num) for num in predicted_probs]
    
    structured = map_model_output_structured(predicted_probs,preds)
    print("structured:",structured)
    
    image_findings=structured['sentences']
    
    prompt = create_prompt(image_findings, patient_info, medical_context, instruction)
    print("Example Prompt:")
    print('prompt:',prompt)
    
    print("LM_response:")
    LM_response=API (prompt)
    
    