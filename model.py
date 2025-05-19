
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
from dotenv import load_dotenv

path_to_repository="D:\\New folder\\cxr-vqa-project"

CKPT_PATH = path_to_repository+'\\model.pth.tar'
N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

# this directory contains our test images. download and put the test images into this directory.
DATA_DIR = path_to_repository+'\\ChestX-ray14\\images'
# this directory addresses the image you want to test and get the report from.
# write the name of your image correctly like :00000002_000.png
# but before anything make sure that this test image located in the DATA_DIR directory.
SINGLE_TEST_IMAGE = path_to_repository+'\\ChestX-ray14\\images\\00000003_002.png'
# this directory is not important for our single image predictions but to avoid error please write the address of this text file properly.
TEST_IMAGE_LIST = path_to_repository+'\\ChestX-ray14\\labels\\a.txt'
# batch size is equal to one to see the performance of model for single images.
BATCH_SIZE = 1


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
        
    # Single image prediction
    if os.path.isfile(SINGLE_TEST_IMAGE):
        print("Single Image Prediction Results:")
        probabilities = predict_single_image(model, SINGLE_TEST_IMAGE)
        predictions, warnings = apply_thresholds(probabilities)

        # Create probability dictionaries
        predicted_probs = {cls: probabilities[CLASS_NAMES.index(cls)] for cls in predictions}
        uncertain_probs = {cls: probabilities[CLASS_NAMES.index(cls)] for cls in warnings}

        # Sort predicted classes by probability (descending)
        sorted_predicted = sorted(predicted_probs.items(), 
                                key=lambda x: x[1], 
                                reverse=True)
        
        # Sort uncertain classes by threshold proximity (descending)
        sorted_uncertain = sorted(uncertain_probs.items(),
                                 key=lambda x: abs(x[1] - CLASS_THRESHOLDS[x[0]]),
                                 reverse=True)

        # Separate into ordered lists
        preds = [item[0] for item in sorted_predicted]
        predicted_probs = [item[1] for item in sorted_predicted]
        uncertain_probs = dict(sorted_uncertain)

        # Display results
        print("Final Predictions:")
        for cls, prob in zip(preds, predicted_probs):
            print(f"{cls}: {prob:.4f} ")
        
        if uncertain_probs:
            print("Uncertain Findings:")
            for cls, prob in sorted_uncertain:
                print(f"{cls}: {prob:.4f}")

        return preds, predicted_probs, uncertain_probs


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

def create_prompt(physician_predictions, patient_info, medical_context):
    """
    Creates a highly detailed and structured prompt for a language model,
    designed to elicit comprehensive medical notes and insights for a physician.
    This version emphasizes critical evaluation and correction of the physician's 
    preliminary diagnoses using patient information and medical context.

    Args:
        physician_predictions (str): Text describing the physician's preliminary diagnoses,
                                     which may contain errors requiring correction.
        patient_info (str): Patient data (e.g., age, sex, symptoms, medical history,
                            comorbidities, medications).
        medical_context (str): Additional relevant medical information (e.g., lab results,
                               prior diagnoses, clinical notes).

    Returns:
        str: The formatted, detailed prompt.
    """

    prompt = f"""You are an expert medical assistant AI, tasked with providing a comprehensive, critically evaluated diagnostic assessment to support a physician. Your primary goal is to synthesize all provided information, including potentially incorrect preliminary diagnoses from the physician, and generate a detailed, clinically reasoned, and structured report that enhances diagnostic accuracy. Your role is to identify and correct any inaccuracies in the physician's preliminary diagnoses by thoroughly analyzing the patient information and medical context.

**I. Provided Information:**

*   **Physician's Preliminary Diagnoses:**
    ```
    {physician_predictions}
    ```
    *(CRITICAL NOTE: These are initial diagnoses from the physician which may have errors or be based on incomplete or misinterpreted information. Your role is to critically evaluate, verify, contextualize, and potentially REFINE, CORRECT, or REJECT these preliminary diagnoses by integrating them with all other available patient and clinical information. Do not accept them at face value.)*

*   **Patient Demographics and Clinical History:**
    ```
    {patient_info}
    ```
    *(Consider patient's age, sex, presenting symptoms, relevant past medical history, known comorbidities, medications, allergies, social history, and family history where relevant.)*

*   **Additional Medical Context:**
    ```
    {medical_context}
    ```
    *(This may include lab results, prior diagnoses, specific clinical questions, or other relevant medical information.)*

**II. Your Task: Generate a Detailed Diagnostic Assessment Report**

Based on a thorough synthesis and critical evaluation of ALL the information provided above, please generate a comprehensive report. The overarching goal is to critically evaluate the physician's preliminary diagnoses, identify any errors or inconsistencies, and provide a refined, evidence-based diagnostic conclusion that corrects any inaccuracies.

Please structure your report precisely as follows, addressing each point in detail:

**1. Critical Evaluation of Physician's Preliminary Diagnoses:**
    a.  Briefly list the physician's preliminary diagnoses as provided.
    b.  For each preliminary diagnosis:
        *   Explicitly state whether it is correct, partially correct, or incorrect based on the patient information and medical context.
        *   Provide detailed justifications for your assessment, referencing specific evidence from the patient data and medical context.
    c.  Identify any significant discrepancies, inconsistencies, or diagnoses that seem clinically unlikely or unsupported given the full clinical picture.

**2. Refined Diagnostic Observations:**
    a.  Provide a detailed description of the most significant findings from the patient information and medical context that are relevant to the diagnosis.
    b.  Highlight any findings that support or refute the physician's preliminary diagnoses, emphasizing where corrections are needed.
    c.  Describe any additional observations that are concerning or pertinent to the diagnostic process, which the physician may have overlooked.

**3. Differential Diagnoses:**
    a.  List the most probable differential diagnoses, ordered from most to least likely, based on your refined observations and integrated evidence.
    b.  For each differential diagnosis:
        *   Provide a brief justification referencing specific findings from the patient information and medical context.
        *   Explain how the patient's attributes (e.g., age, symptoms, risk factors) affect the likelihood of this diagnosis.
        *   Mention any critical diagnoses that must be considered or ruled out.
    
**4. Impression and Summary:**
    a.  Provide a concise overall diagnostic impression that synthesizes your refined key findings and their clinical implications.
    b.  Clearly state the main conclusion, specifying whether it agrees with, partially agrees with, or differs from the physician's preliminary diagnoses, and highlight any corrections made.
    c.  Emphasize any critical, acute, or urgent findings requiring immediate attention.

**5. Recommendations:**
    a.  Suggest specific further diagnostic investigations if necessary (e.g., lab tests, imaging, biopsies). Justify why these are needed to confirm your refined assessment or resolve uncertainties.
    b.  Recommend follow-up actions or specialist consultations if appropriate.
    c.  If findings are critical or emergent, clearly state the need for urgent clinical action.

**III. Critical Reporting Guidelines:**

*   **Target Audience:** Experienced Physician. Use precise medical terminology suitable for a professional audience.
*   **Depth and Detail:** Be thorough and provide specific evidence for all conclusions and corrections.
*   **Evidence-Based:** Justify all interpretations and recommendations with integrated evidence from the provided information, adhering to standard medical knowledge and logical clinical reasoning.
*   **Clarity and Structure:** Adhere strictly to the requested report structure. Use bullet points or numbered lists within sections for readability.
*   **Objectivity:** Distinguish between definitive findings and probable interpretations, using cautious and precise language.
*   **Actionability:** Ensure recommendations are specific, practical, and actionable.
*   **Completeness:** Address all sections fully. If information is unavailable, state so and note its potential impact on the assessment.
"""
    return prompt.strip()

# Example usage
image_findings = "The chest X-ray shows a high probability of pneumonia (0.85)."
patient_info = """
Patient Name: John Doe

Age: 65

Sex: Male

ID: 123456

Chief Complaint
Shortness of breath for 3 days
Dry cough and low-grade fever
History of Present Illness
Patient reports worsening dyspnea (shortness of breath) over the last 3 days
Mild, persistent dry cough
Reports mild chest pain on deep inspiration
Fever around 38°C
Past Medical History
Hypertension (high blood pressure), controlled
Type 2 Diabetes Mellitus (diagnosed 5 years ago)
Former smoker: 20 pack-years, quit 7 years ago
No known history of asthma, tuberculosis, or chronic lung disease
Medications
Metformin 1000mg/day
Lisinopril 10mg/day
Allergies
None known
Family History
Father deceased: Myocardial infarction at 70
Mother: Hypertension
Social History
Lives with spouse
Retired office worker
Physical Examination (Relevant findings):
Temperature: 38.2°C
Respiratory rate: 22/min
Pulse: 90/min
Blood pressure: 130/80 mmHg
Oxygen saturation: %91  on room air
Fine crackles at both lung bases
Laboratory Results
White blood cells: 8,400 /μL (slightly elevated)
C-Reactive Protein (CRP): Elevated
D-dimer: Mildly elevated
Blood glucose: 150 mg/dl
Radiology Request:
Chest X-ray and/or CT scan to evaluate for infectious or inflammatory lung disease (e.g., pneumonia, COVID-19, heart failure)."""
medical_context = "Pneumonia is an infection that inflames the air sacs in one or both lungs."

#########################################################################################################
"""
preparing our API to retrieve responses from the Language Model.
"""

def API (prompt):
    
    # Load environment variables FIRST
    load_dotenv(path_to_repository+"\\API.env")  # This should be outside the OpenAI constructor
    
    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("API key not found in environment variables")
    
    # Then create the client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key  # Use the retrieved key
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
    
    prompt = create_prompt(image_findings, patient_info, medical_context)
    print("Example Prompt:")
    print('prompt:',prompt)
    
    print("LM_response:")
    LM_response=API (prompt)
    
    