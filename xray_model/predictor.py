# xray_model/predictor.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from config import CLASS_NAMES, CLASS_THRESHOLDS, NO_FINDING_THRESHOLD, UNCERTAINTY_MARGIN


# Normalization parameters
normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])

def _preprocess_image_for_prediction(image_path):
    """
    Preprocesses a single image for model input using TenCrop augmentation.
    Converts a single image into a batch of 10 crops.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image)


def run_prediction(model, image_path):
    """
    Runs inference on a single image and returns the raw probabilities.
    """
    model.eval()
    processed_image_tensor = _preprocess_image_for_prediction(image_path)
    
    # Add a batch dimension and send to the specified device
    processed_image_tensor = processed_image_tensor.unsqueeze(0)
    
    with torch.no_grad():
        bs, n_crops, c, h, w = processed_image_tensor.size()
        # Reshape for model input and get predictions
        input_var = processed_image_tensor.view(-1, c, h, w)
        output = model(input_var)
        # Average the predictions across the 10 crops
        output_mean = output.view(bs, n_crops, -1).mean(1)
        
    probabilities = output_mean.cpu().squeeze().numpy()
    return probabilities

def interpret_probabilities(probabilities):
    """
    Applies thresholds to raw probabilities to generate predictions and warnings.
    """
    predictions = [
        cls for cls, prob in zip(CLASS_NAMES, probabilities) 
        if prob > CLASS_THRESHOLDS.get(cls, 0.5)
    ]
    
    uncertainties = []
    if probabilities.size > 0:
        for cls, prob in zip(CLASS_NAMES, probabilities):
            threshold = CLASS_THRESHOLDS.get(cls, 0.5)
            if abs(prob - threshold) <= UNCERTAINTY_MARGIN:
                uncertainties.append(cls)

    if not predictions and probabilities.max() < NO_FINDING_THRESHOLD:
        return ['No Finding'], [], probabilities.tolist(), {}
    
    # Create dictionaries of probabilities for predicted and uncertain classes
    predicted_probs_map = {cls: probabilities[CLASS_NAMES.index(cls)] for cls in predictions}
    uncertain_probs_map = {cls: probabilities[CLASS_NAMES.index(cls)] for cls in uncertainties}
    
    return predictions, uncertainties, predicted_probs_map, uncertain_probs_map

