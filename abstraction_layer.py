# abstraction layer

"""
Mapping/Abstraction Layer Module

This module converts raw output from an image processing model (e.g., a probability vector)
into natural language descriptions or structured formats for use by a language model.

Example:
    Converts [0.85, 0.12, 0.05] to "The image model detects a high probability of pneumonia (0.85)."
"""

from typing import List, Dict, Union, Tuple, Optional

def _validate_inputs(probabilities: List[float], diseases: List[str], threshold: Optional[float] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate input probabilities and diseases for consistency and correctness.

    Args:
        probabilities: List of probabilities from the image model.
        diseases: List of disease names in the same order.
        threshold: Optional probability threshold for validation.

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message). If valid, error_message is None.
    """
    if not probabilities or not diseases:
        return False, "Probabilities and diseases must be non-empty."
    if len(probabilities) != len(diseases):
        return False, "Probabilities and diseases must match in length."
    if not all(isinstance(p, (int, float)) for p in probabilities):
        return False, "All probabilities must be numeric."
    if not all(0 <= p <= 1 for p in probabilities):
        return False, "Probabilities must be between 0 and 1."
    if threshold is not None and not 0 <= threshold <= 1:
        return False, "Threshold must be between 0 and 1."
    return True, None

def map_model_output(probabilities: List[float], diseases: List[str]) -> str:
    """
    Generates sentences for all predicted diseases, with special handling for "No finding".
    """
    is_valid, error = _validate_inputs(probabilities, diseases)
    if not is_valid:
        return f"Error: {error}"

    # Check for "No finding" first
    if "No finding" in diseases:
        return "there is no find disease"
    
    # Generate sentences for all diseases
    sentences = [
        f"the {disease} predicted for this image with the probability {prob:.2f}"
        for disease, prob in zip(diseases, probabilities)
    ]
    return ". ".join(sentences)

def map_model_output_structured(probabilities: List[float], diseases: List[str], threshold: float = 0.5) -> Dict[str, Union[List[str], Dict[str, float], Optional[str]]]:
    """
    Generates structured output with special priority for "No finding".
    """
    is_valid, error = _validate_inputs(probabilities, diseases, threshold)
    if not is_valid:
        return {
            "sentences": [],
            "probabilities": {},
            "error": f"Error: {error}"
        }

    # Handle "No finding" as special case
    if "No finding" in diseases:
        return {
            "sentences": ["there is no find disease"],
            "probabilities": dict(zip(diseases, probabilities)),
            "error": None
        }

    prob_dict = dict(zip(diseases, probabilities))
    sentences = [
        f"the {disease} predicted for this image with the probability {prob:.2f}"
        for disease, prob in prob_dict.items() 
        if prob >= threshold
    ]

    if not sentences:
        sentences = ["No significant findings detected above the threshold."]

    return {
        "sentences": sentences,
        "probabilities": prob_dict,
        "error": None
    }

if __name__ == "__main__":
    # Test case with "No finding"
    print("=== 'No finding' Test ===")
    test_probs = [0.2, 0.7, 0.1]
    test_diseases = ["Pneumonia", "No finding", "Atelectasis"]
    
    print("Basic output:", map_model_output(test_probs, test_diseases))
    structured = map_model_output_structured(test_probs, test_diseases, 0.6)
    print("Structured output:", structured["sentences"])

    # Test case without "No finding"
    print("=== Normal Case ===")
    normal_probs = [0.85, 0.12, 0.05]
    normal_diseases = ["Pneumonia", "Edema", "Atelectasis"]
    
    print("Basic output:", map_model_output(normal_probs, normal_diseases))
    structured = map_model_output_structured(normal_probs, normal_diseases, 0.1)
    print("Structured output:", structured["sentences"])