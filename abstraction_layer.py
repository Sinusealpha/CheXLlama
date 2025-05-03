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
    Convert image model output to a natural language sentence for the highest probability.

    Args:
        probabilities: List of probabilities from the image model (e.g., [0.85, 0.12, 0.05]).
        diseases: List of disease names in the same order (e.g., ["Pneumonia", "Normal", "Atelectasis"]).

    Returns:
        A sentence like "The image model detects a high probability of Pneumonia (0.85)."
        Returns an error message if inputs are invalid.
    """
    is_valid, error = _validate_inputs(probabilities, diseases)
    if not is_valid:
        return f"Error: {error}"

    max_probability = max(probabilities)
    max_index = probabilities.index(max_probability)
    disease = diseases[max_index]

    return f"The image model detects a high probability of {disease} ({max_probability:.2f})."

def map_model_output_structured(probabilities: List[float], diseases: List[str], threshold: float = 0.5) -> Dict[str, Union[List[str], Dict[str, float], Optional[str]]]:
    """
    Convert image model output to a structured format with sentences for diseases above a threshold.

    Args:
        probabilities: List of probabilities from the image model.
        diseases: List of disease names in the same order.
        threshold: Minimum probability to include in sentences (default: 0.5).

    Returns:
        A dictionary containing:
            - 'sentences': List of sentences for diseases with probabilities >= threshold, or a default message.
            - 'probabilities': Dictionary mapping diseases to their probabilities.
            - 'error': Error message if inputs are invalid (otherwise None).
    """
    is_valid, error = _validate_inputs(probabilities, diseases, threshold)
    if not is_valid:
        return {
            "sentences": [],
            "probabilities": {},
            "error": f"Error: {error}"
        }

    prob_dict = dict(zip(diseases, probabilities))
    sentences = [
        f"The image model detects a high probability of {disease} ({prob:.2f})."
        for disease, prob in prob_dict.items() if prob >= threshold
    ]

    if not sentences:
        sentences = ["No significant findings detected above the threshold."]

    return {
        "sentences": sentences,
        "probabilities": prob_dict,
        "error": None
    }

if __name__ == "__main__":
    # Sample inputs for testing
    probabilities = [0.85, 0.12, 0.05]
    diseases = ["Pneumonia", "Normal", "Atelectasis"]

    # Test map_model_output
    print("Testing map_model_output:")
    result = map_model_output(probabilities, diseases)
    print(result)

    # Test map_model_output_structured
    print("\nTesting map_model_output_structured (threshold=0.5):")
    structured_result = map_model_output_structured(probabilities, diseases, threshold=0.5)
    print("Sentences:", structured_result["sentences"])
    print("Probabilities:", structured_result["probabilities"])
    print("Error:", structured_result["error"])

    # Test edge cases
    print("\nTesting edge cases:")
    test_cases = [
        ([0.85], ["Pneumonia", "Normal"], "Invalid length"),
        ([], [], "Empty inputs"),
        ([0.85, -0.1, 0.05], ["Pneumonia", "Normal", "Atelectasis"], "Invalid probability"),
        ([0.85, 0.12, 0.05], ["Pneumonia", "Normal", "Atelectasis"], 1.5, "Invalid threshold")
    ]

    for probs, dis, desc in test_cases:
        print(f"\nTesting {desc}:")
        result = map_model_output_structured(probs, dis, threshold=0.5 if desc != "Invalid threshold" else 1.5)
        print(result)