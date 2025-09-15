# utils/data_mapping.py

def format_findings_for_prompt(predictions, uncertainties, predicted_probs, uncertain_probs):
    """
    Formats the model's findings into a clear, structured text sentence
    for inclusion in the LLM prompt.
    """
    lines = []
    
    if "No Finding" in predictions:
        return "The AI model detected no significant findings on the chest X-ray."
    
    if predictions:
        lines.append("The AI model suggests the presence of the following findings:")
        sorted_preds = sorted(predicted_probs.items(), key=lambda item: item[1], reverse=True)
        for finding, prob in sorted_preds:
            lines.append(f"- {finding} (Probability: {prob:.2f})")
    
    if uncertainties:
        lines.append("\nThe model is uncertain about the following findings (probabilities are near the diagnostic threshold):")
        sorted_uncertain = sorted(uncertain_probs.items(), key=lambda item: item[1], reverse=True)
        for finding, prob in sorted_uncertain:
            lines.append(f"- {finding} (Probability: {prob:.2f})")
            
    return "\n".join(lines)