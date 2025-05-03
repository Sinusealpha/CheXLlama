#prompt builder script

import abstraction_layer
# we need to first import abstraction_layer here.

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
"""
    return prompt.strip()

# Example usage
image_findings = "The chest X-ray shows a high probability of pneumonia (0.85)."
patient_info = "Patient is a 45-year-old male with a cough for 2 weeks."
medical_context = "Pneumonia is an infection that inflames the air sacs in one or both lungs."
instruction = "Explain the chest X-ray findings in simple terms and suggest possible next steps for the patient."

prompt = create_prompt(image_findings, patient_info, medical_context, instruction)
print("Example Prompt:")
print(prompt)