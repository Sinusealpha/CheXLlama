# CheXLlama_RadiographAICoPilot

This project integrates the CheXNet model for disease prediction with a large language model. This synergy enhances the generation of diagnostic reports and conclusions by incorporating patient-specific information, while also allowing radiologists to query and refine these findings through the LLM interactively.

![Alt text](https://github.com/Sinusealpha/CheXLlama_RadiographAICoPilot/blob/main/1404-02-21%2020.32.35.jpg)

## Model in Action 

**Scenario: Analyzing a Chest X-Ray for Pneumonia**

1.  **CheXNet Input:** `chest_xray_sample.png`

## Features

*   **Synergistic AI Integration:** Seamlessly connects the state-of-the-art **CheXNet** radiology image classification model with the powerful **Llama 3.3 Nemotron Super 49B v1 API**.
*   **Automated Radiological Image Analysis:** Leverages CheXNet to automatically predict potential findings from chest X-ray images.
*   **Intelligent Question-Answering:** Allows users to perform in-depth, contextual Q&A sessions with the Llama LLM, leveraging initial findings from CheXNet and the patient's relevant medical data as model inputs.
*   **Enhanced Understanding of Medical Imagery:** Moves beyond simple classification to allow for nuanced exploration and explanation of potential conditions identified in radiology scans.
*   **Flexible Querying:** Users can ask a wide range of questions, from simple clarifications of medical terms to more complex inquiries about potential implications or follow-up considerations.
*   **(Potentially) Extensible Framework:** The architecture can serve as a foundation for integrating other AI models or data sources in the future.
  
## Built With

**Core AI Models & APIs:**
*   [![CheXNet](https://img.shields.io/badge/AI%20Model-CheXNet-blueviolet)](https://stanfordmlgroup.github.io/projects/chexnet/) - For radiological image classification and prediction.
*   [![Llama 3.3 Nemotron](https://img.shields.io/badge/LLM%20API-Llama%203.3%20Nemotron%20Super%2049B%20v1-brightgreen)](https://developer.nvidia.com/nemotron-3-8b) - to interact with language model via API

## Getting Started

### Prerequisites

*  Python 3.4+
*  [PyTorch](https://pytorch.org/) and its dependencies
*  [openai](https://pypi.org/project/openai/) library

### USAGE
***1.*** Clone this repository to your local machine by running the following command:

git clone <repository_url>

***2.*** Download images of ChestX-ray14 from this [released page](https://nihcc.app.box.com/v/ChestXray-NIHCC) and decompress them to the [directory images](https://github.com/Sinusealpha/cxr-vqa-project/tree/main/ChestX-ray14/images).

***3.*** Update model.py with Local Directory Paths:

***a.*** Define the path to the repository.

For example:

path_to_repository="D:\\New folder\\cxr-vqa-project"

***b.*** choose your selected image in  [directory images](https://github.com/Sinusealpha/cxr-vqa-project/tree/main/ChestX-ray14/images) by defining its address.

For example:

SINGLE_TEST_IMAGE = path_to_repository+'\\ChestX-ray14\\images\\***00000003_001.png***'

***4.*** Run the model.py script on your machine to view the results. Feel free to ask any questions afterward. 😀



