
This project integrates the CheXNet model for disease prediction with a large language model. This synergy enhances the generation of diagnostic reports and conclusions by incorporating patient-specific information, while also allowing radiologists to interactively query and refine these findings through the LLM.

![Alt text](https://github.com/aminsistani5589/cheXNet/blob/master/image_0%20(1).png)

# Model in Action 

**Scenario: Analyzing a Chest X-Ray for Pneumonia**

1.  **CheXNet Input:** `chest_xray_sample.png`

## Features
- ...

## Built With
- ...

## Getting Started
### Prerequisites
...
### Installation
...

### USAGE
***1.*** Clone this repository to your local machine by running the following command:

git clone <repository_url>

***2.*** Download images of ChestX-ray14 from this [released page](https://nihcc.app.box.com/v/ChestXray-NIHCC) and decompress them to the [directory images](https://github.com/Sinusealpha/cxr-vqa-project/tree/main/ChestX-ray14/images).

***3.*** Update model.py with Local Directory Paths:

***a.*** define the path to the repository.

for example:

path_to_repository="D:\\New folder\\cxr-vqa-project"

***b.*** choose your selected image in  [directory images](https://github.com/Sinusealpha/cxr-vqa-project/tree/main/ChestX-ray14/images) by defining its address.

for example:

SINGLE_TEST_IMAGE = path_to_repository+'\\ChestX-ray14\\images\\***00000003_001.png***'

***4.*** Run the model.py script on your machine to view the results. Feel free to ask any questions afterward. 😀



