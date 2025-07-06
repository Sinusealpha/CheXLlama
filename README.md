-----

# CheXLlama
![Alt text](https://github.com/Sinusealpha/CheXLlama/blob/main/1404-02-21%2020.32.35.jpg)
CheXLlama is an open-source project that integrates the powerful **CheXNet** model for disease prediction with **Llama 3.3 Nemotron Super 49B v1**, a large language model (LLM). This synergistic approach enhances the generation of diagnostic reports and conclusions by incorporating patient-specific information, while also empowering radiologists to interactively query and refine these findings through the LLM.

-----

## 🌟 Model in Action

Imagine a radiologist analyzing a chest X-ray. CheXLlama streamlines the process:

1.  **CheXNet Input:** A chest X-ray image (e.g., `chest_xray_sample.png`) is fed into CheXNet.
2.  **Automated Analysis:** CheXNet quickly identifies potential findings, such as "Pneumonia detected."
3.  **LLM Integration:** This initial finding, combined with relevant patient data, is passed to the Llama LLM.
4.  **Interactive Querying:** The radiologist can then ask specific questions like:
      * "What are the typical symptoms associated with this finding?"
      * "Are there any other conditions that might present similarly?"
      * "What follow-up procedures are recommended?"
5.  **Refined Reports:** The LLM provides intelligent, context-aware answers, helping the radiologist generate more comprehensive and accurate diagnostic reports.

-----

## ✨ Features

  * **Synergistic AI Integration:** Seamlessly combines the cutting-edge **CheXNet** radiology image classification model with the robust **Llama 3.3 Nemotron Super 49B v1 API**.
  * **Automated Radiological Image Analysis:** Leverages CheXNet to automatically predict potential findings from chest X-ray images, providing a rapid initial assessment.
  * **Intelligent Question-Answering:** Enables users to engage in in-depth, contextual Q\&A sessions with the Llama LLM, utilizing initial CheXNet findings and patient-specific medical data as inputs.
  * **Enhanced Understanding of Medical Imagery:** Moves beyond simple classification, allowing for nuanced exploration and explanation of potential conditions identified in radiology scans.
  * **Flexible Querying:** Supports a wide range of user queries, from basic clarifications of medical terms to complex inquiries about potential implications, differential diagnoses, or recommended follow-up considerations.
  * **(Potentially) Extensible Framework:** The modular architecture serves as a flexible foundation for future integration of other AI models, data sources, or specialized medical knowledge bases.

-----

## 🛠️ Built With

**Core AI Models & APIs:**

  * [](https://stanfordmlgroup.github.io/projects/chexnet/) - Utilized for precise radiological image classification and prediction.
  * [](https://developer.nvidia.com/nemotron-3-8b) - The large language model API for interactive querying and report generation.

-----

## 🚀 Getting Started

Follow these steps to set up and run CheXLlama on your local machine.

### Prerequisites

Ensure you have the following installed:

  * Python 3.4+
  * [PyTorch](https://pytorch.org/) and its dependencies
  * [openai](https://pypi.org/project/openai/) library (for API interaction)

### Usage

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Sinusealpha/CheXLlama.git
    cd CheXLlama
    ```

2.  **Download ChestX-ray14 Images:**

      * Download the ChestX-ray14 dataset images from the official [released page](https://nihcc.app.box.com/v/ChestXray-NIHCC).
      * Decompress the downloaded files and place them into the `ChestX-ray14/images` directory within your cloned repository.
          * *Expected path structure:* `your_cloned_repo/ChestX-ray14/images/`

3.  **Obtain Llama 3.3 Nemotron API Key:**

      * Create your API key for Llama 3.3 from [OpenRouter.ai](https://openrouter.ai/nvidia/llama-3.3-nemotron-super-49b-v1:free).
      * Create a file named `API.env` in the root of your cloned repository and add your API key in the following format:
        ```
        API_KEY="your_api_key_here"
        ```

4.  **Update `model.py` with Local Directory Paths:**

      * Open `model.py` in your preferred text editor.

      * **Define the path to your repository:**

        ```python
        path_to_repository = "D:\\New folder\\cxr-vqa-project" # Update this to your actual path
        ```

        *(Replace `"D:\\New folder\\cxr-vqa-project"` with the absolute path to your cloned CheXLlama directory.)*

      * **Choose your selected image:**

          * Within `model.py`, locate the `SINGLE_TEST_IMAGE` variable.
          * Define its address, pointing to an image within your `ChestX-ray14/images` directory:
            ```python
            SINGLE_TEST_IMAGE = path_to_repository + '\\ChestX-ray14\\images\\00000003_001.png' # Update filename as needed
            ```
            *(Make sure the image filename corresponds to an actual image you downloaded.)*

5.  **Run the Model:**

      * Execute the `model.py` script from your terminal:
        ```bash
        python model.py
        ```
      * The script will process the image and provide initial results. Feel free to ask any follow-up questions within the interactive session\! 😀

-----

## 📄 Project Paper

For a more in-depth understanding of the methodology and experimental results, please refer to our research paper on arXiv:

  * **[Link]** *(it will be added very soon.)*
