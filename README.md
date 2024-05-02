# Open LLM WebUI

This repository contains a web application designed to execute Large Language Models (LLMs).

## Installation

Please follow these steps to install the software:

* Create a new conda environment:

```bash
conda create -n ollm python=3.10
conda activate ollm
```

* Clone the software repository:

```bash
git clone https://github.com/Uminosachi/open-llm-webui.git
cd open-llm-webui
```

* For the CUDA environment, install the following packages:

```bash
pip install -r requirements.txt
```

## Running the application

```bash
python ollm_app.py
```

* Open http://127.0.0.1:7860/ in your browser.

## Downloading the Model

To download the model:
* Launch this application.
* Click on the "Download model" button next to the LLM model ID.
* Wait for the download to complete.
* 🔍 Note: The downloaded model file will be stored in the `.cache/huggingface/hub` directory of your home directory.

### Model List

| Provider      | Model Names                                                                                |
|---------------|--------------------------------------------------------------------------------------------|
| Microsoft     | Phi-3-mini-4k-instruct, Phi-3-mini-128k-instruct                                           |
| Google        | gemma-2b-it, gemma-1.1-2b-it                                                               |
| Apple         | OpenELM-1_1B-Instruct, OpenELM-3B-Instruct                                                 |
| Rakuten       | RakutenAI-7B-chat, RakutenAI-7B-instruct                                                   |
| rinna         | youri-7b-chat, bilingual-gpt-neox-4b-instruction-sft, japanese-gpt-neox-3.6b-instruction-sft-v2, japanese-gpt-neox-3.6b-instruction-ppo |
| TheBloke      | Llama-2-7b-Chat-GPTQ                                                                       |
| stability.ai  | stablelm-tuned-alpha-3b, stablelm-tuned-alpha-7b, japanese-stablelm-instruct-beta-7b       |

* Please check the license in the Model Credit section below.

## Usage

* Enter your message into the "Input text" box.
* Under "Advanced options", adjust the values for "Max New Tokens", "Temperature", "Top k", "Top p", and "Repetition Penalty" as necessary.
* Press "Enter" on your keyboard or click the "Generate" button.
   - ⚠️ Note: If the cloud-based model has been updated, it might be downloaded upon execution.
* If you enable the `CPU execution` checkbox, the model will utilize the argument `device_map="cpu"`.
* When you enable the `Translate (ja->en/en->ja)` checkbox:
   - Any input in Japanese will be translated to English.
   - Responses in English will be translated back into Japanese.
   - ⚠️ Note: Downloading the translation model for the first time may take some time.

![UI image](images/open-ollm-webui_ui_image_1.png)

## Model Credit

| Model                        | Developer           | License                                                        |
|------------------------------|---------------------|----------------------------------------------------------------|
| Phi-3                        | Microsoft           | [The MIT License](https://opensource.org/licenses/MIT)         |
| Gemma                        | Google              | [Gemma Terms of Use](https://ai.google.dev/gemma/terms)        |
| OpenELM                      | Apple               | [Apple sample code license](https://huggingface.co/apple/OpenELM-1_1B-Instruct/blob/main/LICENSE) |
| RakutenAI                    | Rakuten             | [Apache License 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| Youri                        | rinna               | [Llama 2 Community License](https://ai.meta.com/llama/license/) |
| Japanese GPT-NeoX            | rinna               | [The MIT License](https://opensource.org/licenses/MIT)         |
| Llama 2                      | Meta AI             | [Llama 2 Community License](https://github.com/facebookresearch/llama/blob/main/LICENSE) |
| StableLM                     | Stability AI        | [Apache License 2.0](https://github.com/Stability-AI/StableLM/blob/main/LICENSE) |
| Japanese-StableLM-Instruct   | Stability AI        | [Japanese Stablelm Research License Agreement](https://huggingface.co/stabilityai/japanese-stablelm-instruct-alpha-7b/blob/main/LICENSE) |
