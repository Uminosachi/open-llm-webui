# Open LLM WebUI

This repository contains a web application designed to execute relatively compact, locally-operated Large Language Models (LLMs).

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

* Since Mac OS does not support CUDA, please use the following command:

```bash
BUILD_CUDA_EXT=0 pip install -r requirements.txt
```

* If you want to use the llama.cpp with CUDA acceleration, please use the following command:

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install -r requirements.txt
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

### Model List (transformers)

| Provider      | Model Names                                                                                |
|---------------|--------------------------------------------------------------------------------------------|
| Microsoft     | Phi-3-mini-4k-instruct, Phi-3-mini-128k-instruct                                           |
| Google        | gemma-1.1-2b-it, gemma-1.1-7b-it                                                           |
| NVIDIA        | Llama3-ChatQA-1.5-8B                                                                       |
| Apple         | OpenELM-1_1B-Instruct, OpenELM-3B-Instruct                                                 |
| Rakuten       | RakutenAI-7B-chat, RakutenAI-7B-instruct                                                   |
| rinna         | youri-7b-chat, bilingual-gpt-neox-4b-instruction-sft, japanese-gpt-neox-3.6b-instruction-sft-v2, japanese-gpt-neox-3.6b-instruction-ppo |
| TheBloke      | Llama-2-7b-Chat-GPTQ, Kunoichi-7B-GPTQ                                                     |
| Stability AI  | stablelm-tuned-alpha-3b, stablelm-tuned-alpha-7b, japanese-stablelm-instruct-beta-7b       |

* To download Google's Gemma model, please ensure you have obtained the necessary access rights beforehand via the [Hugging Face page](https://huggingface.co/google/gemma-1.1-2b-it).
  - Before downloading the model, you need to log in to Hugging Face via the command line. Please use the following command:
```
huggingface-cli login
```
* To download models based on Llama 2, please ensure you have obtained the necessary access rights beforehand via the [Hugging Face page](https://huggingface.co/meta-llama/Llama-2-7b-hf).
* To download models based on Llama 3, please ensure you have obtained the necessary access rights beforehand via the [Hugging Face page](https://huggingface.co/meta-llama/Meta-Llama-3-8B).

* Please check the license in the Model Credit section below.

### Model List (llama.cpp)

| Provider      | Model Names                                                                                |
|---------------|--------------------------------------------------------------------------------------------|
| Microsoft     | Phi-3-mini-4k-instruct-q4.gguf, Phi-3-mini-4k-instruct-fp16.gguf                           |

* 🔍 Note: Place files with the `.gguf` extension in the `models` directory within the `open-llm-webui` folder. These files will then appear in the model list on the `llama.cpp` tab of the web UI, and can be used accordingly.
* 📝 Note: If the metadata of a GGUF model includes `tokenizer.chat_template`, this template will be used to create the prompts.
* 🛡️ Reminder: Since the Llama 2 tokenizer is utilized, please ensure that you have obtained the necessary permissions in advance by visiting the [Hugging Face page](https://huggingface.co/meta-llama/Llama-2-7b-hf).

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

| Developer           | Model                        | License                                                        |
|---------------------|------------------------------|----------------------------------------------------------------|
| Microsoft           | Phi-3                        | [The MIT License](https://opensource.org/licenses/MIT)         |
| Google              | Gemma                        | [Gemma Terms of Use](https://ai.google.dev/gemma/terms)        |
| NVIDIA              | Llama3-ChatQA                | [Llama 3 Community License](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/LICENSE) |
| Apple               | OpenELM                      | [Apple sample code license](https://huggingface.co/apple/OpenELM-1_1B-Instruct/blob/main/LICENSE) |
| Rakuten             | RakutenAI                    | [Apache License 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| rinna               | Youri                        | [Llama 2 Community License](https://ai.meta.com/llama/license/) |
| rinna               | Japanese GPT-NeoX            | [The MIT License](https://opensource.org/licenses/MIT)         |
| Meta AI             | Llama 2                      | [Llama 2 Community License](https://github.com/facebookresearch/llama/blob/main/LICENSE) |
| Sanji Watsuki       | Kunoichi-7B                  | [CC-BY-NC-4.0](https://spdx.org/licenses/CC-BY-NC-4.0)         |
| Stability AI        | StableLM                     | [Apache License 2.0](https://github.com/Stability-AI/StableLM/blob/main/LICENSE) |
| Stability AI        | Japanese-StableLM-Instruct   | [Japanese Stablelm Research License Agreement](https://huggingface.co/stabilityai/japanese-stablelm-instruct-alpha-7b/blob/main/LICENSE) |
