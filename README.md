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

| Provider | Model Names |
| --- | --- |
| microsoft | Phi-3-mini-4k-instruct, Phi-3-mini-128k-instruct |
| google | gemma-2b-it, gemma-1.1-2b-it |
| apple | OpenELM-1_1B-Instruct, OpenELM-3B-Instruct |
| rinna | bilingual-gpt-neox-4b-instruction-sft, japanese-gpt-neox-3.6b-instruction-sft-v2, japanese-gpt-neox-3.6b-instruction-ppo |
| TheBloke | Llama-2-7b-Chat-GPTQ, Llama-2-13B-chat-GPTQ |
| stabilityai | stablelm-tuned-alpha-3b, stablelm-tuned-alpha-7b, japanese-stablelm-instruct-beta-7b |

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

### Phi-3

* **Developed by**: [Microsoft](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) Copyright (c) 2024 Microsoft Corp.
* **License**: [The MIT License](https://opensource.org/licenses/MIT)

### Gemma

* **Developed by**: [Google](https://huggingface.co/google/gemma-2b-it) Copyright (c) 2024 Google LLC
* **License**: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)

### OpenELM

* **Developed by**: [Apple](https://huggingface.co/apple/OpenELM) Copyright (c) 2024 Apple Inc.
* **License**: [Apple sample code license](https://huggingface.co/apple/OpenELM-1_1B-Instruct/blob/main/LICENSE)

### rinna

* **Developed by**: [rinna Co., Ltd.](https://rinna.co.jp/) Copyright (c) 2023 rinna Co., Ltd.
* **License**: [The MIT License](https://opensource.org/licenses/MIT)

### Llama 2

* **Developed by**: [Meta AI](https://ai.meta.com/) Copyright (c) 2023 Meta Platforms, Inc.
* **License**: [Llama 2 Community License](https://github.com/facebookresearch/llama/blob/main/LICENSE)

### StableLM

* **Developed by**: [Stability AI](https://stability.ai/) Copyright (c) 2023 Stability AI, Ltd.
* **License**: [Apache License 2.0](https://github.com/Stability-AI/StableLM/blob/main/LICENSE)

### Japanese-StableLM-Instruct

* **Developed by**: [Stability AI](https://stability.ai/) Copyright (c) 2023 Stability AI, Ltd.
* **License**: [Japanese Stablelm Research License Agreement](https://huggingface.co/stabilityai/japanese-stablelm-instruct-alpha-7b/blob/main/LICENSE)
