# Open LLM WebUI

This repository contains a web application designed to execute Large Language Models (LLMs), such as the [Open CALM model](https://huggingface.co/cyberagent) and the [Japanese GPT-NeoX model](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft).

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

1. Launch this application.
2. Click on the "Download model" button next to the Open LLM model ID.
3. Wait for the download to complete.
4. The downloaded model file will be stored in the `.cache/huggingface/hub` directory of your home directory.

### Model List

| Provider | Model Names |
| --- | --- |
| rinna | bilingual-gpt-neox-4b, bilingual-gpt-neox-4b-instruction-sft, japanese-gpt-neox-3.6b, japanese-gpt-neox-3.6b-instruction-sft, japanese-gpt-neox-3.6b-instruction-sft-v2, japanese-gpt-neox-3.6b-instruction-ppo |
| TheBloke | Llama-2-7b-Chat-GPTQ, Llama-2-13B-chat-GPTQ |
| stabilityai | stablelm-tuned-alpha-3b, stablelm-tuned-alpha-7b, japanese-stablelm-base-alpha-7b, japanese-stablelm-instruct-alpha-7b |
| cyberagent | open-calm-small, open-calm-medium, open-calm-large, open-calm-1b, open-calm-3b, open-calm-7b |
| decapoda-research | llama-7b-hf, llama-13b-hf |

* Please check the license in the Model Credit section below.

## Usage

* Enter your message in the Input text box.
* Adjust the values of Max New Tokens, Temperature, Top k, Top p and Repetition penalty as necessary.
* Press Enter on your keyboard or click the "Generate" button.
* When you check the `Translate (ja->en/en->ja)`, Japanese input will be translated into English. Likewise, the responses will be translated back into Japanese (Please note that it may take some time to download the model for the first time).

![UI image](images/open-ollm-webui_ui_image_1.png)

## Model Credit

### rinna

* **Developed by**: [rinna Co., Ltd.](https://rinna.co.jp/) Copyright (c) 2023 rinna Co., Ltd.
* **License**: [The MIT license](https://opensource.org/licenses/MIT)

### Llama 2

* **Developed by**: [Meta AI](https://ai.meta.com/) Copyright (c) 2023 Meta Platforms, Inc.
* **License**: [Llama 2 Community License](https://github.com/facebookresearch/llama/blob/main/LICENSE)

### StableLM & Japanese-StableLM-Base

* **Developed by**: [Stability AI](https://stability.ai/) Copyright (c) 2023 Stability AI, Ltd.
* **License**: [Apache License 2.0](https://github.com/Stability-AI/StableLM/blob/main/LICENSE)

### Japanese-StableLM-Instruct

* **Developed by**: [Stability AI](https://stability.ai/) Copyright (c) 2023 Stability AI, Ltd.
* **License**: [Japanese Stablelm Research License Agreement](https://huggingface.co/stabilityai/japanese-stablelm-instruct-alpha-7b/blob/main/LICENSE)

### OpenCALM

* **Developed by**: [CyberAgent, Inc.](https://www.cyberagent.co.jp/) Copyright (c) 2023 CyberAgent, Inc.
* **License**: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) Creative Commons Attribution-ShareAlike 4.0 International License
