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

* If you are using macOS, please install the package from the following file instead:

```bash
pip install -r requirements_mac.txt
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

## Usage

* Enter your message in the Input text box.
* Adjust the values of Max New Tokens, Temperature, Top k, Top p and Repetition penalty as necessary.
* Press Enter on your keyboard or click the "Generate" button.
* When you check the `Translate (ja->en/en->ja)`, Japanese input will be translated into English. Likewise, the responses will be translated back into Japanese (Please note that it may take some time to download the model for the first time).

![UI image](images/open-ollm-webui_ui_image_1.png)

## Model Credit

* **Developed by**: [CyberAgent, Inc.](https://www.cyberagent.co.jp/)
* **Model type**: Transformer-based Language Model
* **Language**: Japanese
* **Library**: [GPT-NeoX](https://github.com/EleutherAI/gpt-neox)
* **License**: OpenCALM is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)). When using this model, please provide appropriate credit to CyberAgent, Inc.
  * Example (en): This model is a fine-tuned version of OpenCALM-XX developed by CyberAgent, Inc. The original model is released under the CC BY-SA 4.0 license, and this model is also released under the same CC BY-SA 4.0 license. For more information, please visit: https://creativecommons.org/licenses/by-sa/4.0/
  * Example (ja): 本モデルは、株式会社サイバーエージェントによるOpenCALM-XXをファインチューニングしたものです。元のモデルはCC BY-SA 4.0ライセンスのもとで公開されており、本モデルも同じくCC BY-SA 4.0ライセンスで公開します。詳しくはこちらをご覧ください: https://creativecommons.org/licenses/by-sa/4.0/
