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

### Python Package Installation

#### General Instructions

* Install the necessary Python packages by executing:

  ```bash
  pip install -r requirements.txt
  ```

#### (Optional) Installation for Flash Attention

* To enable Flash Attention in some models, if CUDA is available, install Flash Attention:

  ```bash
  pip install packaging ninja
  pip install flash-attn --no-build-isolation
  ```

#### Platform-Specific Instructions

* **For Windows (with CUDA support):**

  ##### Install pre-build wheel for Windows

  * It is possible to install a pre-built wheel with CUDA support.
    * Source URL: [https://abetlen.github.io/llama-cpp-python/whl/cu121/llama-cpp-python/](https://abetlen.github.io/llama-cpp-python/whl/cu121/llama-cpp-python/)

    ```bash
    wget https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu121/llama_cpp_python-0.3.4-cp310-cp310-win_amd64.whl
    pip install llama_cpp_python-0.3.4-cp310-cp310-win_amd64.whl
    pip install -r requirements.txt
    ```

  ##### (Optional) Build with CUDA for Windows

  * Install [Visual Studio](https://learn.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2022):
    * ⚠️ Important: Make sure to select `Desktop development with C++` during the installation process.
  * Copy MSBuild extensions for CUDA as an administrator (adjust the CUDA version `v12.1` as needed):

    ```bash
    xcopy /e "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\extras\visual_studio_integration\MSBuildExtensions" "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations"
    ```

  * Configure the required environment variables for the build (adjust the CUDA version as necessary):

    ```bash
    set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin;%PATH%
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    set FORCE_CMAKE=1
    set CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CXX_FLAGS=/utf-8 -DCMAKE_C_FLAGS=/utf-8"
    set CMAKE_BUILD_PARALLEL_LEVEL=16
    ```

  * Install the necessary Python packages (this process may take some time):

    ```bash
    pip install ninja cmake scikit-build-core[pyproject]
    pip install --force-reinstall --no-cache-dir llama-cpp-python
    pip install -r requirements.txt
    ```

* **For Linux (with CUDA support):**

  ##### Install pre-build wheel for Linux

  * It is possible to install a pre-built wheel with CUDA support.
    * Source URL: [https://abetlen.github.io/llama-cpp-python/whl/cu121/llama-cpp-python/](https://abetlen.github.io/llama-cpp-python/whl/cu121/llama-cpp-python/)

    ```bash
    wget https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu121/llama_cpp_python-0.3.4-cp310-cp310-linux_x86_64.whl
    pip install llama_cpp_python-0.3.4-cp310-cp310-linux_x86_64.whl
    pip install -r requirements.txt
    ```

  ##### (Optional) Build with CUDA for Linux

  * Configure the required environment variables for the build (if not already set):

    ```bash
    export PATH=/usr/local/cuda/bin:${PATH}
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
    ```

  * Install the OpenMP libraries used for the build:

    ```bash
    sudo apt-get update
    sudo apt-get install libgomp1 libomp-dev
    ```

  * Install the necessary Python packages:

    ```bash
    conda install -c conda-forge libstdcxx-ng
    pip install ninja cmake scikit-build-core[pyproject]
    export CMAKE_ARGS="-DGGML_CUDA=ON"
    pip install --force-reinstall --no-cache-dir llama-cpp-python
    pip install -r requirements.txt
    ```

* **For Mac OS (without CUDA support):**
  * Install the necessary Python packages:

    ```bash
    BUILD_CUDA_EXT=0 pip install -r requirements.txt
    ```

  * Rebuild the `bitsandbytes` package with the CPU option.

    ```bash
    pip uninstall bitsandbytes
    git clone https://github.com/TimDettmers/bitsandbytes.git
    cd bitsandbytes
    cmake -DCOMPUTE_BACKEND=cpu -S .
    make
    pip install .
    ```

  * Install CMake and set the compiler:
  
    ```bash
    brew install cmake
    export CC=/usr/bin/gcc
    export CXX=/usr/bin/g++
    ```

  * Install `llama-cpp-python` with Metal support:

    ```bash
    export CMAKE_ARGS="-DLLAMA_METAL=on"
    export FORCE_CMAKE=1
    pip install -U llama-cpp-python --no-cache-dir
    ```

  * Known Issue: Running the LLaVA model on Mac results in an error.

## Running the application

```bash
python ollm_app.py
```

* Open <http://127.0.0.1:7860/> in your browser.

## Downloading the Model

To download the model:

* Launch this application.
* Click on the "Download model" button next to the LLM model ID.
* Wait for the download to complete.

### 📜 Model List (transformers)

| Provider      | Model Names                                                                                |
|---------------|--------------------------------------------------------------------------------------------|
| Microsoft     | Phi-3-mini-4k-instruct                                                                     |
| Google        | gemma-2-9b-it, gemma-1.1-2b-it, gemma-1.1-7b-it                                            |
| NVIDIA        | Llama3-ChatQA-1.5-8B                                                                       |
| Qwen          | Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct                                                 |
| Mistral AI    | Mistral-7B-Instruct-v0.3                                                                   |
| Rakuten       | RakutenAI-7B-chat, RakutenAI-7B-instruct                                                   |
| rinna         | youri-7b-chat                                                                              |
| TheBloke      | Llama-2-7b-Chat-GPTQ, Kunoichi-7B-GPTQ                                                     |

* 📋 Note: By adding the repository paths of models to `model_manager/add_tfs_models.txt`, they will be included in the list of Model IDs and displayed in the UI.
* 🔍 Note: The downloaded model file will be stored in the `.cache/huggingface/hub` directory of your home directory.

#### Access and Download Gemma and Llama Models

* Before downloading any models, ensure that you have obtained the necessary access rights through Hugging Face. Please visit the following pages to request access:
  * [Llama 3 model by Meta](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
  * [Llama 2 model by Meta](https://huggingface.co/meta-llama/Llama-2-7b-hf)
  * [Gemma 1.1 model by Google](https://huggingface.co/google/gemma-1.1-2b-it)

#### Login to Hugging Face

* Before downloading any models, please log in via the command line using:

  ```bash
  huggingface-cli login
  ```

### 🦙 Model List (llama.cpp)

| Provider      | Model Names                                                                                |
|---------------|--------------------------------------------------------------------------------------------|
| Microsoft     | Phi-3-mini-4k-instruct-q4.gguf, Phi-3-mini-4k-instruct-fp16.gguf                           |
| TheBloke      | llama-2-7b-chat.Q4_K_M.gguf                                                                |
| QuantFactory  | Meta-Llama-3-8B-Instruct.Q4_K_M.gguf                                                       |

#### Using any GGUF file

* 🔍 File Placement: Place files with the `.gguf` extension in the `models` directory within the `open-llm-webui` folder. These files will then appear in the model list on the `llama.cpp` tab of the web UI and can be used accordingly.
* 📝 Metadata Usage: If the metadata of a GGUF model includes `tokenizer.chat_template`, this template will be used to create the prompts.

### 🖼️ Model List (Multimodal LLaVA)

| Provider      | Model Names                                                                                |
|---------------|--------------------------------------------------------------------------------------------|
| Google        | google/gemma-3-4b-it, google/paligemma2-3b-pt-224, google/paligemma2-3b-pt-448             |
| Microsoft     | Phi-3.5-vision-instruct, Phi-3-vision-128k-instruct                                        |
| Meta          | Llama-3.2-11B-Vision (limited support as a trial)                                          |
| llava-hf      | llava-v1.6-mistral-7b-hf, llava-v1.6-vicuna-7b-hf, llava-1.5-7b-hf                         |
| tinyllava     | TinyLLaVA-Phi-2-SigLIP-3.1B                                                                |
| openbmb       | MiniCPM-V-2_6-int4, MiniCPM-V-2_6, MiniCPM-Llama3-V-2_5-int4, MiniCPM-Llama3-V-2_5         |
| SakanaAI      | EvoVLM-JP-v1-7B                                                                            |

#### Access and Download Llama 3 Models

* Before downloading any models, ensure that you have obtained the necessary access rights through Hugging Face. Please visit the following pages to request access:
  * [Llama 3.2 Vision model by Meta](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)
  * [Llama 3 model by Meta](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

## Usage

* Enter your message into the "Input text" box. Adjust the slider for "Max new tokens" as needed.
* Under "Advanced options" adjust the settings for "Temperature", "Top k", "Top p", and "Repetition Penalty" as needed.
* If replacing the system message of the prompt, under "Advanced options" enable the checkbox and enter text.
* Press "Enter" on your keyboard or click the "Generate" button.
  * ⚠️ Note: If the cloud-based model has been updated, it may be downloaded upon execution.
* If you click the "Clear chat" button, the chat history will be cleared.

### transformers tab

* By enabling the `CPU execution` checkbox, the model will use the argument `device_map="cpu"`.
* Some of the transformers models are loaded with the following 4-bit or 8-bit settings using the `bitsandbytes` package.

### llama.cpp tab

* Use the radio buttons in the `Default chat template` to select the template that will be used if the GGUF model lacks a `chat_template`.

### LLaVA tab

* You can upload an image to the LLaVA Image area of this tab and input a prompt related to the image.
* Some of the LLaVA models are loaded with the following 4-bit or 8-bit settings using the `bitsandbytes` package.

### Continuous Processing of Multiple Prompts

* Enter `input_prompts.json` in the `Input text` textbox.
* Ensure the `input_prompts.json` file in the current folder contains an array of objects with the key `"prompt"`.

* An example of the `input_prompts.json` file structure is as follows:

  ```json
  [
    {
      "prompt": "What is your name?"
    },
    {
      "prompt": "How are you?"
    }
  ]
  ```

### options

* When you enable the `Translate (ja->en/en->ja)` checkbox:
  * Any input in Japanese will be automatically translated to English, and responses in English will be automatically translated back into Japanese.
  * ⚠️ Note: Downloading the translation model for the first time may take some time.

![UI image](images/open-ollm-webui_ui_image_1.png)

## Model Credit

| Developer           | Model                        | License                                                        |
|---------------------|------------------------------|----------------------------------------------------------------|
| Meta                | Llama-3.2                    | [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) |
| Meta                | Llama-3.1                    | [Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE) |
| Meta                | Llama-3                      | [Llama 3 Community License](https://github.com/meta-llama/llama3/blob/main/LICENSE) |
| Meta                | Llama-2                      | [Llama 2 Community License](https://github.com/facebookresearch/llama/blob/main/LICENSE) |
| Microsoft           | Phi-3.5, Phi-3               | [The MIT License](https://opensource.org/licenses/MIT)         |
| Google              | Gemma                        | [Gemma Terms of Use](https://ai.google.dev/gemma/terms)        |
| NVIDIA              | Llama3-ChatQA                | [Llama 3 Community License](https://github.com/meta-llama/llama3/blob/main/LICENSE) |
| Alibaba Group       | Qwen2.5-3B-Instruct          | [Qwen RESEARCH LICENSE](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE) |
| Alibaba Group       | Qwen2.5-7B-Instruct          | [Apache License 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| Mistral AI          | Mistral-7B-Instruct          | [Apache License 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| Rakuten             | RakutenAI                    | [Apache License 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| rinna               | Youri                        | [Llama 2 Community License](https://ai.meta.com/llama/license/) |
| Sanji Watsuki       | Kunoichi-7B                  | [CC-BY-NC-4.0](https://spdx.org/licenses/CC-BY-NC-4.0)         |
| Hugging Face        | llava-v1.6-mistral-7b-hf     | [Apache License 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| Hugging Face        | llava-v1.6-vicuna-7b-hf, llava-1.5-7b-hf | [Llama 2 Community License](https://ai.meta.com/llama/license/) |
| TinyLLaVA           | TinyLLaVA-Phi-2-SigLIP-3.1B  | [Apache License 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| OpenBMB             | MiniCPM                      | [MiniCPM Model License](https://github.com/OpenBMB/MiniCPM/blob/main/MiniCPM%20Model%20License.md) |
| Sakana AI           | EvoVLM-JP-v1-7B              | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
