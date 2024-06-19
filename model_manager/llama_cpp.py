import os
import platform  # noqa: F401
from typing import Any

import torch  # noqa: F401
from llama_cpp import Llama
from torch.hub import download_url_to_file
from transformers import AutoTokenizer

from cache_manager import clear_cache_decorator
from custom_logging import ollm_logging
from model_manager.base import BaseAbstractLLM, LLMConfig, replace_br_and_code
from registry import get_cpp_llm_class, register_cpp_model
from start_messages import llama2_message  # noqa: F401

cpp_download_model_list = [
    ("Phi-3-mini-4k-instruct-q4.gguf",
     "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"),
    ("Phi-3-mini-4k-instruct-fp16.gguf",
     "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-fp16.gguf"),
    ("llama-2-7b-chat.Q4_K_M.gguf",
     "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"),
    ("Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
     "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"),
]

cpp_download_model_map = dict(cpp_download_model_list)

cpp_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "models")
if not os.path.isdir(cpp_models_dir):
    os.makedirs(cpp_models_dir, exist_ok=True)


def get_gguf_file_path(cpp_ollm_model_id):
    gguf_file_path = os.path.join(cpp_models_dir, cpp_ollm_model_id)
    return gguf_file_path


def list_files(directory: str, extension: str) -> list:
    """List all files in the specified directory with the given extension.

    Args:
        directory (str): The path to the directory.
        extension (str): The file extension to filter by, including the dot (e.g., '.txt').

    Returns:
        list: A list of filenames that match the given extension.
    """
    # Ensure the extension starts with a dot
    if not extension.startswith("."):
        extension = "." + extension
    files = []
    for file in os.listdir(directory):
        full_path = os.path.join(directory, file)
        if os.path.isfile(full_path) and file.lower().endswith(extension.lower()):
            files.append(file)
    return files


class CPPChatTemplates:
    llama2_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '[ASST] '  + message['content'] + ' [/ASST]' + eos_token }}"
        "{% endif %}{% endfor %}")

    llama3_template = (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}"
        "{% set content = bos_token + content %}"
        "{% endif %}"
        "{{ content }}"
        "{% endfor %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}")

    gemma_template = (
        "{{ bos_token }}"
        "{% if messages[0]['role'] == 'system' %}"
        "{{ raise_exception('System role not supported') }}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
        "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
        "{% endif %}"
        "{% if (message['role'] == 'assistant') %}"
        "{% set role = 'model' %}"
        "{% else %}"
        "{% set role = message['role'] %}"
        "{% endif %}"
        "{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<start_of_turn>model\\n' }}"
        "{% endif %}")

    phi3_template = (
        "{{ bos_token }}{% for message in messages %}"
        "{% if (message['role'] == 'user') %}"
        "{{'<|user|>' + '\\n' + message['content'] + '<|end|>' + '\\n' + '<|assistant|>' + '\\n'}}"
        "{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '\\n'}}"
        "{% endif %}{% endfor %}")

    mixtral_template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
        "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
        "{% endif %}{% if message['role'] == 'user' %}"
        "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + eos_token}}"
        "{% else %}"
        "{{ raise_exception('Only user and assistant roles are supported!') }}"
        "{% endif %}"
        "{% endfor %}")

    zephyr_template = (
        "{% for message in messages %}\n"
        "{% if message['role'] == 'user' %}\n"
        "{{ '<|user|>\\n' + message['content'] + eos_token }}\n"
        "{% elif message['role'] == 'system' %}\n"
        "{{ '<|system|>\\n' + message['content'] + eos_token }}\n"
        "{% elif message['role'] == 'assistant' %}\n"
        "{{ '<|assistant|>\\n'  + message['content'] + eos_token }}\n"
        "{% endif %}\n"
        "{% if loop.last and add_generation_prompt %}\n"
        "{{ '<|assistant|>' }}\n"
        "{% endif %}\n"
        "{% endfor %}")

    qwen_template = (
        "{% for message in messages %}"
        "{% if loop.first and messages[0]['role'] != 'system' %}"
        "{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
        "{% endif %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\\n' }}"
        "{% endif %}")

    chat_templates_map = {
        "Llama2": [llama2_template, "You are a helpful assistant."],
        "Llama3": [llama3_template, "You are a helpful assistant."],
        "Gemma":  [gemma_template, None],
        "Phi-3":  [phi3_template, None],
        "Mixtral": [mixtral_template, None],
        "Qwen": [qwen_template, "You are a helpful assistant."],
        "Zephyr": [zephyr_template, "You are a helpful assistant."],
    }

    @clear_cache_decorator
    def prepare_tokenizer(self, tokenizer: Any, model: Any, cpp_chat_template: str) -> Any:
        def set_tokenizer_token(metadata_key: str, tokenizer_attr: str, log_message: str):
            token_id = model.metadata.get(metadata_key)
            if token_id is not None:
                token_id = int(token_id)
                token = model._model.token_get_text(token_id)
                setattr(tokenizer, tokenizer_attr, token)
                ollm_logging.info(f"Setting {tokenizer_attr}: {token} - {log_message}")

        if hasattr(model, "metadata") and isinstance(model.metadata, dict):
            if model.metadata.get("tokenizer.chat_template") is not None:
                ollm_logging.info("Using chat template from model metadata")
                tokenizer.chat_template = model.metadata["tokenizer.chat_template"]
            else:
                ollm_logging.info(f"Using {cpp_chat_template} chat template because model is missing template")
                tokenizer.chat_template = self.chat_templates_map[cpp_chat_template][0]

                if self.chat_templates_map[cpp_chat_template][1] is not None:
                    self.__class__.system_message = self.chat_templates_map[cpp_chat_template][1]
                elif hasattr(self.__class__, "system_message"):
                    del self.__class__.system_message

            if hasattr(model, "_model") and hasattr(model._model, "token_get_text"):
                set_tokenizer_token("tokenizer.ggml.bos_token_id", "bos_token", "beginning of sentence")
                set_tokenizer_token("tokenizer.ggml.eos_token_id", "eos_token", "end of sentence")

        return tokenizer


def get_chat_templates_keys():
    return list(CPPChatTemplates.chat_templates_map.keys())


@register_cpp_model("default")
class CPPDefaultModel(LLMConfig, CPPChatTemplates):
    include_name: str = "default"

    system_message = "You are a helpful assistant."

    def __init__(self):
        super().__init__(
            model_class=Llama,
            tokenizer_class=AutoTokenizer,
            model_kwargs=dict(
                n_ctx=4096,
                n_threads=8,
                n_gpu_layers=-1,
            ),
            model_generate_name="create_completion",
            tokenizer_kwargs=dict(
                pretrained_model_name_or_path="microsoft/Phi-3-mini-4k-instruct",
                use_fast=True,
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
                add_special_tokens=False,
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
            require_tokenization=False,
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer,
                                         check_assistant=True, remove_bos_token=True)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = dict(
            prompt=inputs,
            max_tokens=generate_params["max_new_tokens"],
            temperature=generate_params["temperature"],
            top_k=generate_params["top_k"],
            top_p=generate_params["top_p"],
            repeat_penalty=generate_params["repetition_penalty"],
        )

        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        ollm_logging.debug(f"output_text: {output_text}")
        return output_text["choices"][0]["text"]


@register_cpp_model("phi-3")
class CPPPHI3Model(LLMConfig, CPPChatTemplates):
    include_name: str = "Phi-3"

    system_message = "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user."

    def __init__(self):
        super().__init__(
            model_class=Llama,
            tokenizer_class=AutoTokenizer,
            model_kwargs=dict(
                n_ctx=4096,
                n_threads=8,
                n_gpu_layers=35,
            ),
            model_generate_name="create_completion",
            tokenizer_kwargs=dict(
                pretrained_model_name_or_path="microsoft/Phi-3-mini-4k-instruct",
                use_fast=True,
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
                add_special_tokens=False,
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
            require_tokenization=False,
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer,
                                         check_assistant=True, remove_bos_token=True)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = dict(
            prompt=inputs,
            max_tokens=generate_params["max_new_tokens"],
            temperature=generate_params["temperature"],
            top_k=generate_params["top_k"],
            top_p=generate_params["top_p"],
            repeat_penalty=generate_params["repetition_penalty"],
        )

        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        ollm_logging.debug(f"output_text: {output_text}")
        return output_text["choices"][0]["text"]


class LlamaCPPLLM(BaseAbstractLLM):
    @clear_cache_decorator
    @staticmethod
    def download_model(ollm_model_id, local_files_only=False):
        """Download Open LLM and Llama models.

        Args:
            ollm_model_id (str): String of Open LLM model ID.
            local_files_only (bool, optional): If True, use only local files. Defaults to False.

        Returns:
            str: string of download result.
        """
        gguf_file_path = get_gguf_file_path(ollm_model_id)
        if not local_files_only and not os.path.isfile(gguf_file_path):
            try:
                download_url = cpp_download_model_map.get(ollm_model_id)
                if download_url is None:
                    raise FileNotFoundError(f"Model {ollm_model_id} not found in the models list.")
                ollm_logging.info(f"Downloading {ollm_model_id} to {gguf_file_path}")
                download_url_to_file(download_url, gguf_file_path)
            except FileNotFoundError as e:
                return str(e)
            except Exception as e:
                return str(e)
        else:
            if not os.path.isfile(gguf_file_path):
                ollm_logging.debug(f"Model not found: {gguf_file_path}")
                return "Model not found. Please download the model first."

        return LLMConfig.DOWNLOAD_COMPLETE

    @clear_cache_decorator
    @staticmethod
    def get_llm_instance(ollm_model_id, cpu_execution_chk=False):
        """Get model and tokenizer class.

        Args:
            ollm_model_id (str): String of Open LLM model ID.
            cpu_execution_chk (bool, optional): If True, use CPU execution. Defaults to False.

        Returns:
            object: Open LLM model instance.
        """
        llm = get_cpp_llm_class(ollm_model_id)()

        ollm_logging.info(f"model_kwargs: {llm.model_kwargs}")

        return llm

    @clear_cache_decorator
    @staticmethod
    def get_model(ollm_model_id, model_params, generate_params):
        model_kwargs = model_params.model_kwargs
        model_kwargs["model_path"] = get_gguf_file_path(ollm_model_id)

        model = model_params.model_class(**model_kwargs)

        return model

    @clear_cache_decorator
    @staticmethod
    def get_tokenizer(ollm_model_id, model_params):
        tokenizer_kwargs = model_params.tokenizer_kwargs
        tokenizer = model_params.tokenizer_class.from_pretrained(**tokenizer_kwargs)

        return tokenizer


def get_cpp_ollm_model_ids():
    """Get list of model IDs for the llama.cpp models.

    Returns:
        list: List of model IDs for the llama.cpp models.
    """
    cpp_ollm_model_ids = [
        "Phi-3-mini-4k-instruct-q4.gguf",
        "Phi-3-mini-4k-instruct-fp16.gguf",
        "llama-2-7b-chat.Q4_K_M.gguf",
        "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        ]
    list_model_ids = list_files(cpp_models_dir, ".gguf")
    list_model_ids += list_files(cpp_models_dir, ".bin")
    for model_id in list_model_ids:
        if model_id not in cpp_ollm_model_ids:
            cpp_ollm_model_ids.append(model_id)
    return cpp_ollm_model_ids
