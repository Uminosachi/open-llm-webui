import os
import platform  # noqa: F401

import torch  # noqa: F401
from llama_cpp import Llama
from torch.hub import download_url_to_file
from transformers import AutoTokenizer

from cache_manager import clear_cache_decorator
from custom_logging import ollm_logging
from model_manager import LLMConfig, replace_br
from registry import get_cpp_llm_class, register_cpp_model

cpp_download_model_map = {
    "Phi-3-mini-4k-instruct-q4.gguf": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
    "Phi-3-mini-4k-instruct-fp16.gguf": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-fp16.gguf",
}

cpp_models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
if not os.path.isdir(cpp_models_dir):
    os.makedirs(cpp_models_dir, exist_ok=True)


def get_gguf_file_path(cpp_ollm_model_id):
    gguf_file_path = os.path.join(cpp_models_dir, cpp_ollm_model_id)
    return gguf_file_path


@register_cpp_model("phi-3")
class CPPPHI3Model(LLMConfig):
    include_name: str = "phi-3"

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

    @replace_br
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer, check_assistant=True)
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
        return output_text


class LlamaCPPLLM:
    @clear_cache_decorator
    @staticmethod
    def download_model(cpp_ollm_model_id, local_files_only=False):
        """Download Open LLM and Llama models.

        Args:
            ollm_model_id (str): String of Open LLM model ID.
            local_files_only (bool, optional): If True, use only local files. Defaults to False.

        Returns:
            str: string of download result.
        """
        gguf_file_path = get_gguf_file_path(cpp_ollm_model_id)
        if not local_files_only and not os.path.isfile(gguf_file_path):
            try:
                download_url = cpp_download_model_map.get(cpp_ollm_model_id, None)
                if download_url is None:
                    raise FileNotFoundError(f"Model {cpp_ollm_model_id} not found.")
                ollm_logging.info(f"Downloading {cpp_ollm_model_id} to {gguf_file_path}")
                download_url_to_file(download_url, gguf_file_path)
            except FileNotFoundError:
                return "Model not found."
            except Exception as e:
                return str(e)
        else:
            if not os.path.isfile(gguf_file_path):
                ollm_logging.debug(gguf_file_path)
                return "Model not found. Please download the model first."

        return LLMConfig.DOWNLOAD_COMPLETE

    @clear_cache_decorator
    @staticmethod
    def get_model_and_tokenizer_class(ollm_model_id, cpu_execution_chk=False):
        """Get model and tokenizer class.

        Args:
            ollm_model_id (str): String of Open LLM model ID.

        Returns:
            tuple(class, class, dict, dict): Tuple of model class, tokenizer class, model kwargs, and tokenizer kwargs.
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
    """Get Open LLM and Llama model IDs.

    Returns:
        list: List of Open LLM model IDs.
    """
    cpp_ollm_model_ids = [
        "Phi-3-mini-4k-instruct-q4.gguf",
        "Phi-3-mini-4k-instruct-fp16.gguf",
        ]
    return cpp_ollm_model_ids
