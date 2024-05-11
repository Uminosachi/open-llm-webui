import os
import platform

import torch
from torch.hub import download_url_to_file

from cache_manager import clear_cache_decorator
from custom_logging import ollm_logging
from model_manager import LLMConfig
from registry import get_cpp_llm_class

cpp_download_model_map = {
    "Phi-3-mini-4k-instruct-q4.gguf": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
    "Phi-3-mini-4k-instruct-fp16.gguf": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-fp16.gguf",
}

cpp_models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
if not os.path.isdir(cpp_models_dir):
    os.makedirs(cpp_models_dir, exist_ok=True)


class CPPLLMConfig(LLMConfig):
    pass


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
        save_path = os.path.join(cpp_models_dir, cpp_ollm_model_id)
        if not local_files_only and not os.path.isfile(save_path):
            try:
                download_url = cpp_download_model_map.get(cpp_ollm_model_id, None)
                if download_url is None:
                    raise FileNotFoundError(f"Model {cpp_ollm_model_id} not found.")
                ollm_logging.info(f"Downloading {cpp_ollm_model_id} to {save_path}")
                download_url_to_file(download_url, save_path)
            except FileNotFoundError:
                return "Model not found."
            except Exception as e:
                return str(e)
        else:
            if not os.path.isfile(save_path):
                ollm_logging.debug(save_path)
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

        llm.cpu_execution(cpu_execution_chk)

        if platform.system() == "Darwin":
            llm.model_kwargs.update(dict(torch_dtype=torch.float32))

        ollm_logging.info(f"model_kwargs: {llm.model_kwargs}")

        return llm


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
