import importlib.util
from abc import ABC, abstractmethod
from collections import UserDict
from dataclasses import dataclass, field
from functools import wraps
from importlib.metadata import version as implib_version

import numpy as np
import torch
from packaging.version import parse
from transformers import BitsAndBytesConfig
from transformers.utils import ModelOutput

from chat_utils import convert_code_tags_to_md
from custom_logging import ollm_logging


def compare_versions(version1: str, version2: str) -> int:
    v1 = parse(version1)
    v2 = parse(version2)
    if v1 > v2:
        return 1
    elif v1 == v2:
        return 0
    else:
        return -1


def compare_package_version(package_name: str, version: str) -> int:
    try:
        installed_version = implib_version(package_name)
        return compare_versions(installed_version, version)
    except Exception:
        ollm_logging.warning(f"Failed to get version of {package_name}")
        return -1


def replace_br_and_code(func):
    @wraps(func)
    def wrapper(self, chatbot, *args, **kwargs):
        chatbot = [[convert_code_tags_to_md(item.replace("<br>", "\n")) for item in sublist] for sublist in chatbot]
        return func(self, chatbot, *args, **kwargs)
    return wrapper


def print_return(name):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            ret = func(self, *args, **kwargs)
            ollm_logging.info(f"{name}: {ret}")
            return ret
        return wrapper
    return decorator


def ensure_tensor_on_device(inputs, device):
    if isinstance(inputs, (ModelOutput, dict, UserDict)):
        return type(inputs)({name: ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()})
    elif isinstance(inputs, list):
        return [ensure_tensor_on_device(item, device) for item in inputs]
    elif isinstance(inputs, tuple):
        if hasattr(inputs, "_fields"):  # Check for NamedTuple
            return type(inputs)(*(ensure_tensor_on_device(item, device) for item in inputs))
        return tuple(ensure_tensor_on_device(item, device) for item in inputs)
    elif isinstance(inputs, set):
        return {ensure_tensor_on_device(item, device) for item in inputs}
    elif isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    elif isinstance(inputs, np.ndarray):
        return torch.from_numpy(inputs).to(device)
    else:
        return inputs


def ensure_tensor_dtype(inputs, torch_dtype):
    if isinstance(inputs, (ModelOutput, dict, UserDict)):
        return type(inputs)({name: ensure_tensor_dtype(tensor, torch_dtype) for name, tensor in inputs.items()})
    elif isinstance(inputs, list):
        return [ensure_tensor_dtype(item, torch_dtype) for item in inputs]
    elif isinstance(inputs, tuple):
        if hasattr(inputs, "_fields"):  # Check for NamedTuple
            return type(inputs)(*(ensure_tensor_dtype(item, torch_dtype) for item in inputs))
        return tuple(ensure_tensor_dtype(item, torch_dtype) for item in inputs)
    elif isinstance(inputs, set):
        return {ensure_tensor_dtype(item, torch_dtype) for item in inputs}
    elif isinstance(inputs, torch.Tensor):
        return inputs.to(torch_dtype)
    elif isinstance(inputs, np.ndarray):
        return torch.from_numpy(inputs).to(torch_dtype)
    else:
        return inputs


def check_package_installed(package_name):
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None


@dataclass
class LLMConfig(ABC):
    model_class: object
    tokenizer_class: object
    image_processor_class: object = None
    model_kwargs: dict = field(default_factory=dict)
    model_generate_name: str = "generate"
    tokenizer_kwargs: dict = field(default_factory=dict)
    image_processor_kwargs: dict = field(default_factory=dict)
    tokenizer_input_kwargs: dict = field(default_factory=dict)
    image_processor_input_kwargs: dict = field(default_factory=dict)
    tokenizer_decode_kwargs: dict = field(default_factory=dict)
    output_text_only: bool = True
    require_tokenization: bool = True
    multimodal_image: bool = False
    imagep_config: dict = field(
        default_factory=lambda: {"prompt_is_list": True, "image_is_list": False, "image_is_first": False})

    enable_rag_text = False
    DOWNLOAD_COMPLETE = "Download complete"

    quantization_4bit_config = BitsAndBytesConfig(**{
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_storage": "uint8",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "llm_int8_enable_fp32_cpu_offload": False,
        "llm_int8_has_fp16_weight": False,
        "llm_int8_skip_modules": [
            "out_proj",
            "kv_proj",
            "lm_head"
        ],
        "llm_int8_threshold": 6.0,
        "load_in_4bit": True,
        "load_in_8bit": False,
    })

    quantization_8bit_config = BitsAndBytesConfig(**{
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_storage": "uint8",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "llm_int8_enable_fp32_cpu_offload": False,
        "llm_int8_has_fp16_weight": False,
        "llm_int8_skip_modules": [
            "out_proj",
            "kv_proj",
            "lm_head"
        ],
        "llm_int8_threshold": 6.0,
        "load_in_4bit": False,
        "load_in_8bit": True,
    })

    def cpu_execution(self, cpu_execution_chk=False):
        if cpu_execution_chk:
            update_dict = dict(device_map="cpu", torch_dtype=torch.float32)
            self.model_kwargs.update(update_dict)

    @abstractmethod
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        pass

    def create_chat_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None,
                           check_assistant=False, remove_bos_token=False):
        if not hasattr(tokenizer, "pad_token_id") and hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer

        if getattr(self, "system_message", None) is not None:
            messages = [{"role": "system", "content": self.system_message}]
        else:
            messages = []
        len_chat = len(chatbot)
        for i, (user_text, assistant_text) in enumerate(chatbot):
            messages.append({"role": "user", "content": user_text})
            if not check_assistant:
                messages.append({"role": "assistant", "content": assistant_text})
            elif i < (len_chat - 1) or len(assistant_text) > 0:
                messages.append({"role": "assistant", "content": assistant_text})
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            ollm_logging.warning("Failed to apply chat template. Removing system message.")
            messages = [message for message in messages if message["role"] != "system"]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        if remove_bos_token:
            if getattr(tokenizer, "bos_token", None) is not None and prompt.startswith(tokenizer.bos_token):
                ollm_logging.debug("Removing bos_token from prompt")
                prompt = prompt.replace(tokenizer.bos_token, "", 1)
        return prompt

    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        if not hasattr(tokenizer, "pad_token_id") and hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer

        generate_kwargs = dict(
            **inputs,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generate_kwargs.update(generate_params)
        return generate_kwargs

    @abstractmethod
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        pass

    def is_ampere_or_newer(self):
        if not torch.cuda.is_available():
            # raise RuntimeError("CUDA is not available")
            return False

        major, minor = torch.cuda.get_device_capability()
        ollm_logging.info(f"CUDA compute capability: {major}.{minor}")

        # Ampere GPUs have a compute capability of 8.0 or higher
        return major >= 8


class BaseAbstractLLM(ABC):
    @staticmethod
    @abstractmethod
    def download_model(ollm_model_id, local_files_only=False):
        pass

    @staticmethod
    @abstractmethod
    def get_llm_instance(ollm_model_id, cpu_execution_chk=False):
        pass

    @staticmethod
    @abstractmethod
    def get_model(ollm_model_id, model_params, generate_params):
        pass

    @staticmethod
    @abstractmethod
    def get_tokenizer(ollm_model_id, model_params):
        pass
