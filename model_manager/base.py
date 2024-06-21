from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps

import torch

from chat_utils import convert_code_tags_to_md
from custom_logging import ollm_logging


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

    enable_rag_text = False
    DOWNLOAD_COMPLETE = "Download complete"

    def cpu_execution(self, cpu_execution_chk=False):
        if cpu_execution_chk:
            update_dict = dict(device_map="cpu", torch_dtype=torch.float32)
            self.model_kwargs.update(update_dict)

    @abstractmethod
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        pass

    def create_chat_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None,
                           check_assistant=False, remove_bos_token=False):
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
