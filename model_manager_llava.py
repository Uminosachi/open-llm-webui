import importlib.util
import platform

import torch
from huggingface_hub import snapshot_download
from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration, LlavaNextProcessor

from cache_manager import clear_cache_decorator
from custom_logging import ollm_logging
from model_manager import BaseAbstractLLM, LLMConfig, replace_br_and_code
from registry import get_llm_class, register_model


def check_package_installed(package_name):
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None


if not check_package_installed("bitsandbytes"):
    raise ModuleNotFoundError("Please install the bitsandbytes package to use the load_in_4bit option.")


@register_model("llava-mistral")
class LlavaMistralModel(LLMConfig):
    include_name: str = "llava-*-mistral"

    prompt_template = "[INST] <image>\n{prompt} [/INST]"
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    def __init__(self):
        super().__init__(
            model_class=LlavaNextForConditionalGeneration,
            tokenizer_class=LlavaNextProcessor,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=self.quantization_config,
                offload_buffers=True,
            ),
            tokenizer_kwargs=dict(
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
            multimodal_image=True,
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        prompt = self.prompt_template.format(prompt=input_text_box)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


@register_model("llava-vicuna")
class LlavaVicunaModel(LLMConfig):
    include_name: str = "llava-*-vicuna"

    prompt_template = ("A chat between a curious human and an artificial intelligence assistant. "
                       "The assistant gives helpful, detailed, and polite answers to the human's questions. "
                       "USER: <image>\n{prompt} ASSISTANT:")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    def __init__(self):
        super().__init__(
            model_class=LlavaNextForConditionalGeneration,
            tokenizer_class=LlavaNextProcessor,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=self.quantization_config,
                offload_buffers=True,
            ),
            tokenizer_kwargs=dict(
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
            multimodal_image=True,
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        prompt = self.prompt_template.format(prompt=input_text_box)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


class LlavaLLM(BaseAbstractLLM):
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
        if not local_files_only:
            ollm_logging.info(f"Downloading {ollm_model_id}")
        try:
            llm_class = get_llm_class(ollm_model_id)
            if hasattr(llm_class, "download_kwargs") and isinstance(llm_class.download_kwargs, dict):
                snapshot_download(repo_id=ollm_model_id, local_files_only=local_files_only, **llm_class.download_kwargs)
            else:
                snapshot_download(repo_id=ollm_model_id, local_files_only=local_files_only)
        except FileNotFoundError:
            return "Model not found. Please download the model first."
        except Exception as e:
            return str(e)

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
        llm = get_llm_class(ollm_model_id)()

        llm.cpu_execution(cpu_execution_chk)

        if platform.system() == "Darwin":
            llm.model_kwargs.update(dict(torch_dtype=torch.float32))

        ollm_logging.info(f"model_kwargs: {llm.model_kwargs}")

        return llm

    @clear_cache_decorator
    @staticmethod
    def get_model(ollm_model_id, model_params, generate_params):
        pmnop = "pretrained_model_name_or_path"
        if "quantize_config" in model_params.model_kwargs:
            model = model_params.model_class.from_quantized(
                ollm_model_id if pmnop not in model_params.model_kwargs else model_params.model_kwargs.pop(pmnop),
                **model_params.model_kwargs,
            )
        else:
            model = model_params.model_class.from_pretrained(
                ollm_model_id if pmnop not in model_params.model_kwargs else model_params.model_kwargs.pop(pmnop),
                **model_params.model_kwargs,
            )
        model.eval()
        model.tie_weights()

        return model

    @clear_cache_decorator
    @staticmethod
    def get_tokenizer(ollm_model_id, model_params):
        pmnop = "pretrained_model_name_or_path"
        tokenizer = model_params.tokenizer_class.from_pretrained(
            ollm_model_id if pmnop not in model_params.tokenizer_kwargs else model_params.tokenizer_kwargs.pop(pmnop),
            **model_params.tokenizer_kwargs,
        )

        return tokenizer


def get_llava_ollm_model_ids():
    """Get list of model IDs for the LLaVA models.

    Returns:
        list: List of model IDs for the LLaVA models.
    """
    llava_ollm_model_ids = [
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "llava-hf/llava-v1.6-vicuna-7b-hf",
    ]

    return llava_ollm_model_ids
