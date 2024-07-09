import importlib.util
import platform

import torch
from huggingface_hub import snapshot_download
from transformers import (AutoModel, AutoModelForCausalLM, AutoProcessor,  # noqa: F401
                          AutoTokenizer, BitsAndBytesConfig, LlavaForConditionalGeneration,
                          LlavaNextForConditionalGeneration, LlavaNextProcessor,
                          SiglipImageProcessor)

from cache_manager import clear_cache_decorator
from custom_logging import ollm_logging
from registry import get_llm_class, register_model

from .base import BaseAbstractLLM, LLMConfig, ensure_tensor_dtype, replace_br_and_code
from .minicpm.modeling_minicpmv import MiniCPMV, PreTrainedTokenizerFastWrapper


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
            model_generate_name="generate",
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
            model_generate_name="generate",
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


@register_model("llava-llama-3")
class LlavaLlama3Model(LLMConfig):
    include_name: str = "llava-llama-3"

    prompt_template = ("<|start_header_id|>user<|end_header_id|>\n\n<image>\n{prompt}<|eot_id|>"
                       "<|start_header_id|>assistant<|end_header_id|>\n\n")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    def __init__(self):
        super().__init__(
            model_class=LlavaForConditionalGeneration,
            tokenizer_class=AutoProcessor,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=self.quantization_config,
                offload_buffers=True,
            ),
            model_generate_name="generate",
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


@register_model("minicpm-llama3")
class MiniCPMLlama3Model(LLMConfig):
    include_name: str = "MiniCPM-Llama3"

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

    def __init__(self):
        model_kwargs = dict(
            device_map=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            offload_buffers=True,
        )
        if hasattr(self, "model_id") and "int4" not in self.model_id:
            model_kwargs.update(dict(quantization_config=self.quantization_8bit_config))

        super().__init__(
            model_class=MiniCPMV,
            tokenizer_class=PreTrainedTokenizerFastWrapper,
            model_kwargs=model_kwargs,
            model_generate_name="generate",
            tokenizer_kwargs=dict(
            ),
            tokenizer_input_kwargs=dict(
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
            multimodal_image=True,
            require_tokenization=False,
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        prompt = {"role": "user", "content": input_text_box}
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        inputs.update(generate_params)
        inputs["return_vision_hidden_states"] = False
        return inputs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        generated_text = "".join(output_text)
        return generated_text


@register_model("llava-1.5")
class Llava15Model(LLMConfig):
    include_name: str = "llava-1.5"

    prompt_template = "USER: <image>\n{prompt}\nASSISTANT: "
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    def __init__(self):
        super().__init__(
            model_class=LlavaForConditionalGeneration,
            tokenizer_class=AutoProcessor,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=self.quantization_config,
                offload_buffers=True,
            ),
            model_generate_name="generate",
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


@register_model("llava-calm2")
class LlavaCALM2Model(LLMConfig):
    include_name: str = "llava-calm2"

    prompt_template = "USER: <image>\n{prompt}\nASSISTANT: "
    torch_dtype = torch.bfloat16

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

    def __init__(self):
        super().__init__(
            model_class=LlavaForConditionalGeneration,
            tokenizer_class=AutoProcessor,
            model_kwargs=dict(
                device_map=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                quantization_config=self.quantization_4bit_config,
                offload_buffers=True,
            ),
            model_generate_name="generate",
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
        if "pixel_values" in generate_kwargs:
            generate_kwargs["pixel_values"] = ensure_tensor_dtype(generate_kwargs["pixel_values"], self.torch_dtype)
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


@register_model("tinyllava")
class TinyLLaVAModel(LLMConfig):
    include_name: str = "TinyLLaVA"

    prompt_template = ("User: <image>\n{prompt} Assistant:")
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    def __init__(self):
        super().__init__(
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            image_processor_class=SiglipImageProcessor,
            model_kwargs=dict(
                device_map=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                torch_dtype=torch.float16,
                # quantization_config=self.quantization_config,
                offload_buffers=True,
                trust_remote_code=True,
            ),
            model_generate_name="generate",
            tokenizer_kwargs=dict(
                use_fast=False,
            ),
            image_processor_kwargs=dict(
                pretrained_model_name_or_path="google/siglip-so400m-patch14-384",
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
            ),
            image_processor_input_kwargs=dict(
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


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


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
            if "quantization_config" in llm.model_kwargs:
                llm.model_kwargs.pop("quantization_config")

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
        if model_params.image_processor_class is None:
            tokenizer = model_params.tokenizer_class.from_pretrained(
                ollm_model_id if pmnop not in model_params.tokenizer_kwargs else model_params.tokenizer_kwargs.pop(pmnop),
                **model_params.tokenizer_kwargs,
            )
            return tokenizer
        else:
            tokenizer = model_params.tokenizer_class.from_pretrained(
                ollm_model_id if pmnop not in model_params.tokenizer_kwargs else model_params.tokenizer_kwargs.pop(pmnop),
                **model_params.tokenizer_kwargs,
            )
            image_processor = model_params.image_processor_class.from_pretrained(
                ollm_model_id if pmnop not in model_params.image_processor_kwargs else model_params.image_processor_kwargs.pop(pmnop),
                **model_params.image_processor_kwargs,
            )
            tokenizer.image_processor = image_processor
            return tokenizer

    @clear_cache_decorator
    @staticmethod
    def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == "pt":
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f"Unsupported tensor type: {return_tensors}")
        return input_ids


def get_llava_ollm_model_ids():
    """Get list of model IDs for the LLaVA models.

    Returns:
        list: List of model IDs for the LLaVA models.
    """
    llava_ollm_model_ids = [
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "llava-hf/llava-v1.6-vicuna-7b-hf",
        "llava-hf/llava-1.5-7b-hf",
        "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B",
        "openbmb/MiniCPM-Llama3-V-2_5-int4",
        "openbmb/MiniCPM-Llama3-V-2_5",
        "xtuner/llava-llama-3-8b-v1_1-transformers",
        "cyberagent/llava-calm2-siglip",
    ]

    return llava_ollm_model_ids
