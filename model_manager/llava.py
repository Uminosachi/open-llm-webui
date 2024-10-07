import copy
import platform
import warnings

import torch
from huggingface_hub import snapshot_download
from transformers import (AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
                          BitsAndBytesConfig, LlavaForConditionalGeneration,
                          LlavaNextForConditionalGeneration, LlavaNextProcessor,
                          SiglipImageProcessor)

from cache_manager import clear_cache_decorator
from custom_logging import ollm_logging
from registry import get_llm_class, register_model

from .base import (BaseAbstractLLM, LLMConfig, check_package_installed, compare_package_version,
                   ensure_tensor_dtype, replace_br_and_code)
from .llama32vision.configuration_mllama import MllamaConfig
from .llama32vision.modeling_mllama import MllamaForConditionalGeneration
from .llama32vision.processing_mllama import MllamaProcessor
from .minicpm25.modeling_minicpmv import MiniCPMV, PreTrainedTokenizerFastWrapper
from .minicpm26.modeling_minicpmv import MiniCPMV as MiniCPMV26
from .minicpm26.tokenization_minicpmv_fast import MiniCPMVTokenizerFast as MiniCPMVTokenizerFast26
from .phi35vision.modeling_phi3_v import Phi3VForCausalLM
from .phi35vision.processing_phi3_v import Phi3VProcessor

if not check_package_installed("bitsandbytes"):
    raise ModuleNotFoundError("Please install the bitsandbytes package to use the load_in_4bit option.")


@register_model("llava-mistral")
class LlavaMistralModel(LLMConfig):
    include_name: str = "llava-*-mistral"

    prompt_template = "[INST] <image>\n{prompt} [/INST]"
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    def __init__(self):
        tokenizer_kwargs = {}
        if compare_package_version("transformers", "4.45.0") >= 0:
            tokenizer_kwargs.update(dict(patch_size=14, vision_feature_select_strategy="default"))

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
            tokenizer_kwargs=tokenizer_kwargs,
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
            multimodal_image=True,
            imagep_config=dict(prompt_is_list=True, image_is_list=False,
                               image_is_first=(compare_package_version("transformers", "4.45.0") >= 0)),
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
        tokenizer_kwargs = {}
        if compare_package_version("transformers", "4.45.0") >= 0:
            tokenizer_kwargs.update(dict(patch_size=14, vision_feature_select_strategy="default"))

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
            tokenizer_kwargs=tokenizer_kwargs,
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
            multimodal_image=True,
            imagep_config=dict(prompt_is_list=True, image_is_list=False,
                               image_is_first=(compare_package_version("transformers", "4.45.0") >= 0)),
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
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    quantization_config.bnb_4bit_quant_type = "nf4"

    def __init__(self):
        tokenizer_kwargs = {}
        if compare_package_version("transformers", "4.45.0") >= 0:
            tokenizer_kwargs.update(dict(patch_size=14, vision_feature_select_strategy="default"))

        super().__init__(
            model_class=LlavaForConditionalGeneration,
            tokenizer_class=AutoProcessor,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                quantization_config=self.quantization_config,
                offload_buffers=True,
            ),
            model_generate_name="generate",
            tokenizer_kwargs=tokenizer_kwargs,
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
            multimodal_image=True,
            imagep_config=dict(prompt_is_list=True, image_is_list=False,
                               image_is_first=(compare_package_version("transformers", "4.45.0") >= 0)),
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


@register_model("minicpm-v")
class MiniCPMVModel(LLMConfig):
    include_name: str = "MiniCPM-V"

    def __init__(self):
        custom_config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        custom_config.auto_map = {
            "AutoConfig": "model_manager.minicpm26.configuration_minicpm",
            "AutoModelForCausalLM": "model_manager.minicpm26.modeling_minicpmv"
        }
        model_kwargs = dict(
            device_map=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            offload_buffers=True,
            config=custom_config,
        )
        if hasattr(self, "model_id") and "int4" not in self.model_id:
            quantization_8bit_config = copy.deepcopy(self.quantization_8bit_config)
            quantization_8bit_config.bnb_4bit_compute_dtype = "bfloat16"
            model_kwargs.update(dict(quantization_config=self.quantization_8bit_config))

        super().__init__(
            model_class=MiniCPMV26,
            tokenizer_class=MiniCPMVTokenizerFast26,
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
        tokenizer_kwargs = {}
        if compare_package_version("transformers", "4.45.0") >= 0:
            tokenizer_kwargs.update(dict(patch_size=14, vision_feature_select_strategy="default"))

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
            tokenizer_kwargs=tokenizer_kwargs,
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
            multimodal_image=True,
            imagep_config=dict(prompt_is_list=True, image_is_list=False,
                               image_is_first=(compare_package_version("transformers", "4.45.0") >= 0)),
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


@register_model("llama-3.2-vision")
class Llama3VisionModel(LLMConfig):
    include_name: str = "Llama-3.2-*-Vision"

    download_kwargs = dict(ignore_patterns=["*.pth"])

    prompt_template = "<|image|><|begin_of_text|>{prompt}"

    layer_device = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens = 32

    def __init__(self):
        model_kwargs = dict(
            device_map="cpu",
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            offload_buffers=True,
        )
        tokenizer_kwargs = {}
        if compare_package_version("transformers", "4.45.0") >= 0:
            tokenizer_kwargs.update(dict(patch_size=14, vision_feature_select_strategy="default"))

        config = MllamaConfig.from_pretrained(self.model_id)
        config.layer_device = self.layer_device
        config.text_config.layer_device = self.layer_device
        config.text_config.max_new_tokens = self.max_new_tokens
        model_kwargs.update(dict(config=config))

        super().__init__(
            model_class=MllamaForConditionalGeneration,
            tokenizer_class=MllamaProcessor,
            model_kwargs=model_kwargs,
            model_generate_name="generate",
            tokenizer_kwargs=tokenizer_kwargs,
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
            multimodal_image=True,
            imagep_config=dict(prompt_is_list=False, image_is_list=False,
                               image_is_first=(compare_package_version("transformers", "4.45.0") >= 0)),
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        input_text = self.prompt_template.format(prompt=input_text_box)
        return input_text

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        generate_kwargs.pop("repetition_penalty")

        generate_kwargs["do_sample"] = False
        generate_kwargs.pop("temperature")
        generate_kwargs["max_new_tokens"] = self.max_new_tokens
        ollm_logging.info(f"`do_sample` set to False and `max_new_tokens` set to {self.max_new_tokens} for Llama-3.2 Vision model to save time.")

        return generate_kwargs

    @clear_cache_decorator
    def reset_model(self, model):
        if (hasattr(model, "language_model") and hasattr(model.language_model, "model")
                and hasattr(model.language_model.model, "progress_bar")):
            model.language_model.model.progress_bar.reset()

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


@register_model("phi-3-vision")
class Phi3VisionModel(LLMConfig):
    include_name: str = "Phi-3*-vision"

    image_header = "<|image_1|>\n"
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    def __init__(self):
        model_kwargs = dict(
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            quantization_config=self.quantization_config,
            _attn_implementation="flash_attention_2",
        )
        if not self.is_ampere_or_newer() or not check_package_installed("flash_attn"):
            model_kwargs.update(dict(_attn_implementation="eager"))

        if "Phi-3.5" in self.model_id:
            model_class = Phi3VForCausalLM
            model_kwargs.pop("trust_remote_code")
            tokenizer_class = Phi3VProcessor
            tokenizer_kwargs = dict(num_crops=4)
        else:
            model_class = AutoModelForCausalLM
            tokenizer_class = AutoProcessor
            tokenizer_kwargs = dict(trust_remote_code=True)
            warnings.filterwarnings(action="ignore", category=UserWarning, message="Phi-3-V modifies")

        super().__init__(
            model_class=model_class,
            tokenizer_class=tokenizer_class,
            model_kwargs=model_kwargs,
            model_generate_name="generate",
            tokenizer_kwargs=tokenizer_kwargs,
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
            multimodal_image=True,
            imagep_config=dict(prompt_is_list=False, image_is_list=True, image_is_first=False),
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        messages = [
            {"role": "user", "content": self.image_header + input_text_box},
        ]
        prompt = tokenizer.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        generate_kwargs.pop("repetition_penalty")
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


@register_model("llava-calm2")
class LlavaCALM2Model(LLMConfig):
    include_name: str = "llava-calm2"

    prompt_template = "USER: <image>\n{prompt}\nASSISTANT: "
    torch_dtype = torch.bfloat16

    def __init__(self):
        model_kwargs = dict(
            device_map=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            offload_buffers=True,
        )
        quantization_config = copy.deepcopy(self.quantization_4bit_config)
        quantization_config.llm_int8_skip_modules = ["o_proj", "lm_head", "out_proj", "head"]
        model_kwargs.update(dict(quantization_config=quantization_config))

        tokenizer_kwargs = {}
        # if compare_package_version("transformers", "4.45.0") >= 0:
        #     tokenizer_kwargs.update(dict(patch_size=14, vision_feature_select_strategy="default"))

        super().__init__(
            model_class=LlavaForConditionalGeneration,
            tokenizer_class=AutoProcessor,
            model_kwargs=model_kwargs,
            model_generate_name="generate",
            tokenizer_kwargs=tokenizer_kwargs,
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
            multimodal_image=True,
            imagep_config=dict(prompt_is_list=True, image_is_list=False,
                               image_is_first=(compare_package_version("transformers", "4.45.0") >= 0)),
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


@register_model("evovlm-jp")
class EvoVLMModel(LLMConfig):
    include_name: str = "EvoVLM-JP"

    prompt_template = """<s>[INST] <image>\n{prompt} [/INST]"""
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    def __init__(self):
        tokenizer_kwargs = {}
        if compare_package_version("transformers", "4.45.0") >= 0:
            tokenizer_kwargs.update(dict(patch_size=14, vision_feature_select_strategy="default"))

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
            tokenizer_kwargs=tokenizer_kwargs,
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
            multimodal_image=True,
            imagep_config=dict(prompt_is_list=True, image_is_list=False,
                               image_is_first=(compare_package_version("transformers", "4.45.0") >= 0)),
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
        "microsoft/Phi-3.5-vision-instruct",
        "microsoft/Phi-3-vision-128k-instruct",
        "meta-llama/Llama-3.2-11B-Vision",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "llava-hf/llava-v1.6-vicuna-7b-hf",
        "llava-hf/llava-1.5-7b-hf",
        "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B",
        "openbmb/MiniCPM-V-2_6-int4",
        "openbmb/MiniCPM-V-2_6",
        "openbmb/MiniCPM-Llama3-V-2_5-int4",
        "openbmb/MiniCPM-Llama3-V-2_5",
        "SakanaAI/EvoVLM-JP-v1-7B",
    ]

    return llava_ollm_model_ids
