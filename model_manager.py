import platform
from dataclasses import dataclass, field

import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer,
                          StoppingCriteriaList, TextIteratorStreamer)

from cache_manager import clear_cache_decorator, model_cache
from registry import MODEL_REGISTRY, register_model
from start_messages import StopOnTokens


@dataclass
class LLMConfig:
    model_class: object
    tokenizer_class: object
    model_kwargs: dict = field(default_factory=dict)
    tokenizer_kwargs: dict = field(default_factory=dict)
    tokenizer_input_kwargs: dict = field(default_factory=dict)
    tokenizer_decode_kwargs: dict = field(default_factory=dict)

    def cpu_execution(self, cpu_execution_chk=False):
        if cpu_execution_chk:
            self.model_kwargs.update({"device_map": "cpu"})


@register_model("default")
class DefaultModel(LLMConfig):
    include_name: str = "default"

    def __init__(self):
        super().__init__(
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype="auto",
            ),
            tokenizer_kwargs=dict(
                use_fast=True,
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
                add_special_tokens=True,
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
        )


@register_model("open-calm")
class OpenCalmModel(LLMConfig):
    include_name: str = "open-calm"

    def __init__(self):
        super().__init__(
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype="auto",
            ),
            tokenizer_kwargs=dict(
                use_fast=True,
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
                add_special_tokens=True,
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
        )


@register_model("gpt-neox")
class GPTNeoXModel(LLMConfig):
    include_name: str = "gpt-neox"

    def __init__(self):
        super().__init__(
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype="auto",
            ),
            tokenizer_kwargs=dict(
                use_fast=False,
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
                add_special_tokens=False,
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
        )


@register_model("stablelm-tuned")
class StableLMTunedModel(LLMConfig):
    include_name: str = "stablelm-tuned"

    def __init__(self):
        super().__init__(
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype="auto",
            ),
            tokenizer_kwargs=dict(
                use_fast=True,
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
                add_special_tokens=True,
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
        )


@register_model("japanese-stablelm")
class JapaneseStableLMModel(LLMConfig):
    include_name: str = "japanese-stablelm"

    def __init__(self):
        super().__init__(
            model_class=AutoModelForCausalLM,
            tokenizer_class=LlamaTokenizer,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True,
            ),
            tokenizer_kwargs=dict(
                use_fast=True,
                pretrained_model_name_or_path="novelai/nerdstash-tokenizer-v1",
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
                add_special_tokens=False,
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=False,
            ),
        )


@register_model("llama")
class LlamaModel(LLMConfig):
    include_name: str = "llama-"

    def __init__(self):
        super().__init__(
            model_class=LlamaForCausalLM,
            tokenizer_class=LlamaTokenizer,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype="auto",
            ),
            tokenizer_kwargs=dict(
                use_fast=True,
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
                add_special_tokens=True,
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
        )


@register_model("gptq")
class GPTQModel(LLMConfig):
    include_name: str = "-gptq"

    def __init__(self):
        super().__init__(
            model_class=AutoGPTQForCausalLM,
            tokenizer_class=AutoTokenizer,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype=torch.float16,
                use_safetensors=True,
                trust_remote_code=True,
                use_triton=False,
                quantize_config=None,
            ),
            tokenizer_kwargs=dict(
                use_fast=True,
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
                add_special_tokens=True,
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
        )


def get_ollm_model_ids():
    """Get Open LLM and Llama model IDs.

    Returns:
        list: List of Open LLM model IDs.
    """
    ollm_model_ids = [
        "rinna/bilingual-gpt-neox-4b",
        "rinna/bilingual-gpt-neox-4b-instruction-sft",
        "rinna/japanese-gpt-neox-3.6b",
        "rinna/japanese-gpt-neox-3.6b-instruction-sft",
        "rinna/japanese-gpt-neox-3.6b-instruction-sft-v2",
        "rinna/japanese-gpt-neox-3.6b-instruction-ppo",
        "TheBloke/Llama-2-7b-Chat-GPTQ",
        "TheBloke/Llama-2-13B-chat-GPTQ",
        "stabilityai/stablelm-tuned-alpha-3b",
        "stabilityai/stablelm-tuned-alpha-7b",
        "stabilityai/japanese-stablelm-base-alpha-7b",
        "stabilityai/japanese-stablelm-instruct-alpha-7b",
        "cyberagent/open-calm-small",
        "cyberagent/open-calm-medium",
        "cyberagent/open-calm-large",
        "cyberagent/open-calm-1b",
        "cyberagent/open-calm-3b",
        "cyberagent/open-calm-7b",
        ]
    return ollm_model_ids


@clear_cache_decorator
def get_model_and_tokenizer_class(ollm_model_id, cpu_execution_chk=False):
    """Get model and tokenizer class.

    Args:
        ollm_model_id (str): String of Open LLM model ID.

    Returns:
        tuple(class, class, dict, dict): Tuple of model class, tokenizer class, model kwargs, and tokenizer kwargs.
    """
    llm = None
    for _, model_class in MODEL_REGISTRY.items():
        if model_class.include_name in ollm_model_id.lower():
            llm = model_class()

    if llm is None:
        llm = DefaultModel()

    llm.cpu_execution(cpu_execution_chk)

    # if "FreeWilly" in ollm_model_id:
    #     os.environ["SAFETENSORS_FAST_GPU"] = str(1)

    if platform.system() == "Darwin":
        llm.model_kwargs.update(dict(torch_dtype=torch.float32))

    print(f"model_kwargs: {llm.model_kwargs}")

    return llm


@clear_cache_decorator
def get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params):
    """Get generate kwargs.

    Args:
        tokenizer (class): Tokenizer class.
        inputs (dict): Inputs for generate method.
        ollm_model_id (str): String of Open LLM model ID.
        generate_params (dict): Generate parameters.

    Returns:
        dict: Generate kwargs.
    """
    generate_kwargs = dict(
        **inputs,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generate_kwargs.update(generate_params)

    if "stablelm-tuned" in ollm_model_id:
        stop = StopOnTokens()
        streamer = TextIteratorStreamer(
            tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)

        model_cache["preloaded_streamer"] = streamer

        stablelm_generate_kwargs = dict(
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([stop]),
        )

        generate_kwargs.update(stablelm_generate_kwargs)

    if "-GPTQ" in ollm_model_id:
        generate_kwargs.pop("token_type_ids", None)

    # print("generate_kwargs: " + str(generate_kwargs))
    return generate_kwargs
