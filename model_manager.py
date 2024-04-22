import platform
from dataclasses import dataclass, field

import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer,
                          StoppingCriteriaList, TextIteratorStreamer)

from cache_manager import clear_cache_decorator, model_cache
from registry import MODEL_REGISTRY, register_model
from start_messages import StopOnTokens, llama2_message, start_message


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
            update_dict = dict(device_map="cpu", torch_dtype=torch.float32)
            self.model_kwargs.update(update_dict)


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

    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box):
        prompt = input_text_box

        return prompt

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id):
        return output_text


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

    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box):
        prompt = input_text_box

        return prompt

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id):
        return output_text


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

    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box):
        if "instruction-sft" in ollm_model_id or "instruction-ppo" in ollm_model_id:
            sft_input_text = []
            new_line = "\n" if "bilingual-gpt-neox" in ollm_model_id else "<NL>"
            for user_text, system_text in chatbot:
                sft_input_text.append(f"ユーザー: {user_text}{new_line}システム: {system_text}")

            sft_input_text = f"{new_line}".join(sft_input_text)

            prompt = sft_input_text
        else:
            prompt = input_text_box

        return prompt

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id):
        if "instruction-sft" in ollm_model_id or "instruction-ppo" in ollm_model_id:
            new_line = "\n" if "bilingual-gpt-neox" in ollm_model_id else "<NL>"
            output_text = output_text.split(f"{new_line}")[-1].replace("システム: ", "")

        return output_text


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

    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box):
        prompt = start_message + "".join(["".join(["<|USER|>"+item[0], "<|ASSISTANT|>"+item[1]]) for item in chatbot])

        return prompt

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id):
        if model_cache.get("preloaded_streamer") is not None:
            streamer = model_cache.get("preloaded_streamer")
            partial_text = ""
            for new_text in streamer:
                # print(new_text)
                partial_text += new_text

            output_text = partial_text

        return output_text


@register_model("japanese-stablelm")
class JapaneseStableLMModel(LLMConfig):
    include_name: str = "japanese-stablelm"

    def __init__(self):
        super().__init__(
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ),
            tokenizer_kwargs=dict(
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
                add_special_tokens=False,
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
        )

    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box):
        if "stablelm-instruct" in ollm_model_id:
            def build_prompt(user_query, inputs):
                sys_msg = "<s>[INST] <<SYS>>\nあなたは役立つアシスタントです。\n<<SYS>>\n\n"
                p = sys_msg + user_query + "\n\n" + inputs + " [/INST] "
                return p

            user_inputs = {
                "user_query": "チャットボットとして応答に答えてください。",
                "inputs": input_text_box,
            }
            prompt = build_prompt(**user_inputs)
        else:
            prompt = input_text_box

        return prompt

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id):
        if "stablelm-instruct" in ollm_model_id:
            output_text = output_text.split("[/INST]")[-1].lstrip()

        return output_text


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

    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box):
        if len(chatbot) < 2:
            prompt = f"[INST] <<SYS>>\n{llama2_message}\n<</SYS>>\n\n{input_text_box} [/INST] "
        else:
            prompt = f"[INST] <<SYS>>\n{llama2_message}\n<</SYS>>\n\n{chatbot[0][0]} [/INST] {chatbot[0][1]}"
            prompt = prompt + "".join([(" [INST] "+item[0]+" [/INST] "+item[1]) for item in chatbot[1:]])

        return prompt

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id):
        if "Llama-2-" in ollm_model_id:
            output_text = output_text.split("[/INST]")[-1].lstrip()
        elif "llama-" in ollm_model_id:
            output_text = output_text.lstrip(input_text + "\n").lstrip()

        return output_text


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
                offload_buffers=True,
                use_marlin=True,
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

    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box):
        prompt = input_text_box

        return prompt

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id):
        return output_text


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
        "stabilityai/japanese-stablelm-base-beta-7b",
        "stabilityai/japanese-stablelm-instruct-beta-7b",
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
    pad_to_eos = [GPTQModel.include_name]
    pad_token_id = tokenizer.eos_token_id if any([name in ollm_model_id.lower() for name in pad_to_eos]) else tokenizer.pad_token_id
    generate_kwargs = dict(
        **inputs,
        do_sample=True,
        pad_token_id=pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generate_kwargs.update(generate_params)

    if StableLMTunedModel.include_name in ollm_model_id.lower():
        stop = StopOnTokens()
        streamer = TextIteratorStreamer(
            tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)

        model_cache["preloaded_streamer"] = streamer

        stablelm_generate_kwargs = dict(
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([stop]),
        )

        generate_kwargs.update(stablelm_generate_kwargs)

    if GPTQModel.include_name in ollm_model_id.lower():
        generate_kwargs.pop("token_type_ids", None)

    # print("generate_kwargs: " + str(generate_kwargs))
    return generate_kwargs
