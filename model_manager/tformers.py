import copy
import os
import platform

import torch
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import snapshot_download
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList,
                          TextIteratorStreamer)

from cache_manager import clear_cache_decorator, model_cache
from custom_logging import ollm_logging
from registry import get_llm_class, register_model
from start_messages import (StopOnTokens, chatqa_message, llama2_message, rakuten_message,
                            stablelm_message)

from .base import BaseAbstractLLM, LLMConfig, check_package_installed, replace_br_and_code


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
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        if tokenizer is not None and hasattr(tokenizer, "chat_template"):
            prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer,
                                             check_assistant=True)
        else:
            prompt = input_text_box

        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
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
            output_text_only=False,
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        prompt = input_text_box

        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


@register_model("gpt-neox")
class GPTNeoXModel(LLMConfig):
    include_name: str = "gpt-neox"

    download_kwargs = dict(ignore_patterns=["pytorch_model*"])

    chat_template1 = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ 'ユーザー: ' + message['content'] + '\\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% if not loop.last or message['content'] %}"
        "{{ 'システム: '  + message['content'] + '\\n' }}"
        "{% else %}"
        "{{ 'システム: ' }}"
        "{% endif %}"
        "{% endif %}{% endfor %}")

    chat_template2 = chat_template1.replace("\\n", "<NL>")

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
                legacy=False,
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
                add_special_tokens=False,
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        tokenizer.chat_template = self.chat_template1 if "bilingual-gpt-neox" in ollm_model_id else self.chat_template2
        prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer,
                                         check_assistant=False)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


@register_model("stablelm-tuned")
class StableLMTunedModel(LLMConfig):
    include_name: str = "stablelm-tuned"

    system_message = stablelm_message

    chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|USER|>' + message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|ASSISTANT|>' + message['content'] }}"
        "{% endif %}{% endfor %}")

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
            output_text_only=False,
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        tokenizer.chat_template = self.chat_template
        prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer,
                                         check_assistant=False)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)

        stop = StopOnTokens()
        streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)

        model_cache["preloaded_streamer"] = streamer

        stablelm_generate_kwargs = dict(
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([stop]),
        )

        generate_kwargs.update(stablelm_generate_kwargs)

        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        if model_cache.get("preloaded_streamer") is not None:
            streamer = model_cache.get("preloaded_streamer")
            partial_text = ""
            for new_text in streamer:
                partial_text += new_text

            output_text = partial_text

        return output_text


@register_model("japanese-stablelm")
class JapaneseStableLMModel(LLMConfig):
    include_name: str = "japanese-stablelm"

    system_message = "あなたは役立つアシスタントです。"
    user_query = "チャットボットとして応答に答えてください。"

    prompt_template = "[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user_query}\n\n{prompt} [/INST]"

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
                add_special_tokens=True,
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        prompt = self.prompt_template.format(system=self.system_message, user_query=self.user_query, prompt=input_text_box)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


@register_model("llama-2-chat-gptq")
class ChatGPTQModel(LLMConfig):
    include_name: str = "Llama-2-7b-Chat-GPTQ"

    system_message = llama2_message

    prompt_template = "[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt} [/INST]"

    def __init__(self):
        super().__init__(
            model_class=AutoGPTQForCausalLM,
            tokenizer_class=AutoTokenizer,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype=torch.float16,
                use_safetensors=True,
                trust_remote_code=False,
                revision="main",
                use_triton=False,
                quantize_config=None,
                offload_buffers=True,
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
            output_text_only=True,
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        prompt = self.prompt_template.format(system=self.system_message, prompt=input_text_box)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        generate_kwargs.pop("token_type_ids", None)
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


@register_model("kunoichi")
class KunoichiGPTQModel(LLMConfig):
    include_name: str = "Kunoichi"

    system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    prompt_template = "{system}\n\n### Instruction:\n{prompt}\n\n### Response:\n"

    def __init__(self):
        super().__init__(
            model_class=AutoGPTQForCausalLM,
            tokenizer_class=AutoTokenizer,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype=torch.float16,
                use_safetensors=True,
                trust_remote_code=False,
                revision="main",
                use_triton=False,
                quantize_config=None,
                offload_buffers=True,
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
            output_text_only=True,
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        prompt = self.prompt_template.format(system=self.system_message, prompt=input_text_box)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        generate_kwargs.pop("token_type_ids", None)
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


@register_model("phi-3")
class PHI3Model(LLMConfig):
    include_name: list = ["Phi-3-mini", "Phi-3-small"]

    system_message = "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user."

    def __init__(self):
        model_kwargs = dict(
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        if hasattr(self, "model_id") and "-small" in self.model_id:
            quantization_config = copy.deepcopy(self.quantization_4bit_config)
            quantization_config.llm_int8_skip_modules = ["o_proj", "lm_head"]
            model_kwargs.update(dict(quantization_config=quantization_config, torch_dtype=torch.float16))
        if not self.is_ampere_or_newer() or not check_package_installed("flash_attn"):
            model_kwargs.update(dict(dense_attention_every_n_layers=None))

        super().__init__(
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=dict(
                use_fast=True,
                trust_remote_code=True,
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
                add_special_tokens=False,
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer,
                                         check_assistant=True)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


@register_model("openelm")
class OpenELMModel(LLMConfig):
    include_name: str = "OpenELM"

    system_message = "You are a helpful assistant."

    chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ 'System: ' + message['content'] + '\\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ 'User: ' + message['content'] + '\\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% if not loop.last or message['content'] %}"
        "{{ 'Assistant: ' + message['content'] + '\\n' }}"
        "{% else %}"
        "{{ 'Assistant: ' }}"
        "{% endif %}"
        "{% endif %}{% endfor %}")

    def __init__(self):
        super().__init__(
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True,
            ),
            tokenizer_kwargs=dict(
                pretrained_model_name_or_path="meta-llama/Llama-2-7b-hf",
                use_fast=True,
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
                add_special_tokens=True,
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        tokenizer.chat_template = self.chat_template
        prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer,
                                         check_assistant=False)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        generate_kwargs["pad_token_id"] = 0
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


@register_model("gemma")
class GemmaModel(LLMConfig):
    include_name: str = "Gemma"

    download_kwargs = dict(ignore_patterns=["*.gguf"])

    def __init__(self):
        model_kwargs = dict(
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if hasattr(self, "model_id") and "-9b" in self.model_id:
            quantization_config = copy.deepcopy(self.quantization_4bit_config)
            quantization_config.llm_int8_skip_modules = ["o_proj", "lm_head"]
            model_kwargs.update(dict(quantization_config=quantization_config, torch_dtype=torch.float16))

        super().__init__(
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=dict(
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
                add_special_tokens=False,
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer,
                                         check_assistant=True)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


@register_model("qwen")
class QwenModel(LLMConfig):
    include_name: str = "Qwen"

    def __init__(self):
        super().__init__(
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype="auto",
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
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer,
                                         check_assistant=True)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


@register_model("rakuten")
class RakutenAIModel(LLMConfig):
    include_name: str = "Rakuten"

    download_kwargs = dict(ignore_patterns=["pytorch_model*"])

    system_message = rakuten_message

    chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ message['content'] + ' '}}"
        "{% elif message['role'] == 'user' %}"
        "{{ 'USER: ' + message['content'] + ' '}}"
        "{% elif message['role'] == 'assistant' %}"
        "{% if not loop.last or message['content'] %}"
        "{{ 'ASSISTANT: ' + message['content'] + ' '}}"
        "{% else %}"
        "{{ 'ASSISTANT: ' }}"
        "{% endif %}"
        "{% endif %}{% endfor %}")

    def __init__(self):
        super().__init__(
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype=torch.bfloat16,
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
            output_text_only=True,
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        tokenizer.chat_template = self.chat_template
        prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer,
                                         check_assistant=False)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


@register_model("youri-chat")
class RinnaYouriModel(LLMConfig):
    include_name: str = "youri-7b-chat"

    download_kwargs = dict(ignore_patterns=["pytorch_model*"])

    system_message = "チャットをしましょう。"

    chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '設定: ' + message['content'] + '\\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ 'ユーザー: ' + message['content'] + '\\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% if not loop.last or message['content'] %}"
        "{{ 'システム: ' + message['content'] + '\\n' }}"
        "{% else %}"
        "{{ 'システム: ' }}"
        "{% endif %}"
        "{% endif %}{% endfor %}")

    def __init__(self):
        super().__init__(
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype=torch.bfloat16,
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
            output_text_only=True,
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        tokenizer.chat_template = self.chat_template
        prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer,
                                         check_assistant=False)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


@register_model("chatqa")
class ChatQAModel(LLMConfig):
    include_name: str = "ChatQA"

    enable_rag_text = True
    system_message = chatqa_message
    instruction = "Please give a full and complete answer for the question."

    chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ 'System: ' + message['content'] + '\\n\\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ 'User: ' + message['content'] + '\\n\\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% if not loop.last or message['content'] %}"
        "{{ 'Assistant: ' + message['content'] + '\\n\\n' }}"
        "{% else %}"
        "{{ 'Assistant: ' }}"
        "{% endif %}"
        "{% endif %}{% endfor %}")

    def __init__(self):
        model_kwargs = dict(
            device_map="auto",
            torch_dtype=torch.float16,
        )
        if hasattr(self, "model_id") and "-8B" in self.model_id:
            quantization_config = copy.deepcopy(self.quantization_4bit_config)
            quantization_config.llm_int8_skip_modules = ["o_proj", "lm_head"]
            model_kwargs.update(dict(quantization_config=quantization_config))

        super().__init__(
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=dict(
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
                add_special_tokens=False,
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        tokenizer.chat_template = self.chat_template
        messages = [{"role": "system", "content": self.system_message + "\n\n" + rag_text_box}]
        for i, (user_text, assistant_text) in enumerate(chatbot):
            user_text = user_text if i > 0 else (self.instruction + " " + user_text)
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
        )
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        generate_kwargs["eos_token_id"] = terminators
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


@register_model("mistral")
class MistralModel(LLMConfig):
    include_name: str = "Mistral-*-Instruct"

    download_kwargs = dict(ignore_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"])

    # system_message = "You are a pirate chatbot who always responds in pirate speak!"

    chat_template = (
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

    def __init__(self):
        super().__init__(
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype=torch.bfloat16,
            ),
            tokenizer_kwargs=dict(
                use_fast=False,
                padding_side="left",
            ),
            tokenizer_input_kwargs=dict(
                return_tensors="pt",
                add_special_tokens=False,
            ),
            tokenizer_decode_kwargs=dict(
                skip_special_tokens=True,
            ),
            output_text_only=True,
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        tokenizer.chat_template = self.chat_template
        prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer,
                                         check_assistant=True)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


class TransformersLLM(BaseAbstractLLM):
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
        if os.path.isdir(ollm_model_id) and os.path.isfile(os.path.join(ollm_model_id, "config.json")):
            return LLMConfig.DOWNLOAD_COMPLETE
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
        if os.path.isdir(ollm_model_id) and os.path.isfile(os.path.join(ollm_model_id, "config.json")):
            model_params.model_kwargs.update(dict(local_files_only=True))
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


add_tfs_models_txt = os.path.join(os.path.dirname(os.path.realpath(__file__)), "add_tfs_models.txt")
add_tfs_models_store_txt = os.path.join(os.path.dirname(os.path.realpath(__file__)), "add_tfs_models_store.txt")


def get_ollm_model_ids():
    """Get Open LLM and Llama model IDs.

    Returns:
        list: List of Open LLM model IDs.
    """
    ollm_model_ids = [
        "microsoft/Phi-3-mini-4k-instruct",
        "google/gemma-2-9b-it",
        "google/gemma-1.1-2b-it",
        "google/gemma-1.1-7b-it",
        "nvidia/Llama3-ChatQA-1.5-8B",
        "Qwen/Qwen2-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Rakuten/RakutenAI-7B-chat",
        "Rakuten/RakutenAI-7B-instruct",
        "rinna/youri-7b-chat",
        "TheBloke/Llama-2-7b-Chat-GPTQ",
        "TheBloke/Kunoichi-7B-GPTQ",
        ]

    add_tfs_model_ids = []
    store_model_ids = []

    if os.path.isfile(add_tfs_models_txt):
        try:
            with open(add_tfs_models_txt, "r") as f:
                add_tfs_model_ids = [repo.strip() for repo in f.read().splitlines() if len(repo) > 0]

            with open(add_tfs_models_txt, "w") as f:
                pass
        except Exception:
            pass

    if os.path.isfile(add_tfs_models_store_txt):
        try:
            with open(add_tfs_models_store_txt, "r") as f:
                store_model_ids = [repo.strip() for repo in f.read().splitlines() if len(repo) > 0]
        except Exception:
            pass

    combined_model_ids = list(dict.fromkeys(add_tfs_model_ids + store_model_ids))
    ollm_logging.debug(f"combined_model_ids: {combined_model_ids}")

    try:
        with open(add_tfs_models_store_txt, "w") as f:
            f.write("\n".join(combined_model_ids) + "\n")
    except Exception:
        pass

    ollm_model_ids = combined_model_ids + ollm_model_ids

    return ollm_model_ids
