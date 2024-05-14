import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps

import torch
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import snapshot_download
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM,  # noqa: F401
                          LlamaTokenizer, StoppingCriteriaList, TextIteratorStreamer)

from cache_manager import clear_cache_decorator, model_cache
from chat_utils import convert_code_tags_to_md
from custom_logging import ollm_logging
from registry import get_llm_class, register_model
from start_messages import (StopOnTokens, chatqa_message, llama2_message, rakuten_message,
                            stablelm_message)


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
    model_kwargs: dict = field(default_factory=dict)
    tokenizer_kwargs: dict = field(default_factory=dict)
    tokenizer_input_kwargs: dict = field(default_factory=dict)
    tokenizer_decode_kwargs: dict = field(default_factory=dict)
    output_text_only: bool = True
    enable_rag_text: bool = False
    require_tokenization: bool = True

    DOWNLOAD_COMPLETE = "Download complete"

    def cpu_execution(self, cpu_execution_chk=False):
        if cpu_execution_chk:
            update_dict = dict(device_map="cpu", torch_dtype=torch.float32)
            self.model_kwargs.update(update_dict)

    @abstractmethod
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        pass

    def create_chat_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None, check_assistant=False):
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
        return prompt

    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
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
            output_text_only=True,
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
        prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer, check_assistant=False)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        # if "instruction-sft" in ollm_model_id or "instruction-ppo" in ollm_model_id:
        #     output_text = output_text.split("ユーザー: ")[-1].split("システム: ")[-1]
        #     output_text = output_text.replace("<NL>", "\n").lstrip()
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
        prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer, check_assistant=False)
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
                ollm_logging.debug(new_text)
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
        # if "stablelm-instruct" in ollm_model_id:
        #     output_text = output_text.split("[/INST]")[-1].lstrip()
        return output_text


@register_model("chat-gptq")
class ChatGPTQModel(LLMConfig):
    include_name: str = "chat-gptq"

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
    include_name: str = "kunoichi"

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
    include_name: str = "phi-3"

    system_message = "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user."

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
        )

    @replace_br_and_code
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer=None):
        prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer, check_assistant=True)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        # output_text = output_text.split("<|user|>")[-1].split("<|end|>")[1].split("<|assistant|>")
        # output_text = "\n".join([text.replace("<|end|>", "").lstrip() for text in output_text if len(text.strip()) > 0])
        return output_text


@register_model("openelm")
class OpenELMModel(LLMConfig):
    include_name: str = "openelm"

    system_message = "Let's chat!"

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
        prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer, check_assistant=False)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        generate_kwargs["pad_token_id"] = 0
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        # output_text = output_text.lstrip(input_text).lstrip()
        return output_text


@register_model("gemma")
class GemmaModel(LLMConfig):
    include_name: str = "gemma"

    download_kwargs = dict(ignore_patterns=["*.gguf"])

    def __init__(self):
        super().__init__(
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
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
        prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer, check_assistant=True)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        # output_text = output_text.split("<start_of_turn>model\n")[-1].split("<end_of_turn>\n")[0].rstrip("<eos>")
        return output_text


@register_model("rakuten")
class RakutenAIModel(LLMConfig):
    include_name: str = "rakuten"

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
        prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer, check_assistant=False)
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = super().get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


@register_model("youri")
class RinnaYouriModel(LLMConfig):
    include_name: str = "youri"

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
        prompt = self.create_chat_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer, check_assistant=False)
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
    include_name: str = "chatqa"

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
        super().__init__(
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            model_kwargs=dict(
                device_map="auto",
                torch_dtype=torch.float16,
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
            enable_rag_text=True,
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


class TransformersLLM:
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

        Returns:
            tuple(class, class, dict, dict): Tuple of model class, tokenizer class, model kwargs, and tokenizer kwargs.
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


def get_ollm_model_ids():
    """Get Open LLM and Llama model IDs.

    Returns:
        list: List of Open LLM model IDs.
    """
    ollm_model_ids = [
        "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-mini-128k-instruct",
        "google/gemma-1.1-2b-it",
        "google/gemma-1.1-7b-it",
        "nvidia/Llama3-ChatQA-1.5-8B",
        "apple/OpenELM-1_1B-Instruct",
        "apple/OpenELM-3B-Instruct",
        "Rakuten/RakutenAI-7B-chat",
        "Rakuten/RakutenAI-7B-instruct",
        "rinna/youri-7b-chat",
        "rinna/bilingual-gpt-neox-4b-instruction-sft",
        "rinna/japanese-gpt-neox-3.6b-instruction-sft-v2",
        "rinna/japanese-gpt-neox-3.6b-instruction-ppo",
        "TheBloke/Llama-2-7b-Chat-GPTQ",
        "TheBloke/Kunoichi-7B-GPTQ",
        "stabilityai/stablelm-tuned-alpha-3b",
        "stabilityai/stablelm-tuned-alpha-7b",
        "stabilityai/japanese-stablelm-instruct-beta-7b",
        ]
    return ollm_model_ids
