import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer,
                          StoppingCriteriaList, TextIteratorStreamer)

from cache_manager import clear_cache_decorator, model_cache
from registry import MODEL_REGISTRY, register_model
from start_messages import StopOnTokens, llama2_message, start_message


def replace_br(func):
    def wrapper(self, chatbot, *args, **kwargs):
        chatbot = [[item.replace("<br>", "\n") for item in sublist] for sublist in chatbot]
        return func(self, chatbot, *args, **kwargs)
    return wrapper


@dataclass
class LLMConfig(ABC):
    model_class: object
    tokenizer_class: object
    model_kwargs: dict = field(default_factory=dict)
    tokenizer_kwargs: dict = field(default_factory=dict)
    tokenizer_input_kwargs: dict = field(default_factory=dict)
    tokenizer_decode_kwargs: dict = field(default_factory=dict)
    output_text_only: bool = True

    def cpu_execution(self, cpu_execution_chk=False):
        if cpu_execution_chk:
            update_dict = dict(device_map="cpu", torch_dtype=torch.float32)
            self.model_kwargs.update(update_dict)

    @abstractmethod
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, tokenizer=None):
        pass

    @abstractmethod
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        pass

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

    @replace_br
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, tokenizer=None):
        prompt = input_text_box

        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = dict(
            **inputs,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generate_kwargs.update(generate_params)

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

    @replace_br
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, tokenizer=None):
        prompt = input_text_box

        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = dict(
            **inputs,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generate_kwargs.update(generate_params)

        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


@register_model("gpt-neox")
class GPTNeoXModel(LLMConfig):
    include_name: str = "gpt-neox"

    chat_template1 = ("{% for message in messages %}"
                      "{% if message['role'] == 'user' %}"
                      "{{ 'ユーザー: ' + message['content'] + '\\n' }}"
                      "{% elif message['role'] == 'system' %}"
                      "{% if message['content'] %}"
                      "{{ 'システム: ' + message['content'] + '\\n' }}"
                      "{% else %}"
                      "{{ 'システム: ' }}"
                      "{% endif %}"
                      "{% elif message['role'] == 'assistant' %}"
                      "{% if message['content'] %}"
                      "{{ 'アシスタント: '  + message['content'] + '\\n' }}"
                      "{% else %}"
                      "{{ 'アシスタント: ' }}"
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

    @replace_br
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, tokenizer=None):
        if "instruction-sft" in ollm_model_id or "instruction-ppo" in ollm_model_id:
            tokenizer.chat_template = self.chat_template1 if "bilingual-gpt-neox" in ollm_model_id else self.chat_template2
            messages = []
            for user_text, assistant_text in chatbot:
                messages.append({"role": "user", "content": user_text})
                messages.append({"role": "system", "content": assistant_text})
            prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
            )
        else:
            prompt = input_text_box

        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = dict(
            **inputs,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generate_kwargs.update(generate_params)

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

    chat_template = ("{% for message in messages %}"
                     "{% if message['role'] == 'user' %}"
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

    @replace_br
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, tokenizer=None):
        tokenizer.chat_template = self.chat_template
        messages = []
        for user_text, assistant_text in chatbot:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
        )
        prompt = start_message + prompt

        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = dict(
            **inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generate_kwargs.update(generate_params)

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
            output_text_only=True,
        )

    @replace_br
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, tokenizer=None):
        if "stablelm-instruct" in ollm_model_id:
            def build_prompt(user_query, inputs):
                sys_msg = "[INST] <<SYS>>\nあなたは役立つアシスタントです。\n<<SYS>>\n\n"
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
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = dict(
            **inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generate_kwargs.update(generate_params)

        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        # if "stablelm-instruct" in ollm_model_id:
        #     output_text = output_text.split("[/INST]")[-1].lstrip()
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
            output_text_only=False,
        )

    @replace_br
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, tokenizer=None):
        if len(chatbot) < 2:
            prompt = f"[INST] <<SYS>>\n{llama2_message}\n<</SYS>>\n\n{input_text_box} [/INST] "
        else:
            prompt = f"[INST] <<SYS>>\n{llama2_message}\n<</SYS>>\n\n{chatbot[0][0]} [/INST] {chatbot[0][1]}"
            prompt = prompt + "".join([(" [INST] "+item[0]+" [/INST] "+item[1]) for item in chatbot[1:]])

        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = dict(
            **inputs,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generate_kwargs.update(generate_params)

        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        if "Llama-2-" in ollm_model_id:
            output_text = output_text.split("[/INST]")[-1].lstrip()
        elif "llama-" in ollm_model_id:
            output_text = output_text.lstrip(input_text).lstrip()

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
            output_text_only=False,
        )

    @replace_br
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, tokenizer=None):
        prompt = input_text_box

        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = dict(
            **inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generate_kwargs.update(generate_params)

        generate_kwargs.pop("token_type_ids", None)

        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        return output_text


@register_model("phi-3")
class PHI3Model(LLMConfig):
    include_name: str = "phi-3"

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

    @replace_br
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, tokenizer=None):
        messages = [
            {"role": "system", "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user."},
        ]
        for user_text, assistant_text in chatbot:
            messages.append({"role": "user", "content": user_text})
            if len(assistant_text) > 0:
                messages.append({"role": "assistant", "content": assistant_text})
        prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
        )
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = dict(
            **inputs,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generate_kwargs.update(generate_params)

        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        # output_text = output_text.split("<|user|>")[-1].split("<|end|>")[1].split("<|assistant|>")
        # output_text = "\n".join([text.replace("<|end|>", "").lstrip() for text in output_text if len(text.strip()) > 0])
        return output_text


@register_model("openelm")
class OpenELMModel(LLMConfig):
    include_name: str = "openelm"

    chat_template = ("{% for message in messages %}"
                     "{% if message['role'] == 'user' %}"
                     "{{ message['content'] + '\\n' }}"
                     "{% elif message['role'] == 'system' %}"
                     "{{ message['content'] + '\\n' }}"
                     "{% elif message['role'] == 'assistant' %}"
                     "{{ message['content'] + '\\n' }}"
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

    @replace_br
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, tokenizer=None):
        tokenizer.chat_template = self.chat_template

        messages = [
            {"role": "system", "content": "Let's chat"},
        ]
        for user_text, assistant_text in chatbot:
            messages.append({"role": "user", "content": user_text})
            if len(assistant_text) > 0:
                messages.append({"role": "assistant", "content": assistant_text})
        prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
        )
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = dict(
            **inputs,
            do_sample=True,
            pad_token_id=0,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generate_kwargs.update(generate_params)

        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        # output_text = output_text.lstrip(input_text).lstrip()
        return output_text


@register_model("gemma")
class GemmaModel(LLMConfig):
    include_name: str = "gemma"

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

    @replace_br
    @clear_cache_decorator
    def create_prompt(self, chatbot, ollm_model_id, input_text_box, tokenizer=None):
        messages = []
        for user_text, assistant_text in chatbot:
            messages.append({"role": "user", "content": user_text})
            if len(assistant_text) > 0:
                messages.append({"role": "assistant", "content": assistant_text})
        prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
        )
        return prompt

    @clear_cache_decorator
    def get_generate_kwargs(self, tokenizer, inputs, ollm_model_id, generate_params):
        generate_kwargs = dict(
            **inputs,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generate_kwargs.update(generate_params)

        return generate_kwargs

    @clear_cache_decorator
    def retreive_output_text(self, input_text, output_text, ollm_model_id, tokenizer=None):
        # output_text = output_text.split("<start_of_turn>model\n")[-1].split("<end_of_turn>\n")[0].rstrip("<eos>")
        return output_text


def get_ollm_model_ids():
    """Get Open LLM and Llama model IDs.

    Returns:
        list: List of Open LLM model IDs.
    """
    ollm_model_ids = [
        "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-mini-128k-instruct",
        "google/gemma-2b-it",
        "google/gemma-1.1-2b-it",
        "apple/OpenELM-1_1B-Instruct",
        "apple/OpenELM-3B-Instruct",
        "rinna/bilingual-gpt-neox-4b-instruction-sft",
        "rinna/japanese-gpt-neox-3.6b-instruction-sft-v2",
        "rinna/japanese-gpt-neox-3.6b-instruction-ppo",
        "TheBloke/Llama-2-7b-Chat-GPTQ",
        "TheBloke/Llama-2-13B-chat-GPTQ",
        "stabilityai/stablelm-tuned-alpha-3b",
        "stabilityai/stablelm-tuned-alpha-7b",
        "stabilityai/japanese-stablelm-instruct-beta-7b",
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
