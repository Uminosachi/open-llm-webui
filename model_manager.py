import platform

import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LlamaForCausalLM, LlamaTokenizer,
                          StoppingCriteriaList, TextIteratorStreamer)

from start_messages import StopOnTokens


def get_ollm_model_ids():
    """Get Open LLM and Llama model IDs.

    Returns:
        list: List of Open LLM model IDs.
    """
    ollm_model_ids = [
        "rinna/japanese-gpt-neox-3.6b",
        "rinna/japanese-gpt-neox-3.6b-instruction-sft",
        "rinna/japanese-gpt-neox-3.6b-instruction-sft-v2",
        "rinna/japanese-gpt-neox-3.6b-instruction-ppo",
        "cyberagent/open-calm-small",
        "cyberagent/open-calm-medium",
        "cyberagent/open-calm-large",
        "cyberagent/open-calm-1b",
        "cyberagent/open-calm-3b",
        "cyberagent/open-calm-7b",
        "TheBloke/Llama-2-7b-Chat-GPTQ",
        "TheBloke/Llama-2-13B-chat-GPTQ",
        "stabilityai/stablelm-tuned-alpha-3b",
        "stabilityai/stablelm-tuned-alpha-7b",
        "decapoda-research/llama-7b-hf",
        "decapoda-research/llama-13b-hf",
        ]
    return ollm_model_ids


def get_model_and_tokenizer_class(ollm_model_id):
    """Get model and tokenizer class.

    Args:
        ollm_model_id (str): String of Open LLM model ID.

    Returns:
        tuple(class, class, dict, dict): Tuple of model class, tokenizer class, model kwargs, and tokenizer kwargs.
    """
    if ("open-calm" in ollm_model_id or
            "japanese-gpt-neox" in ollm_model_id or
            "stablelm" in ollm_model_id):
        model_class = AutoModelForCausalLM
        tokenizer_class = AutoTokenizer

    elif "llama" in ollm_model_id:
        model_class = LlamaForCausalLM
        tokenizer_class = LlamaTokenizer

    elif "-GPTQ" in ollm_model_id:
        model_class = AutoGPTQForCausalLM
        tokenizer_class = AutoTokenizer

    else:
        model_class = AutoModelForCausalLM
        tokenizer_class = AutoTokenizer

    # if "FreeWilly" in ollm_model_id:
    #     os.environ["SAFETENSORS_FAST_GPU"] = str(1)

    if platform.system() == "Darwin":
        model_kwargs = dict(
            torch_dtype=torch.float32,
        )
    else:
        model_kwargs = dict(
            device_map="auto",
            torch_dtype=torch.float16,
        )
    tokenizer_kwargs = dict(
        use_fast=True,
    )

    if "japanese-gpt-neox" in ollm_model_id:
        tokenizer_kwargs["use_fast"] = False

    elif "-GPTQ" in ollm_model_id:
        model_basename = "gptq_model-4bit-128g"
        use_triton = False

        model_kwargs = dict(
            # revision="gptq-4bit-32g-actorder_True",
            model_basename=model_basename,
            inject_fused_attention=False,  # Required for Llama 2 70B model at this time.
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            use_triton=use_triton,
            quantize_config=None
        )

    print("model_class: " + model_class.__name__)
    print(f"model_kwargs: {model_kwargs}")
    return model_class, tokenizer_class, model_kwargs, tokenizer_kwargs


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
    global model_cache

    generate_kwargs = dict(
        **inputs,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generate_kwargs.update(generate_params)

    if "stablelm" in ollm_model_id:
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
