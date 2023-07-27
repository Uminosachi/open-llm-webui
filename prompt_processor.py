from cache_manager import clear_cache_decorator
from model_manager import model_cache
from start_messages import (freewilly1_prompt, freewilly2_prompt,
                            llama2_message, start_message)


@clear_cache_decorator
def create_prompt(chatbot, ollm_model_id, input_text_box):
    """Create prompt for generate method.

    Args:
        chatbot (list): Chatbot history.
        ollm_model_id (str): String of Open LLM model ID.
        input_text_box (str): Input text.

    Returns:
        str: Prompt for generate method.
    """
    if "instruction-sft" in ollm_model_id or "instruction-ppo" in ollm_model_id:
        sft_input_text = []
        for user_text, system_text in chatbot:
            sft_input_text.append("ユーザー: " + user_text + "<NL>システム: " + system_text)

        sft_input_text = "<NL>".join(sft_input_text)

        prompt = sft_input_text
    elif "stablelm" in ollm_model_id:
        prompt = start_message + "".join(["".join(["<|USER|>"+item[0], "<|ASSISTANT|>"+item[1]]) for item in chatbot])

    elif "FreeWilly1" in ollm_model_id:
        prompt = f"{freewilly1_prompt}### Input:\n{input_text_box}\n\n### Response:\n"

    elif "FreeWilly2" in ollm_model_id:
        prompt = f"{freewilly2_prompt}" + "".join([
            "\n\n".join(["### User:\n"+item[0],
                         "### Assistant:\n"+(item[1] if len(item[1]) == 0 else (item[1] + "\n\n"))
                         ]) for item in chatbot
            ])

    elif "Llama-2-" in ollm_model_id:
        if len(chatbot) < 2:
            prompt = f"[INST] <<SYS>>\n{llama2_message}\n<</SYS>>\n\n{input_text_box} [/INST] "
        else:
            prompt = f"[INST] <<SYS>>\n{llama2_message}\n<</SYS>>\n\n{chatbot[0][0]} [/INST] {chatbot[0][1]}"
            prompt = prompt + "".join([(" [INST] "+item[0]+" [/INST] "+item[1]) for item in chatbot[1:]])

    else:
        prompt = input_text_box

    return prompt


@clear_cache_decorator
def retreive_output_text(input_text, output_text, ollm_model_id):
    """Retreive output text from generate method.

    Args:
        input_text (str): Input text.
        output_text (str): Output text from generate method.
        ollm_model_id (str): String of Open LLM model ID.

    Returns:
        str: Retreived output text.
    """
    if "instruction-sft" in ollm_model_id or "instruction-ppo" in ollm_model_id:
        output_text = output_text.split("<NL>")[-1].replace("システム: ", "")

    elif "stablelm" in ollm_model_id:
        if model_cache.get("preloaded_streamer") is not None:
            streamer = model_cache.get("preloaded_streamer")
            partial_text = ""
            for new_text in streamer:
                # print(new_text)
                partial_text += new_text

            output_text = partial_text
        else:
            output_text = output_text

    elif "FreeWilly1" in ollm_model_id:
        output_text = output_text.split("### Response:\n")[-1]

    elif "FreeWilly2" in ollm_model_id:
        output_text = output_text.split("### Assistant:\n")[-1]

    elif "Llama-2-" in ollm_model_id:
        output_text = output_text.split("[/INST]")[-1].lstrip()

    elif "llama-" in ollm_model_id:
        output_text = output_text.lstrip(input_text + "\n").lstrip()

    else:
        output_text = output_text

    return output_text
