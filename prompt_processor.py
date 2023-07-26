from stablelm import start_message, system1_prompt, system2_prompt


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
        prompt = f"{system1_prompt}### Input:\n{input_text_box}\n\n### Response:\n"

    elif "FreeWilly2" in ollm_model_id:
        prompt = f"{system2_prompt}" + "".join([
            "\n\n".join(["### User:\n"+item[0],
                         "### Assistant:\n"+(item[1] if len(item[1]) == 0 else (item[1] + "\n\n"))
                         ]) for item in chatbot
            ])

    else:
        prompt = input_text_box

    return prompt


def retreive_output_text(input_text, output_text, ollm_model_id):
    """Retreive output text from generate method.

    Args:
        input_text (str): Input text.
        output_text (str): Output text from generate method.
        ollm_model_id (str): String of Open LLM model ID.

    Returns:
        str: Retreived output text.
    """
    global model_cache

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

    elif "llama" in ollm_model_id:
        output_text = output_text.lstrip(input_text + "\n")

    elif "FreeWilly1" in ollm_model_id:
        output_text = output_text.split("### Response:\n")[-1]

    elif "FreeWilly2" in ollm_model_id:
        output_text = output_text.split("### Assistant:\n")[-1]

    else:
        output_text = output_text

    return output_text
