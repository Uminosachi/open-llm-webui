import time

import gradio as gr
import torch
from huggingface_hub import snapshot_download

from cache_manager import clear_cache, clear_cache_decorator, model_cache
from model_manager import get_generate_kwargs, get_model_and_tokenizer_class, get_ollm_model_ids
from prompt_processor import retreive_output_text
from translator import load_translator, translate

# from huggingface_hub import try_to_load_from_cache
# from transformers import OpenLlamaModel, OpenLlamaConfig


_DOWNLOAD_COMPLETED = "Download complete"


@clear_cache_decorator
def download_model(ollm_model_id, local_files_only=False):
    """Download Open LLM and Llama models.

    Args:
        ollm_model_id (str): String of Open LLM model ID.
        local_files_only (bool, optional): If True, use only local files. Defaults to False.

    Returns:
        str: string of download result.
    """
    if not local_files_only:
        print(f"Downloading {ollm_model_id}")
    try:
        snapshot_download(repo_id=ollm_model_id, local_files_only=local_files_only)
    except FileNotFoundError:
        return "Model not found. Please click Download model button."
    except Exception as e:
        return str(e)

    return _DOWNLOAD_COMPLETED


@clear_cache_decorator
def ollm_inference(chatbot, ollm_model_id, input_text_box,
                   max_new_tokens, temperature, top_k, top_p, repetition_penalty, translate_chk, cpu_execution_chk=False):
    """Open LLM inference.

    Args:
        chatbot (list): Chatbot history.
        ollm_model_id (str): String of Open LLM model ID.
        input_text_box (str): Input text.
        max_new_tokens (int): Parameter for generate method.
        temperature (float): Parameter for generate method.
        top_k (int): Parameter for generate method.
        top_p (float): Parameter for generate method.
        repetition_penalty (float): Parameter for generate method.
        translate_chk (bool): If True, translate output text.
        cpu_execution_chk (bool, optional): If True, use CPU. Defaults to False.

    Returns:
        tuple(str, list, str): Input text, chatbot history, and inference result.
    """
    generate_params = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=float(repetition_penalty),
    )

    if input_text_box is None or len(input_text_box.strip()) == 0:
        return "", chatbot, "Input text is empty.", ""

    chatbot = [] if chatbot is None else chatbot

    dwonload_result = download_model(ollm_model_id, local_files_only=True)
    if dwonload_result != _DOWNLOAD_COMPLETED:
        return input_text_box, chatbot, dwonload_result, ""

    model_params = get_model_and_tokenizer_class(ollm_model_id, cpu_execution_chk)

    pmnop = "pretrained_model_name_or_path"

    print(f"Loading {ollm_model_id}")
    if (model_cache.get("preloaded_model_id") != ollm_model_id or
            model_cache.get("preloaded_model") is None or
            model_cache.get("preloaded_tokenizer") is None):

        for key in model_cache.keys():
            model_cache[key] = None

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
        model.tie_weights()

        tokenizer = model_params.tokenizer_class.from_pretrained(
            ollm_model_id if pmnop not in model_params.tokenizer_kwargs else model_params.tokenizer_kwargs.pop(pmnop),
            **model_params.tokenizer_kwargs,
        )

        model_cache["preloaded_model_id"] = ollm_model_id
        model_cache["preloaded_model"] = model
        model_cache["preloaded_tokenizer"] = tokenizer
        model_cache["preloaded_device"] = "cpu device" if cpu_execution_chk else "auto device"
        clear_cache()

    else:
        print("Using preloaded model on {}".format(model_cache.get("preloaded_device")))
        model = model_cache["preloaded_model"]
        tokenizer = model_cache["preloaded_tokenizer"]

    prompt = model_params.create_prompt(chatbot, ollm_model_id, input_text_box)

    print("Input text: " + prompt)
    print("Generating...")
    inputs = tokenizer(
        [prompt],
        **model_params.tokenizer_input_kwargs,
    ).to(model.device)

    generate_kwargs = get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
    clear_cache()

    t1 = time.time()
    with torch.no_grad():
        tokens = model.generate(
            **generate_kwargs
        )
    t2 = time.time()
    elapsed_time = t2-t1
    print(f"Generation time: {elapsed_time} seconds")

    clear_cache()

    output = tokenizer.decode(
        tokens[0],
        **model_params.tokenizer_decode_kwargs,
    )

    output = retreive_output_text(input_text_box, output, ollm_model_id)

    print("Generation complete")
    print("Output text: " + output)

    if translate_chk:
        translated_output_text = translate(output, "en", "ja")
        print("Translated output text: " + translated_output_text)
    else:
        translated_output_text = ""

    output = output.replace("\n", "<br>")
    # chatbot.append((input_text_box, output))
    chatbot[-1][1] = output

    return "", chatbot, f"Generation time: {elapsed_time} seconds", translated_output_text


@clear_cache_decorator
def user(message, history, translate_chk):
    # Append the user's message to the conversation history
    if len(message.strip()) > 0:
        if translate_chk:
            message = translate(message, "ja", "en")
            print("Translated input text: " + message)
        return message, history + [[message, ""]]
    else:
        return message, history


@clear_cache_decorator
def translate_change(translate_chk):
    if translate_chk:
        load_translator()

    return "", "Translation enabled" if translate_chk else "Translation disabled"


def on_ui_tabs():
    ollm_model_ids = get_ollm_model_ids()
    ollm_model_index = ollm_model_ids.index("rinna/bilingual-gpt-neox-4b-instruction-sft") \
        if "rinna/bilingual-gpt-neox-4b-instruction-sft" in ollm_model_ids else 0

    block = gr.Blocks().queue()
    block.title = "Open LLM WebUI"
    with block as ollm_interface:
        with gr.Row():
            gr.Markdown("## Open LLM WebUI")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    chatbot = gr.Chatbot(value=[], elem_id="chatbot", height=640)

            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            ollm_model_id = gr.Dropdown(label="LLM model ID", elem_id="ollm_model_id", choices=ollm_model_ids,
                                                        value=ollm_model_ids[ollm_model_index], show_label=True)
                        with gr.Row():
                            cpu_execution_chk = gr.Checkbox(label="CPU execution", elem_id="cpu_execution_chk", value=False, show_label=True)
                    with gr.Column():
                        with gr.Row():
                            download_model_btn = gr.Button("Download model", elem_id="download_model_btn")
                        with gr.Row():
                            status_text = gr.Textbox(label="", max_lines=1, show_label=False, interactive=False)
                with gr.Row():
                    input_text_box = gr.Textbox(
                        label="Input text",
                        placeholder="Send a message",
                        show_label=True,
                    )
                with gr.Row():
                    max_new_tokens = gr.Slider(minimum=1, maximum=512, step=1, value=128, label="Max new tokens", elem_id="max_new_tokens")
                with gr.Row():
                    with gr.Accordion("Advanced options", open=False):
                        translate_chk = gr.Checkbox(label="Translate (ja->en/en->ja)", elem_id="translate_chk", value=False, show_label=True)
                        temperature = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.7, label="Temperature", elem_id="temperature")
                        top_k = gr.Slider(minimum=1, maximum=200, step=1, value=50, label="Top k", elem_id="top_k")
                        top_p = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=1.0, label="Top p", elem_id="top_p")
                        repetition_penalty = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=1.0, label="Repetition penalty", elem_id="repetition_penalty")
                with gr.Row():
                    generate_btn = gr.Button("Generate", elem_id="generate_btn")
                with gr.Row():
                    translated_output_text = gr.Textbox(label="Translated output text", show_label=True, lines=3, interactive=False)
                with gr.Row():
                    clear_btn = gr.Button("Clear text", elem_id="clear_btn")

            download_model_btn.click(fn=download_model, inputs=[ollm_model_id], outputs=[status_text])
            translate_chk.change(fn=translate_change, inputs=[translate_chk], outputs=[input_text_box, status_text])

            generate_inputs = [chatbot, ollm_model_id, input_text_box,
                               max_new_tokens, temperature, top_k, top_p, repetition_penalty, translate_chk, cpu_execution_chk]
            generate_btn.click(fn=user, inputs=[input_text_box, chatbot, translate_chk], outputs=[input_text_box, chatbot]).then(
                fn=ollm_inference, inputs=generate_inputs, outputs=[input_text_box, chatbot, status_text, translated_output_text])
            input_text_box.submit(fn=user, inputs=[input_text_box, chatbot, translate_chk], outputs=[input_text_box, chatbot]).then(
                fn=ollm_inference, inputs=generate_inputs, outputs=[input_text_box, chatbot, status_text, translated_output_text])

            clear_btn.click(lambda: [None, None], None, [input_text_box, chatbot])

    return [(ollm_interface, "Open LLM", "open_llm")]


block, _, _ = on_ui_tabs()[0]
block.launch()
