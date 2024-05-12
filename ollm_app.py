import os
import time
import types  # noqa: F401
from collections import UserDict

import gradio as gr
import torch

from cache_manager import ClearCacheContext, clear_cache, clear_cache_decorator, model_cache
from chat_utils import replace_newlines
from custom_logging import ollm_logging
from model_manager import LLMConfig, TransformersLLM, get_ollm_model_ids
from model_manager_cpp import CPPDefaultModel, LlamaCPPLLM, get_cpp_ollm_model_ids
from registry import get_llm_class
from translator import load_translator, translate

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

methods_tabs = ["transformers", "llama.cpp"]
selected_tab = methods_tabs[0]


@clear_cache_decorator
def change_tab_first():
    global selected_tab
    selected_tab = methods_tabs[0]
    ollm_logging.debug(f"Selected tab: {selected_tab}")
    return None


@clear_cache_decorator
def change_tab_second():
    global selected_tab
    selected_tab = methods_tabs[1]
    ollm_logging.debug(f"Selected tab: {selected_tab}")
    return None


@clear_cache_decorator
def ollm_inference(chatbot, ollm_model_id, cpp_ollm_model_id, cpp_chat_template, input_text_box, rag_text_box,
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

    return_status = ["", ""]
    if input_text_box is None or len(input_text_box.strip()) == 0:
        return_status[methods_tabs.index(selected_tab)] = "Input text is empty."
        return ("", chatbot, *return_status, "")

    chatbot = [] if chatbot is None else chatbot

    if selected_tab == methods_tabs[0]:
        method_class = TransformersLLM
    else:
        method_class = LlamaCPPLLM
        ollm_model_id = cpp_ollm_model_id

    dwonload_result = method_class.download_model(ollm_model_id, local_files_only=True)
    ollm_logging.debug(f"Download result: {dwonload_result}")
    if dwonload_result != LLMConfig.DOWNLOAD_COMPLETE:
        return_status[methods_tabs.index(selected_tab)] = dwonload_result
        return (input_text_box, chatbot, *return_status, "")

    model_params = method_class.get_model_and_tokenizer_class(ollm_model_id, cpu_execution_chk)

    # pmnop = "pretrained_model_name_or_path"

    ollm_logging.info(f"Loading {ollm_model_id}")
    if (model_cache.get("preloaded_model_id") != ollm_model_id or
            model_cache.get("preloaded_model") is None or model_cache.get("preloaded_tokenizer") is None or
            model_cache.get("preloaded_device") != ("cpu device" if cpu_execution_chk else "auto device")):

        for key in model_cache.keys():
            model_cache[key] = None
        clear_cache()

        model = method_class.get_model(ollm_model_id, model_params, generate_params)

        tokenizer = method_class.get_tokenizer(ollm_model_id, model_params)

        model_cache["preloaded_model_id"] = ollm_model_id
        model_cache["preloaded_model"] = model
        model_cache["preloaded_tokenizer"] = tokenizer
        model_cache["preloaded_device"] = "cpu device" if cpu_execution_chk else "auto device"

    else:
        ollm_logging.info("Using preloaded model on {}".format(model_cache.get("preloaded_device")))
        model = model_cache["preloaded_model"]
        tokenizer = model_cache["preloaded_tokenizer"]

    if selected_tab == methods_tabs[1]:
        if hasattr(model, "metadata") and model.metadata.get("tokenizer.chat_template", None) is not None:
            ollm_logging.info("Using chat template from model metadata")
            tokenizer.chat_template = model.metadata["tokenizer.chat_template"]
        else:
            # ["Llama2", "Llama3", "Gemma", "Phi-3"]
            if cpp_chat_template == "Llama2":
                ollm_logging.info("Using Llama 2 chat template")
                tokenizer.chat_template = CPPDefaultModel.llama2_template
                CPPDefaultModel.system_message = "Let's chat!"
            elif cpp_chat_template == "Llama3":
                ollm_logging.info("Using Llama 3 chat template")
                tokenizer.chat_template = CPPDefaultModel.llama3_template
                CPPDefaultModel.system_message = "Let's chat!"
            elif cpp_chat_template == "Gemma":
                ollm_logging.info("Using Gemma chat template")
                tokenizer.chat_template = CPPDefaultModel.gemma_template
                if hasattr(CPPDefaultModel, "system_message"):
                    del CPPDefaultModel.system_message
            elif cpp_chat_template == "Phi-3":
                ollm_logging.info("Using Phi-3 chat template")
                tokenizer.chat_template = CPPDefaultModel.phi3_template
                if hasattr(CPPDefaultModel, "system_message"):
                    del CPPDefaultModel.system_message
    prompt = model_params.create_prompt(chatbot, ollm_model_id, input_text_box, rag_text_box, tokenizer)

    ollm_logging.info("Input text: " + prompt)
    ollm_logging.info("Generating...")
    if model_params.require_tokenization:
        with ClearCacheContext():
            inputs = tokenizer(
                [prompt],
                **model_params.tokenizer_input_kwargs,
            )
        if hasattr(model, "device"):
            if isinstance(inputs, UserDict):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(model.device)
    else:
        inputs = prompt

    generate_kwargs = model_params.get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)

    t1 = time.time()
    if model_params.require_tokenization:
        with ClearCacheContext(), torch.no_grad():
            tokens = model.generate(
                **generate_kwargs
            )
    else:
        with ClearCacheContext():
            tokens = model.create_completion(
                **generate_kwargs
            )
    t2 = time.time()
    elapsed_time = t2-t1
    ollm_logging.info(f"Generation time: {elapsed_time} seconds")

    if model_params.require_tokenization:
        input_ids = generate_kwargs["input_ids"]
        with ClearCacheContext():
            output_text = tokenizer.decode(
                tokens[0] if not model_params.output_text_only else tokens[0][len(input_ids[0]):],
                **model_params.tokenizer_decode_kwargs,
            )
    else:
        output_text = tokens["choices"][0]["text"]

    output_text = model_params.retreive_output_text(prompt, output_text, ollm_model_id, tokenizer)

    ollm_logging.info("Generation complete")
    ollm_logging.info("Output text: " + output_text)

    if translate_chk:
        translated_output_text = translate(output_text, "en", "ja")
        ollm_logging.info("Translated output text: " + translated_output_text)
    else:
        translated_output_text = ""

    if "```" not in output_text:
        output_text = output_text.replace("\n", "<br>")
    else:
        output_text = replace_newlines(output_text)
    # chatbot.append((input_text_box, output_text))
    chatbot[-1][1] = output_text

    return_status[methods_tabs.index(selected_tab)] = f"Generation time: {elapsed_time} seconds"
    return ("", chatbot, *return_status, translated_output_text)


@clear_cache_decorator
def user(message, history, translate_chk):
    # Append the user's message to the conversation history
    if len(message.strip()) > 0:
        if translate_chk:
            message = translate(message, "ja", "en")
            ollm_logging.info("Translated input text: " + message)
        return message, history + [[message, ""]]
    else:
        return message, history


@clear_cache_decorator
def translate_change(translate_chk):
    if translate_chk:
        load_translator()

    return "", "Translation enabled" if translate_chk else "Translation disabled"


@clear_cache_decorator
def change_model(ollm_model_id):
    if get_llm_class(ollm_model_id)().enable_rag_text:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def on_ui_tabs():
    ollm_model_ids = get_ollm_model_ids()
    ollm_model_index = ollm_model_ids.index("microsoft/Phi-3-mini-4k-instruct") \
        if "microsoft/Phi-3-mini-4k-instruct" in ollm_model_ids else 0

    cpp_ollm_model_ids = get_cpp_ollm_model_ids()
    cpp_ollm_model_index = cpp_ollm_model_ids.index("Phi-3-mini-4k-instruct-q4.gguf") \
        if "Phi-3-mini-4k-instruct-q4.gguf" in cpp_ollm_model_ids else 0

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
                    with gr.Tabs(elem_id="methods_tabs", selected=selected_tab):
                        with gr.TabItem(methods_tabs[0], elem_id=(methods_tabs[0]+"_tab"), id=methods_tabs[0]) as first_tab:
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
                        with gr.TabItem(methods_tabs[1], elem_id=(methods_tabs[1]+"_tab"), id=methods_tabs[1]) as second_tab:
                            with gr.Row():
                                with gr.Column():
                                    with gr.Row():
                                        cpp_ollm_model_id = gr.Dropdown(label="LLM model ID", elem_id="cpp_ollm_model_id", choices=cpp_ollm_model_ids,
                                                                        value=cpp_ollm_model_ids[cpp_ollm_model_index], show_label=True)
                                with gr.Column():
                                    with gr.Row():
                                        cpp_download_model_btn = gr.Button("Download model", elem_id="cpp_download_model_btn")
                                    with gr.Row():
                                        cpp_status_text = gr.Textbox(label="", max_lines=1, show_label=False, interactive=False)
                            with gr.Row():
                                with gr.Column():
                                    cpp_chat_template = gr.Radio(label="Default chat template", elem_id="cpp_chat_template",
                                                                 choices=["Llama2", "Llama3", "Gemma", "Phi-3"], value="Llama2", type="value")

                with gr.Row():
                    input_text_box = gr.Textbox(
                        label="Input text",
                        placeholder="Send a message",
                        show_label=True,
                    )
                with gr.Row():
                    rag_text_box = gr.Textbox(
                        label="Context document for ChatQA model",
                        placeholder="Context document",
                        show_label=True,
                        visible=False,
                    )
                with gr.Row():
                    max_new_tokens = gr.Slider(minimum=1, maximum=4096, step=1, value=256, label="Max new tokens", elem_id="max_new_tokens")
                with gr.Row():
                    with gr.Accordion("Advanced options", open=False):
                        temperature = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, value=0.7, label="Temperature", elem_id="temperature")
                        top_k = gr.Slider(minimum=1, maximum=200, step=1, value=50, label="Top k", elem_id="top_k")
                        top_p = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, value=1.0, label="Top p", elem_id="top_p")
                        repetition_penalty = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=1.1, label="Repetition penalty", elem_id="repetition_penalty")
                        translate_chk = gr.Checkbox(label="Translate (ja->en/en->ja)", elem_id="translate_chk", value=False, show_label=True)
                with gr.Row():
                    generate_btn = gr.Button("Generate", elem_id="generate_btn")
                with gr.Row():
                    translated_output_text = gr.Textbox(label="Translated output text", show_label=True, lines=2, interactive=False)
                with gr.Row():
                    clear_btn = gr.Button("Clear text", elem_id="clear_btn")

            download_model_btn.click(fn=TransformersLLM.download_model, inputs=[ollm_model_id], outputs=[status_text])
            translate_chk.change(fn=translate_change, inputs=[translate_chk], outputs=[input_text_box, status_text])
            ollm_model_id.change(fn=change_model, inputs=[ollm_model_id], outputs=[rag_text_box])

            cpp_download_model_btn.click(fn=LlamaCPPLLM.download_model, inputs=[cpp_ollm_model_id], outputs=[cpp_status_text])
            tabs_fn_map = {first_tab: change_tab_first, second_tab: change_tab_second}
            for tab, fn in tabs_fn_map.items():
                tab.select(fn=fn, inputs=None, outputs=None)

            generate_inputs = [chatbot, ollm_model_id, cpp_ollm_model_id, cpp_chat_template, input_text_box, rag_text_box,
                               max_new_tokens, temperature, top_k, top_p, repetition_penalty, translate_chk, cpu_execution_chk]
            generate_btn.click(fn=user, inputs=[input_text_box, chatbot, translate_chk], outputs=[input_text_box, chatbot]).then(
                fn=ollm_inference, inputs=generate_inputs, outputs=[input_text_box, chatbot, status_text, cpp_status_text, translated_output_text])
            input_text_box.submit(fn=user, inputs=[input_text_box, chatbot, translate_chk], outputs=[input_text_box, chatbot]).then(
                fn=ollm_inference, inputs=generate_inputs, outputs=[input_text_box, chatbot, status_text, cpp_status_text, translated_output_text])

            clear_btn.click(lambda: [None, None], None, [input_text_box, chatbot])

    return [(ollm_interface, "Open LLM", "open_llm")]


block, _, _ = on_ui_tabs()[0]
block.launch()
