import functools
import json
import os
import time
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning, module="gradio")  # noqa: E402

import gradio as gr
import torch

from cache_manager import ClearCacheContext, clear_cache, clear_cache_decorator, model_cache
from chat_utils import replace_newlines_code_blocks
from custom_logging import ollm_logging
from model_manager.base import LLMConfig, ensure_tensor_on_device
from model_manager.llama_cpp import LlamaCPPLLM, get_chat_templates_keys, get_cpp_ollm_model_ids
from model_manager.llava import LlavaLLM, get_llava_ollm_model_ids
from model_manager.tformers import TransformersLLM, get_ollm_model_ids
from registry import get_llm_class
from translator import load_translator, translate

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

methods_tabs = ["transformers", "llama.cpp", "LLaVA"]
selected_tab = methods_tabs[0]


@clear_cache_decorator
def change_tab(tab_num):
    global selected_tab
    selected_tab = methods_tabs[tab_num]
    ollm_logging.debug(f"Selected tab: {selected_tab}")
    return None


change_tab_first = functools.partial(change_tab, tab_num=0)
change_tab_second = functools.partial(change_tab, tab_num=1)
change_tab_third = functools.partial(change_tab, tab_num=2)


def _message_content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                    continue
                if isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return " ".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
        return ""
    return str(content)


def _pairs_to_messages(chatbot_pairs):
    messages = []
    for user_text, assistant_text in chatbot_pairs:
        messages.append({"role": "user", "content": _message_content_to_text(user_text)})
        messages.append({"role": "assistant", "content": _message_content_to_text(assistant_text)})
    return messages


def _messages_to_pairs(chatbot_messages):
    if not chatbot_messages:
        return []
    first_item = chatbot_messages[0]
    if isinstance(first_item, (list, tuple)) and len(first_item) == 2:
        return [list(item) for item in chatbot_messages]

    pairs = []
    pending_user = None
    for message in chatbot_messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = _message_content_to_text(message.get("content", ""))
        if role == "user":
            if pending_user is not None:
                pairs.append([pending_user, ""])
            pending_user = content
        elif role == "assistant":
            if pending_user is None:
                pairs.append(["", content])
            else:
                pairs.append([pending_user, content])
                pending_user = None
    if pending_user is not None:
        pairs.append([pending_user, ""])
    return pairs


def _normalize_chatbot_messages(chatbot):
    if not chatbot:
        return []
    if isinstance(chatbot[0], dict):
        return chatbot
    return _pairs_to_messages(_messages_to_pairs(chatbot))


@clear_cache_decorator
def ollm_inference(chatbot, ollm_model_id, cpp_ollm_model_id, llava_ollm_model_id, cpp_chat_template, input_text_box, rag_text_box,
                   llava_image, max_new_tokens, temperature, top_k, top_p, repetition_penalty, translate_chk,
                   cpu_execution_chk=False, llava_cpu_execution_chk=False, replace_system=False, system_message=""):
    """Open LLM inference.

    Args:
        chatbot (list): Chatbot history.
        ollm_model_id (str): String of Open LLM model ID.
        cpp_ollm_model_id (str): String of Open LLM model ID for llama.cpp.
        llava_ollm_model_id (str): String of Open LLM model ID for LLaVA.
        cpp_chat_template (str): String of chat template for llama.cpp.
        input_text_box (str): Input text.
        rag_text_box (str): Context document for ChatQA model.
        llava_image (PIL.Image): Image for LLaVA.
        max_new_tokens (int): Parameter for generate method.
        temperature (float): Parameter for generate method.
        top_k (int): Parameter for generate method.
        top_p (float): Parameter for generate method.
        repetition_penalty (float): Parameter for generate method.
        translate_chk (bool): If True, translate output text.
        cpu_execution_chk (bool, optional): If True, use CPU. Defaults to False.
        llava_cpu_execution_chk (bool, optional): If True, use CPU for LLaVA. Defaults to False.

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

    return_status = [""] * len(methods_tabs)
    if input_text_box is None or len(input_text_box.strip()) == 0:
        return_status[methods_tabs.index(selected_tab)] = "Input text is empty."
        return ("", chatbot, *return_status, "")

    chatbot = _normalize_chatbot_messages(chatbot)
    chatbot_pairs = _messages_to_pairs(chatbot)

    if selected_tab == methods_tabs[0]:
        method_class = TransformersLLM
    elif selected_tab == methods_tabs[1]:
        method_class = LlamaCPPLLM
        ollm_model_id = cpp_ollm_model_id
    elif selected_tab == methods_tabs[2]:
        method_class = LlavaLLM
        ollm_model_id = llava_ollm_model_id
        cpu_execution_chk = llava_cpu_execution_chk
        if llava_image is None:
            ollm_logging.error("LLaVA image is not exist.")
            return_status[methods_tabs.index(selected_tab)] = "LLaVA image is not exist."
            return (input_text_box, chatbot, *return_status, "")
    else:
        return_status[methods_tabs.index(selected_tab)] = "Selected tab is not supported."
        return ("", chatbot, *return_status, "")

    dwonload_result = method_class.download_model(ollm_model_id, local_files_only=True)
    ollm_logging.debug(f"Download result: {dwonload_result}")
    if dwonload_result != LLMConfig.DOWNLOAD_COMPLETE:
        return_status[methods_tabs.index(selected_tab)] = dwonload_result
        return (input_text_box, chatbot, *return_status, "")

    model_params = method_class.get_llm_instance(ollm_model_id, cpu_execution_chk)

    if replace_system:
        model_params.system_message = system_message

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

    if selected_tab == methods_tabs[1] and hasattr(model_params, "prepare_tokenizer"):
        tokenizer = model_params.prepare_tokenizer(tokenizer, model, cpp_chat_template)

    def extract_prompts_from_json(json_file_path):
        prompt_list = []

        try:
            with open(json_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "prompt" in item:
                        prompt_list.append(item["prompt"])
            else:
                ollm_logging.error("The top level of the JSON file is not an array.")
        except Exception as e:
            ollm_logging.error(f"Error reading JSON file: {e}")

        return prompt_list

    enable_loop = False
    if input_text_box == "input_prompts.json" and os.path.isfile("input_prompts.json"):
        enable_loop = True
        model_outputs = []

        prompts = extract_prompts_from_json(input_text_box)
    else:
        prompts = [input_text_box]

    for input_text in prompts:
        if enable_loop:
            if chatbot_pairs:
                chatbot_pairs[-1][0] = input_text
                chatbot_pairs[-1][1] = ""
                chatbot_pairs = [chatbot_pairs[-1]]
            else:
                chatbot_pairs = [[input_text, ""]]

        prompt = model_params.create_prompt(chatbot_pairs, ollm_model_id, input_text, rag_text_box, tokenizer)

        ollm_logging.info(f"Input text: {prompt}")
        ollm_logging.info("Generating...")
        if model_params.multimodal_image:
            with ClearCacheContext():
                if model_params.image_processor_class is None:
                    if hasattr(model, "chat"):
                        inputs = model.prepare_chat(
                            llava_image,
                            [prompt],
                            tokenizer,
                            **model_params.tokenizer_input_kwargs,
                        )
                    elif isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
                        image_message = {
                            "type": "image",
                            "image": llava_image,
                        }
                        for message in prompt:
                            if message.get("role") == "user":
                                message["content"].insert(0, image_message)
                                break
                        inputs = tokenizer.apply_chat_template(
                            prompt, **model_params.tokenizer_input_kwargs)
                    else:
                        input_args = [
                            [prompt] if model_params.imagep_config.get("prompt_is_list") else prompt,
                            llava_image if not model_params.imagep_config.get("image_is_list") else [llava_image],
                        ]
                        if model_params.imagep_config.get("image_is_first"):
                            input_args = input_args[::-1]
                        inputs = tokenizer(
                            *input_args,
                            **model_params.tokenizer_input_kwargs,
                        )
                        # ollm_logging.debug(f"Tokenizer class: {tokenizer.tokenizer.__class__.__name__}")
                        # ollm_logging.debug(f"Processor class: {tokenizer.image_processor.__class__.__name__}")
                elif hasattr(tokenizer, "image_processor"):
                    input_ids = LlavaLLM.tokenizer_image_token(
                        prompt,
                        tokenizer,
                        **model_params.tokenizer_input_kwargs,
                    ).unsqueeze(0)
                    images = tokenizer.image_processor(
                        llava_image,
                        **model_params.image_processor_input_kwargs,
                    )["pixel_values"]
                    # ollm_logging.debug(f"Tokenizer class: {tokenizer.__class__.__name__}")
                    # ollm_logging.debug(f"Processor class: {tokenizer.image_processor.__class__.__name__}")
                    inputs = dict(inputs=input_ids, images=images)
                else:
                    ollm_logging.error("Image processor is not exist.")
                    return_status[methods_tabs.index(selected_tab)] = "Image processor is not exist."
                    return (input_text, chatbot, *return_status, "")
            if hasattr(model, "device"):
                inputs = ensure_tensor_on_device(inputs, model.device)
        elif model_params.require_tokenization:
            with ClearCacheContext():
                inputs = tokenizer(
                    [prompt],
                    **model_params.tokenizer_input_kwargs,
                )
            if hasattr(model, "device"):
                inputs = ensure_tensor_on_device(inputs, model.device)
        else:
            inputs = prompt

        generate_kwargs = model_params.get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        if hasattr(model_params, "reset_model") and callable(model_params.reset_model):
            model_params.reset_model(model)

        t1 = time.time()
        with ClearCacheContext(), torch.no_grad():
            tokens = getattr(model, model_params.model_generate_name)(**generate_kwargs)
        t2 = time.time()
        elapsed_time = t2-t1
        ollm_logging.info(f"Generation time: {elapsed_time} seconds")

        if model_params.multimodal_image and model_params.image_processor_class is not None:
            with ClearCacheContext():
                output_text = tokenizer.batch_decode(tokens, **model_params.tokenizer_decode_kwargs)[0].strip()
        elif model_params.require_tokenization:
            input_ids = generate_kwargs["input_ids"]
            with ClearCacheContext():
                output_text = tokenizer.decode(
                    tokens[0] if not model_params.output_text_only else tokens[0][len(input_ids[0]):],
                    **model_params.tokenizer_decode_kwargs,
                )
        else:
            output_text = tokens

        output_text = model_params.retreive_output_text(prompt, output_text, ollm_model_id, tokenizer)

        ollm_logging.info("Generation complete")
        ollm_logging.info("Output text: " + output_text)

        if enable_loop:
            model_outputs.append({"prompt": input_text, "output": output_text})

    def write_outputs_to_file(json_data, json_file_path):
        try:
            with open(json_file_path, "w", encoding="utf-8") as file:
                json.dump(json_data, file, ensure_ascii=False, indent=4)
            return f"The JSON file was saved: {json_file_path}"
        except Exception as e:
            return f"Error writing JSON file: {e}"

    if enable_loop and len(model_outputs) > 0:
        if chatbot_pairs:
            chatbot_pairs[-1][0] = f"The JSON file was loaded: {input_text_box}"
        else:
            chatbot_pairs = [[f"The JSON file was loaded: {input_text_box}", ""]]
        output_text = write_outputs_to_file(model_outputs, "model_outputs.json")

    if translate_chk:
        translated_output_text = translate(output_text, "en", "ja")
        ollm_logging.info("Translated output text: " + translated_output_text)
    else:
        translated_output_text = ""

    if "```" not in output_text:
        output_text = output_text.replace("\n", "<br>")
    else:
        output_text = replace_newlines_code_blocks(output_text)
        ollm_logging.debug(output_text)
    # chatbot.append((input_text_box, output_text))
    if chatbot_pairs:
        chatbot_pairs[-1][1] = output_text
    else:
        chatbot_pairs = [[input_text_box, output_text]]
    chatbot = _pairs_to_messages(chatbot_pairs)

    return_status[methods_tabs.index(selected_tab)] = f"Generation time: {elapsed_time} seconds"
    return ("", chatbot, *return_status, translated_output_text)


@clear_cache_decorator
def user(message, history, translate_chk):
    # Append the user's message to the conversation history
    history = _normalize_chatbot_messages(history)
    if len(message.strip()) > 0:
        if translate_chk:
            message = translate(message, "ja", "en")
            ollm_logging.info("Translated input text: " + message)
        return message, history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    else:
        return message, history


@clear_cache_decorator
def translate_change(translate_chk):
    if translate_chk:
        load_translator()
        gr_update = gr.update(visible=True)
    else:
        gr_update = gr.update(visible=False)

    ret_status = ["Translation enable" if translate_chk else "Translation disable"] * len(methods_tabs)
    return ("", gr_update, *ret_status)


@clear_cache_decorator
def change_model(ollm_model_id):
    if get_llm_class(ollm_model_id).enable_rag_text:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def on_ui_tabs():
    ollm_model_ids = get_ollm_model_ids()
    ollm_model_index = 0

    cpp_ollm_model_ids = get_cpp_ollm_model_ids()
    cpp_ollm_model_index = 0

    llava_ollm_model_ids = get_llava_ollm_model_ids()
    llava_ollm_model_index = 0

    block = gr.Blocks().queue()
    block.title = "Open LLM WebUI"
    with block as ollm_interface:
        with gr.Row():
            gr.Markdown("## Open LLM WebUI")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    chatbot = gr.Chatbot(value=[], elem_id="chatbot", height=640)
                with gr.Row():
                    clear_btn = gr.Button("Clear chat", elem_id="clear_btn")

            with gr.Column():
                with gr.Row():
                    with gr.Tabs(elem_id="methods_tabs", selected=selected_tab):
                        with gr.TabItem(methods_tabs[0], elem_id=(methods_tabs[0]+"_tab"), id=methods_tabs[0]) as first_tab:
                            with gr.Row():
                                with gr.Column():
                                    with gr.Row():
                                        ollm_model_id = gr.Dropdown(
                                            label="LLM model ID", elem_id="ollm_model_id", choices=ollm_model_ids,
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
                                        cpp_ollm_model_id = gr.Dropdown(
                                            label="LLM model ID (models folder included)", elem_id="cpp_ollm_model_id",
                                            choices=cpp_ollm_model_ids,
                                            value=cpp_ollm_model_ids[cpp_ollm_model_index], show_label=True)
                                with gr.Column():
                                    with gr.Row():
                                        cpp_download_model_btn = gr.Button("Download model", elem_id="cpp_download_model_btn")
                                    with gr.Row():
                                        cpp_status_text = gr.Textbox(label="", max_lines=1, show_label=False, interactive=False)
                            with gr.Row():
                                with gr.Column():
                                    cpp_chat_template = gr.Radio(label="Default chat template (when GGUF file is missing template)",
                                                                 elem_id="cpp_chat_template",
                                                                 choices=get_chat_templates_keys(),
                                                                 value="Llama2", type="value")
                        with gr.TabItem(methods_tabs[2], elem_id=(methods_tabs[2]+"_tab"), id=methods_tabs[2]) as third_tab:
                            with gr.Row():
                                with gr.Column():
                                    with gr.Row():
                                        llava_ollm_model_id = gr.Dropdown(
                                            label="LLM model ID", elem_id="llava_ollm_model_id",
                                            choices=llava_ollm_model_ids,
                                            value=llava_ollm_model_ids[llava_ollm_model_index], show_label=True)
                                    with gr.Row():
                                        llava_cpu_execution_chk = gr.Checkbox(
                                            label="CPU execution", elem_id="llava_cpu_execution_chk", value=False, show_label=True)
                                with gr.Column():
                                    with gr.Row():
                                        llava_download_model_btn = gr.Button("Download model", elem_id="llava_download_model_btn")
                                    with gr.Row():
                                        llava_status_text = gr.Textbox(label="", max_lines=1, show_label=False, interactive=False)
                            with gr.Row():
                                llava_image = gr.Image(label="LLaVA Image", elem_id="llava_image", sources=["upload"], type="pil",
                                                       interactive=True, height=240)

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
                        value="",
                        show_label=True,
                        visible=False,
                    )
                with gr.Row():
                    generate_btn = gr.Button("Generate", elem_id="generate_btn", variant="primary")
                with gr.Row():
                    max_new_tokens = gr.Slider(minimum=1, maximum=4096, step=1, value=512, label="Max new tokens", elem_id="max_new_tokens")
                with gr.Row():
                    with gr.Accordion("Advanced options", open=False):
                        with gr.Row():
                            with gr.Column(scale=0, min_width=160):
                                replace_system = gr.Checkbox(label="Replace system message", elem_id="replace_system_message",
                                                             value=False, show_label=True)
                            with gr.Column():
                                system_message = gr.Textbox(label="System message", placeholder="System message", value="",
                                                            show_label=True)
                        temperature = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, value=0.7, label="Temperature", elem_id="temperature")
                        top_k = gr.Slider(minimum=1, maximum=200, step=1, value=50, label="Top k", elem_id="top_k")
                        top_p = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, value=1.0, label="Top p", elem_id="top_p")
                        repetition_penalty = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=1.1, label="Repetition penalty", elem_id="repetition_penalty")
                        translate_chk = gr.Checkbox(label="Translate (ja->en/en->ja)", elem_id="translate_chk", value=False, show_label=True)
                with gr.Row():
                    translated_output_text = gr.Textbox(
                        label="Translated output text", show_label=True, lines=1, interactive=False, visible=False)

            status_text_boxes = [status_text, cpp_status_text, llava_status_text]
            assert len(status_text_boxes) == len(methods_tabs), "Status text boxes length is not equal to methods tabs length"
            download_model_btn.click(fn=TransformersLLM.download_model, inputs=[ollm_model_id], outputs=[status_text])
            translate_chk.change(fn=translate_change, inputs=[translate_chk],
                                 outputs=[input_text_box, translated_output_text] + status_text_boxes)
            ollm_model_id.change(fn=change_model, inputs=[ollm_model_id], outputs=[rag_text_box])

            cpp_download_model_btn.click(fn=LlamaCPPLLM.download_model, inputs=[cpp_ollm_model_id], outputs=[cpp_status_text])
            llava_download_model_btn.click(fn=LlavaLLM.download_model, inputs=[llava_ollm_model_id], outputs=[llava_status_text])
            tabs_fn_map = {first_tab: change_tab_first, second_tab: change_tab_second, third_tab: change_tab_third}
            for tab, fn in tabs_fn_map.items():
                tab.select(fn=fn, inputs=None, outputs=None)

            generate_inputs = [
                chatbot, ollm_model_id, cpp_ollm_model_id, llava_ollm_model_id, cpp_chat_template, input_text_box, rag_text_box,
                llava_image, max_new_tokens, temperature, top_k, top_p, repetition_penalty, translate_chk,
                cpu_execution_chk, llava_cpu_execution_chk, replace_system, system_message]
            inference_outputs = [input_text_box, chatbot] + status_text_boxes + [translated_output_text]
            generate_btn.click(
                fn=user, inputs=[input_text_box, chatbot, translate_chk], outputs=[input_text_box, chatbot]
            ).then(fn=ollm_inference, inputs=generate_inputs, outputs=inference_outputs)
            input_text_box.submit(
                fn=user, inputs=[input_text_box, chatbot, translate_chk],
                outputs=[input_text_box, chatbot]
            ).then(fn=ollm_inference, inputs=generate_inputs, outputs=inference_outputs)

            clear_btn.click(lambda: [], None, [chatbot])

    return [(ollm_interface, "Open LLM", "open_llm")]


block, _, _ = on_ui_tabs()[0]
block.launch()
