import gradio as gr
from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc
import platform
from transformers import OpenLlamaModel, OpenLlamaConfig
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import TextIteratorStreamer, StoppingCriteriaList
from stablelm import StopOnTokens, start_message
from translator import load_translator, translate

_DOWNLOAD_COMPLETED = "Download complete"

model_cache = dict(
    preloaded_model_id=None,
    preloaded_model=None,
    preloaded_tokenizer=None,
    preloaded_streamer=None,
)

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
        "stabilityai/stablelm-tuned-alpha-3b",
        "stabilityai/stablelm-tuned-alpha-7b",
        "decapoda-research/llama-7b-hf",
        "decapoda-research/llama-13b-hf",
        ]
    return ollm_model_ids

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

def get_model_and_tokenizer_class(ollm_model_id):
    """Get model and tokenizer class.

    Args:
        ollm_model_id (str): String of Open LLM model ID.

    Returns:
        tuple(class, class): Tuple of model and tokenizer class.
    """
    if ("open-calm" in ollm_model_id or
        "japanese-gpt-neox" in ollm_model_id or
        "stablelm" in ollm_model_id):
        model_class = AutoModelForCausalLM
        tokenizer_class = AutoTokenizer
    elif "llama" in ollm_model_id:
        model_class = LlamaForCausalLM
        tokenizer_class = LlamaTokenizer
    else:
        model_class = AutoModelForCausalLM
        tokenizer_class = AutoTokenizer
    
    return model_class, tokenizer_class

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
    else:
        prompt = input_text_box
    
    return prompt

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
    
    return generate_kwargs

def retreive_output_text(input_text, output_text, ollm_model_id):
    """Retreive output text from generate method.

    Args:
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
    else:
        output_text = output_text
    
    return output_text

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def clear_cache():
    gc.collect()
    torch_gc()

def ollm_inference(chatbot, ollm_model_id, input_text_box, max_new_tokens, temperature, top_k, top_p, repetition_penalty, translate_chk):
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

    Returns:
        tuple(str, list, str): Input text, chatbot history, and inference result.
    """
    clear_cache()
    global model_cache
    
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
    
    model_class, tokenizer_class = get_model_and_tokenizer_class(ollm_model_id)
    
    print(f"Loading {ollm_model_id}")
    if (model_cache.get("preloaded_model_id") != ollm_model_id or 
        model_cache.get("preloaded_model") is None or 
        model_cache.get("preloaded_tokenizer") is None):
        
        for key in model_cache.keys():
            model_cache[key] = None
        clear_cache()
        
        if platform.system() == "Darwin":
            model = model_class.from_pretrained(ollm_model_id, torch_dtype=torch.float32)
        else:
            model = model_class.from_pretrained(ollm_model_id, device_map="auto", torch_dtype=torch.float16)
        
        model.tie_weights()

        tokenizer = tokenizer_class.from_pretrained(
            ollm_model_id,
            use_fast=False if "japanese-gpt-neox" in ollm_model_id else True,
        )
        
        model_cache["preloaded_model_id"] = ollm_model_id
        model_cache["preloaded_model"] = model
        model_cache["preloaded_tokenizer"] = tokenizer
    else:
        print("Using preloaded model")
        model = model_cache["preloaded_model"]
        tokenizer = model_cache["preloaded_tokenizer"]

    prompt = create_prompt(chatbot, ollm_model_id, input_text_box)

    print("Input text: " + prompt)
    print(f"Generating...")
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        add_special_tokens=False if "japanese-gpt-neox" in ollm_model_id else True,
    ).to(model.device)

    t1 = time.time()
    with torch.no_grad():
        tokens = model.generate(
            **get_generate_kwargs(tokenizer, inputs, ollm_model_id, generate_params)
        )
    t2 = time.time()
    elapsed_time = t2-t1
    print(f"Generation time: {elapsed_time} seconds")

    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    
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

def user(message, history, translate_chk):
    # Append the user's message to the conversation history
    if len(message.strip()) > 0:
        if translate_chk:
            message = translate(message, "ja", "en")
            print("Translated input text: " + message)
        return message, history + [[message, ""]]
    else:
        return message, history

def translate_change(translate_chk):
    if translate_chk:
        load_translator()
    
    return "", "Translation enabled" if translate_chk else "Translation disabled"

def on_ui_tabs():
    ollm_model_ids = get_ollm_model_ids()
    ollm_model_index = ollm_model_ids.index("rinna/japanese-gpt-neox-3.6b-instruction-ppo") if "rinna/japanese-gpt-neox-3.6b-instruction-ppo" in ollm_model_ids else 0

    block = gr.Blocks().queue()
    block.title = "Open LLM WebUI"
    with block as ollm_interface:
        with gr.Row():
            gr.Markdown("## Open LLM WebUI")
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(value=[], elem_id="chatbot", height=640)
            
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        ollm_model_id = gr.Dropdown(label="LLM model ID", elem_id="ollm_model_id", choices=ollm_model_ids,
                                                         value=ollm_model_ids[ollm_model_index], show_label=True)
                        translate_chk = gr.Checkbox(label="Translate (ja->en/en->ja)", elem_id="translate_chk", value=False, show_label=True)
                    with gr.Column():
                        with gr.Row():
                            download_model_btn = gr.Button("Download model", elem_id="download_model_btn")
                        with gr.Row():
                            status_text = gr.Textbox(label="", max_lines=1, show_label=False, interactive=False)
                
                input_text_box = gr.Textbox(
                    label="Input text",
                    placeholder="Send a message",
                    show_label=True,
                )
                
                max_new_tokens = gr.Slider(minimum=1, maximum=512, step=1, value=128, label="Max new tokens", elem_id="max_new_tokens")
                with gr.Accordion("Advanced options", open=False):
                    temperature = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.7, label="Temperature", elem_id="temperature")
                    top_k = gr.Slider(minimum=1, maximum=200, step=1, value=50, label="Top k", elem_id="top_k")
                    top_p = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=1.0, label="Top p", elem_id="top_p")
                    repetition_penalty = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=1.0, label="Repetition penalty", elem_id="repetition_penalty")
                
                generate_btn = gr.Button("Generate", elem_id="generate_btn")
                translated_output_text = gr.Textbox(label="Translated output text", show_label=True, lines=3, interactive=False)
                clear_btn = gr.Button("Clear text", elem_id="clear_btn")
            
            download_model_btn.click(fn=download_model, inputs=[ollm_model_id], outputs=[status_text])
            translate_chk.change(fn=translate_change, inputs=[translate_chk], outputs=[input_text_box, status_text])
            
            generate_inputs = [chatbot, ollm_model_id, input_text_box, max_new_tokens, temperature, top_k, top_p, repetition_penalty, translate_chk]
            generate_btn.click(fn=user, inputs=[input_text_box, chatbot, translate_chk], outputs=[input_text_box, chatbot]).then(
                fn=ollm_inference, inputs=generate_inputs, outputs=[input_text_box, chatbot, status_text, translated_output_text])
            input_text_box.submit(fn=user, inputs=[input_text_box, chatbot, translate_chk], outputs=[input_text_box, chatbot]).then(
                fn=ollm_inference, inputs=generate_inputs, outputs=[input_text_box, chatbot, status_text, translated_output_text])
            
            clear_btn.click(lambda: [None, None], None, [input_text_box, chatbot])
            
    return [(ollm_interface, "Open LLM", "open_llm")]

block, _, _ = on_ui_tabs()[0]
block.launch()
