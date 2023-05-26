import gradio as gr
from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc
import platform
from transformers import OpenLlamaModel, OpenLlamaConfig
from transformers import LlamaTokenizer, LlamaForCausalLM

def get_open_calm_model_ids():
    """Get Open CALM and Llama model IDs.

    Returns:
        list: List of Open CALM model IDs.
    """
    open_calm_model_ids = [
        "cyberagent/open-calm-small",
        "cyberagent/open-calm-medium",
        "cyberagent/open-calm-large",
        "cyberagent/open-calm-1b",
        "cyberagent/open-calm-3b",
        "cyberagent/open-calm-7b",
        "decapoda-research/llama-7b-hf",
        "decapoda-research/llama-13b-hf",
        "rinna/japanese-gpt-neox-3.6b",
        "rinna/japanese-gpt-neox-3.6b-instruction-sft",
        ]
    return open_calm_model_ids

def download_model(open_calm_model_id, local_files_only=False):
    """Download Open CALM and Llama models.

    Args:
        open_calm_model_id (str): String of Open CALM model ID.
        local_files_only (bool, optional): If True, use only local files. Defaults to False.

    Returns:
        str: string of download result.
    """
    if not local_files_only:
        print(f"Downloading {open_calm_model_id}")
    try:
        snapshot_download(repo_id=open_calm_model_id, local_files_only=local_files_only)
    except FileNotFoundError:
        return "Model not found. Please click Download model button."
    except Exception as e:
        return str(e)

    return "Download completed."

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def clear_cache():
    gc.collect()
    torch_gc()

def open_calm_inference(chatbot, open_calm_model_id, input_text_box, max_new_tokens, temperature, top_k, top_p, repetition_penalty):
    """Open CALM inference.

    Args:
        chatbot (list): Chatbot history.
        open_calm_model_id (str): String of Open CALM model ID.
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
    if input_text_box is None or len(input_text_box.strip()) == 0:
        return "", chatbot, "Input text is empty."
    
    chatbot = [] if chatbot is None else chatbot
    
    dwonload_result = download_model(open_calm_model_id, local_files_only=True)
    if dwonload_result != "Download completed.":
        return input_text_box, chatbot, dwonload_result
    
    if "open-calm" in open_calm_model_id:
        model_class = AutoModelForCausalLM
        tokenizer_class = AutoTokenizer
    elif "llama" in open_calm_model_id:
        model_class = LlamaForCausalLM
        tokenizer_class = LlamaTokenizer
    elif "japanese-gpt-neox" in open_calm_model_id:
        model_class = AutoModelForCausalLM
        tokenizer_class = AutoTokenizer
        
        if "instruction-sft" in open_calm_model_id:
            sft_input_text = []
            for user_text, system_text in chatbot:
                sft_input_text.append("ユーザー: " + user_text + "<NL>システム: " + system_text)

            if len(sft_input_text) > 0:
                sft_input_text = "<NL>".join(sft_input_text) + "<NL>ユーザー: " + input_text_box + "<NL>システム: "
            else:
                sft_input_text = "ユーザー: " + input_text_box + "<NL>システム: "
    
    print(f"Loading {open_calm_model_id}")
    if platform.system() == "Darwin":
        model = model_class.from_pretrained(open_calm_model_id, torch_dtype=torch.float32)
    else:
        model = model_class.from_pretrained(open_calm_model_id, device_map="auto", torch_dtype=torch.float16)
    
    tokenizer = tokenizer_class.from_pretrained(
        open_calm_model_id,
        use_fast=False if "japanese-gpt-neox" in open_calm_model_id else True,
    )

    print("Input text: " + (input_text_box if "instruction-sft" not in open_calm_model_id else sft_input_text))
    print(f"Generating...")
    inputs = tokenizer(
        input_text_box if "instruction-sft" not in open_calm_model_id else sft_input_text,
        return_tensors="pt",
        add_special_tokens=False if "japanese-gpt-neox" in open_calm_model_id else True,
    ).to(model.device)

    t1 = time.time()
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=float(repetition_penalty),
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    t2 = time.time()
    elapsed_time = t2-t1
    print(f"Generation time: {elapsed_time} seconds")

    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    if "instruction-sft" in open_calm_model_id:
        output = output.split("<NL>")[-1].replace("システム: ", "")

    print("Generation completed.")
    print("Output text: " + output)
    
    output = output.replace("\n", "<br>")
    chatbot.append((input_text_box, output))
    return "", chatbot, f"Generation time: {elapsed_time} seconds"

def on_ui_tabs():
    open_calm_model_ids = get_open_calm_model_ids()

    block = gr.Blocks().queue()
    block.title = "Open LLM WebUI"
    with block as open_calm_interface:
        with gr.Row():
            gr.Markdown("## Open LLM WebUI")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        open_calm_model_id = gr.Dropdown(label="LLM model ID", elem_id="open_calm_model_id", choices=open_calm_model_ids,
                                                         value=open_calm_model_ids[3], show_label=True)
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
                
                max_new_tokens = gr.Slider(minimum=1, maximum=512, step=1, value=64, label="Max new tokens", elem_id="max_new_tokens")
                with gr.Accordion("Advanced options", open=False):
                    temperature = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.7, label="Temperature", elem_id="temperature")
                    top_k = gr.Slider(minimum=1, maximum=100, step=1, value=50, label="Top k", elem_id="top_k")
                    top_p = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=1.0, label="Top p", elem_id="top_p")
                    repetition_penalty = gr.Slider(minimum=1.0, maximum=10.0, step=0.5, value=1.0, label="Repetition penalty", elem_id="repetition_penalty")

                generate_btn = gr.Button("Generate", elem_id="generate_btn")
                clear_btn = gr.Button("Clear text", elem_id="clear_btn")
            
            with gr.Column():
                chatbot = gr.Chatbot([], elem_id="chatbot").style(height=640)
            
            download_model_btn.click(download_model, inputs=[open_calm_model_id], outputs=[status_text])
            generate_inputs = [chatbot, open_calm_model_id, input_text_box, max_new_tokens, temperature, top_k, top_p, repetition_penalty]
            generate_btn.click(open_calm_inference, inputs=generate_inputs, outputs=[input_text_box, chatbot, status_text], queue=False)
            input_text_box.submit(open_calm_inference, inputs=generate_inputs, outputs=[input_text_box, chatbot, status_text], queue=False)
            
            clear_btn.click(lambda: [None, None], None, [input_text_box, chatbot], queue=False)
            
    return [(open_calm_interface, "Open CALM", "open_calm")]

block, _, _ = on_ui_tabs()[0]
block.launch()
