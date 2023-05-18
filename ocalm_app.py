import gradio as gr
from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc
import platform

def get_open_calm_model_ids():
    """Get Open CALM model IDs.

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
        ]
    return open_calm_model_ids

def download_model(open_calm_model_id, local_files_only=False):
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

def open_calm_inference(open_calm_model_id, input_text_box, max_new_tokens, temperature, top_k, top_p, repetition_penalty):
    clear_cache()
    
    dwonload_result = download_model(open_calm_model_id, local_files_only=True)
    if dwonload_result != "Download completed.":
        return "", dwonload_result
    
    if len(input_text_box.strip()) == 0:
        return "", "Input text is empty."
    
    print(f"Loading {open_calm_model_id}")
    if platform.system() == "Darwin":
        model = AutoModelForCausalLM.from_pretrained(open_calm_model_id, torch_dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(open_calm_model_id, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(open_calm_model_id)

    print(f"Generating...")
    inputs = tokenizer(input_text_box, return_tensors="pt").to(model.device)

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
        )
    t2 = time.time()
    elapsed_time = t2-t1
    print(f"Generation time: {elapsed_time} seconds")

    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print("Generation completed.")

    return output, f"Generation time: {elapsed_time} seconds"

def on_ui_tabs():
    open_calm_model_ids = get_open_calm_model_ids()

    block = gr.Blocks().queue()
    block.title = "Open CALM"
    with block as open_calm_interface:
        with gr.Row():
            gr.Markdown("## [Open CALM by CyberAgent, Inc.](https://huggingface.co/cyberagent)")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        open_calm_model_id = gr.Dropdown(label="Open CALM model ID", elem_id="open_calm_model_id", choices=open_calm_model_ids,
                                                         value=open_calm_model_ids[3], show_label=True)
                    with gr.Column():
                        with gr.Row():
                            download_model_btn = gr.Button("Download model", elem_id="download_model_btn")
                        with gr.Row():
                            status_text = gr.Textbox(label="", max_lines=1, show_label=False, interactive=False)
                
                input_text_box = gr.Textbox(label="Input text", elem_id="input_text_box", placeholder="Input text here", max_lines=16, show_label=True)
                max_new_tokens = gr.Slider(minimum=1, maximum=512, step=1, value=64, label="Max new tokens", elem_id="max_new_tokens")
                with gr.Accordion("Advanced options", open=False):
                    temperature = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.7, label="Temperature", elem_id="temperature")
                    top_k = gr.Slider(minimum=1, maximum=100, step=1, value=50, label="Top k", elem_id="top_k")
                    top_p = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=1.0, label="Top p", elem_id="top_p")
                    repetition_penalty = gr.Slider(minimum=1.0, maximum=10.0, step=0.5, value=1.0, label="Repetition penalty", elem_id="repetition_penalty")

                generate_btn = gr.Button("Generate", elem_id="generate_btn")
                output_text_box = gr.Textbox(label="Output text", elem_id="output_text_box", placeholder="Output text here", show_label=True, interactive=False)
            
            download_model_btn.click(download_model, inputs=[open_calm_model_id], outputs=[status_text])
            generate_inputs = [open_calm_model_id, input_text_box, max_new_tokens, temperature, top_k, top_p, repetition_penalty]
            generate_btn.click(open_calm_inference, inputs=generate_inputs, outputs=[output_text_box, status_text])

    return [(open_calm_interface, "Open CALM", "open_calm")]

block, _, _ = on_ui_tabs()[0]
block.launch()
