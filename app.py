import gradio as gr
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import spaces
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id="mychen76/paligemma-3b-mix-448-med_30k-ct-brain"

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id,trust_remote_code=True).to(device).eval()
processor = PaliGemmaProcessor.from_pretrained(model_id)

MAX_TOKENS = 512

import re

def modify_caption(caption: str) -> str:
    """
    Removes specific prefixes from captions.

    Args:
        caption (str): A string containing a caption.

    Returns:
        str: The caption with the prefix removed if it was present.
    """
    # Define the prefixes to remove
    prefix_substrings = [
        ('captured from ', ''),
        ('captured at ', '')
    ]
    
    # Create a regex pattern to match any of the prefixes
    pattern = '|'.join([re.escape(opening) for opening, _ in prefix_substrings])
    replacers = {opening: replacer for opening, replacer in prefix_substrings}
    
    # Function to replace matched prefix with its corresponding replacement
    def replace_fn(match):
        return replacers[match.group(0)]
    
    # Apply the regex to the caption
    return re.sub(pattern, replace_fn, caption, count=1, flags=re.IGNORECASE)

# enable space
# @spaces.GPU
def create_captions_rich(image):   
    prompt = "caption"
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)

        # Modify the caption to remove specific prefixes
        modified_caption = modify_caption(decoded)
    return modified_caption

css = """
  #mkd {
    height: 500px; 
    overflow: auto; 
    border: 1px solid #ccc; 
  }
"""

with gr.Blocks(css=css) as demo:
  gr.HTML("<h1><center>PaliGemma Fine-tuned with 30k brain CT scan images<center><h1>")
  with gr.Tab(label="PaliGemma Med brain-CT-scan"):
    with gr.Row():
      with gr.Column():
        input_img = gr.Image(label="Input Picture")
        submit_btn = gr.Button(value="Submit")
      output = gr.Text(label="Caption")

    gr.Examples(
    [["brain_ct_image1.jpg"], ["brain_ct_image2.jpg"], ["brain_ct_image3.png"],["brain_ct_image4.png"]],
    inputs = [input_img],
    outputs = [output],
    fn=create_captions_rich,
    label='Try captioning on examples'
    )

    submit_btn.click(create_captions_rich, [input_img], [output])

demo.queue().launch(share=True,server_name="0.0.0.0",debug=True)