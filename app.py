from diffusers import AutoPipelineForText2Image
import torch
from datetime import datetime
import os

print("Loading model...")

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
)

pipe = pipe.to("cuda")

prompt = """
futuristic electric vehicle dashboard,
industrial engineering concept design,
high-tech interior,
minimalist automotive UI,
premium materials,
cinematic lighting,
ultra detailed
"""

print("Generating image...")

image = pipe(
    prompt=prompt,
    num_inference_steps=2,
    guidance_scale=0.0
).images[0]

# Create folder if not exists
os.makedirs("generated_outputs", exist_ok=True)

# Unique filename using timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

filename = f"generated_outputs/engineering_concept_{timestamp}.png"

image.save(filename)

print(f"Image saved as: {filename}")