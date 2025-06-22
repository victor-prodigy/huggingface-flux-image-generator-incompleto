from accelerate import Accelerator
from diffusers import DiffusionPipeline
import torch

accelerator = Accelerator()

# Configurar modelo
model_id = "stabilityai/stable-diffusion-2-1"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to(accelerator.device)

# Paralelização automática
pipe, _ = accelerator.prepare(pipe, None)

# Gerar imagem
prompt = "A serene landscape with mountains and a river at sunset."
image = pipe(prompt).images[0]
image.save("output.png")  # Salvar a imagem gerada
