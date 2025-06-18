# https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev
from gradio_client import Client

client = Client("black-forest-labs/FLUX.1-dev")
result = client.predict(
		prompt="A cat holding a sign that says hello world",
		seed=0,
		randomize_seed=True,
		width=1024,
		height=1024,
		guidance_scale=3.5,
		num_inference_steps=28,
		api_name="/infer"
)
print(result)