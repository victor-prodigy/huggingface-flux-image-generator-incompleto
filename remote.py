# https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev
from gradio_client import Client

client = Client("black-forest-labs/FLUX.1-dev")
result = client.predict(
		# prompt="A cat holding a sign that says hello world",
		prompt="The photo: Create a cinematic, photorealistic medium shot capturing the nostalgic warmth of a late 90s indie film. The focus is a young woman with brightly dyed pink-gold hair and freckled skin, looking directly and intently into the camera lens with a hopeful yet slightly uncertain smile, she is slightly off-center. She wears an oversized, vintage band t-shirt that says 'Lonuvel' (slightly worn) over a long-sleeved striped top and simple silver stud earrings. The lighting is soft, golden hour sunlight streaming through a slightly dusty window, creating lens flare and illuminating dust motes in the air. The background shows a blurred, cluttered bedroom with posters on the wall and fairy lights, rendered with a shallow depth of field. Natural film grain, a warm, slightly muted color palette, and sharp focus on her expressive eyes enhance the intimate, authentic feel",
		seed=0,
		randomize_seed=True,
		width=1024,
		height=1024,
		guidance_scale=3.5,
		num_inference_steps=28,
		api_name="/infer"
)
print(result)