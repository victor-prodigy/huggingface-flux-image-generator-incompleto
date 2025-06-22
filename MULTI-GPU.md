# 1. Executar com paralelismo Use o CLI do Accelerate para ativar multi-GPU:
`accelerate launch --multi_gpu --num_processes=4 seu_script.py`

# 2. Otimizações para imagens Para modelos grandes (ex: SDXL):
- Sharding de modelo: Divida o modelo entre GPUs com device_map="auto":
`pipe = StableDiffusionPipeline.from_pretrained(model_id, device_map="auto")`

# 3. Solução para hardware limitado 
- Use DeepSpeed para otimização de memória:
`accelerate config --deepspeed`