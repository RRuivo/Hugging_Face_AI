from diffusers import StableDiffusionPipeline
import torch

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a cute purple cow with a big smile"
resposta = pipe(prompt)
print(resposta)
imagem = resposta.images[0]
    
imagem.save("image.png")