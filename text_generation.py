from transformers import pipeline

modelo = pipeline("text-generation", model="ahxt/LiteLlama-460M-1T", device="cuda")

prompt = "What is python programming language?"

resposta = modelo(prompt)

print(resposta)