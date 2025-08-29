from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# #Summarization
cliente = InferenceClient()

load_dotenv()
texto = """
    Greek (Modern Greek: Ελληνικά, romanized: Elliniká, [eliniˈka]; Ancient Greek: Ἑλληνική, romanized: Hellēnikḗ) is an Indo-European language, constituting an independent Hellenic branch within the Indo-European language family. It is native to Greece, Cyprus, Italy (in Calabria and Salento), southern Albania, and other regions of the Balkans, Caucasus, the Black Sea coast, Asia Minor, and the Eastern Mediterranean. It has the longest documented history of any Indo-European language, spanning at least 3,400 years of written records.[10] Its writing system is the Greek alphabet, which has been used for approximately 2,800 years;[11][12] previously, Greek was recorded in writing systems such as Linear B and the Cypriot syllabary.[13] The alphabet arose from the Phoenician script and was in turn the basis of the Latin, Cyrillic, Coptic, Gothic, and many other writing systems.

    The Greek language holds a very important place in the history of the Western world. Beginning with the epics of Homer, ancient Greek literature includes many works of lasting importance in the European canon. Greek is also the language in which many of the foundational texts in science and philosophy were originally composed. The New Testament of the Christian Bible was also originally written in Greek.[14][15] Together with the Latin texts and traditions of the Roman world, the Greek texts and Greek societies of antiquity constitute the objects of study of the discipline of Classics.

    During antiquity, Greek was by far the most widely spoken lingua franca in the Mediterranean world.[16] It eventually became the official language of the Byzantine Empire and developed into Medieval Greek.[17] In its modern form, Greek is the official language of Greece and Cyprus and one of the 24 official languages of the European Union. It is spoken by at least 13.5 million people today in Greece, Cyprus, Italy, Albania, Turkey, and the many other countries of the Greek diaspora.

    Greek roots have been widely used for centuries and continue to be widely used to coin new words in other languages; Greek and Latin are the predominant sources of international scientific vocabulary.
"""

resposta = cliente.summarization(texto, model="facebook/bart-large-cnn")
print(resposta)


# #text_completion
load_dotenv()

prompt = [{"role": "user", "content": "What is Python programming language?"}]
cliente = InferenceClient(model="meta-llama/Llama-3.2-3B-Instruct")

resposta = cliente.chat_completion(prompt, max_tokens=100)

print(resposta)

load_dotenv()

cliente = InferenceClient(model="meta-llama/Llama-3.2-3B-Instruct")

mensagens = [
    {"role": "system", "content": "Responda as perguntas de forma correta e precisa."},
]

while True:
    prompt = input("Digite a sua mensagem: ")
    mensagem_usuario = {"role": "user", "content": prompt}
    mensagens.append(mensagem_usuario)

    resposta = cliente.chat_completion(mensagens)
    resposta_ia = resposta.choices[0]
    role_resposta = resposta_ia.message.role
    content_resposta = resposta_ia.message.content

    dicionario_resposta_ia = {"role": role_resposta, "content": content_resposta}
    mensagens.append(dicionario_resposta_ia)
    print("IA: ", content_resposta)