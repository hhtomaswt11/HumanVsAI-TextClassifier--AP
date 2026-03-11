import os
import time
import pandas as pd
import random
import requests
from dotenv import load_dotenv

load_dotenv()
from groq import Groq
import google.generativeai as genai


GROQ_API_KEY    = os.environ["GROQ_API_KEY"]
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]

groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemma-3-27b-it")

# 2. Funções de Geração 
def gerar_groq(modelo, texto_original):
    prompt = (
        "You are an expert scientist. Rewrite the following scientific text in your own words. "
        "The new text MUST be strictly between 80 and 160 words long. "
        "Maintain a formal, scientific and encyclopedia-style tone. "
        "Do not include titles, introductory remarks, concluding remarks, or the word count. "
        f"Original text to rewrite:\n{texto_original}"
    )
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=modelo,
            temperature=0.7,
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Erro Groq: {e}")
        return None

def gerar_mistral(texto_original):
    prompt = (
        "You are an expert scientist. Rewrite the following scientific text in your own words. "
        "The new text MUST be strictly between 80 and 160 words long. "
        "Maintain a formal, scientific and encyclopedia-style tone. "
        "Do not include titles, introductory remarks, concluding remarks, or the word count. "
        f"Original text to rewrite:\n{texto_original}"
    )
    
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }
    data = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 250
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            print(f"Erro Mistral (Status {response.status_code}): {response.text}")
            return None
    except Exception as e:
        print(f"Erro Mistral Python: {e}")
        return None

def truncar_para_limite(texto, max_palavras=160):
    """Se o texto tiver mais de max_palavras, corta na última frase que caiba."""
    palavras = texto.split()
    if len(palavras) <= max_palavras:
        return texto
    texto_cortado = ' '.join(palavras[:max_palavras])
    for sep in ['. ', '! ', '? ']:
        ultimo = texto_cortado.rfind(sep)
        if ultimo != -1:
            return texto_cortado[:ultimo + 1].strip()
    return texto_cortado.strip()

def gerar_gemma(texto_original):
    prompt = (
        "You are an expert scientist. Write a single paragraph that rewrites the following scientific text in your own words. "
        "CRITICAL RULES: (1) Write ONLY ONE paragraph with NO line breaks. "
        "(2) The paragraph MUST contain between 80 and 120 words — count carefully. "
        "(3) Use a formal, scientific, encyclopedia-style tone. "
        "(4) Do NOT include titles, headers, bullet points, word count, or any commentary. "
        "(5) Output ONLY the rewritten paragraph, nothing else.\n\n"
        f"Text to rewrite:\n{texto_original}"
    )
    try:
        response = gemini_model.generate_content(prompt)
        texto = response.text.strip()
        primeiro_paragrafo = texto.split('\n\n')[0].replace('\n', ' ').strip()
        return truncar_para_limite(primeiro_paragrafo, max_palavras=160)
    except Exception as e:
        print(f"Erro Gemini: {e}")
        return None

modelos_para_gerar = {
    "Meta":    "llama-3.1-8b-instant",
    "Mistral": "mistral-small-latest",
    "Google":  "gemma-3-27b-it",
}

OBJETIVO_POR_MODELO = 600

NOME_FICHEIRO = "dataset_human_openai_pronto.csv" 
print(f"A carregar o ficheiro: {NOME_FICHEIRO}...")

try:
    df_base = pd.read_csv(NOME_FICHEIRO, sep=";")
    resultados = df_base.to_dict('records')
    print(f"Dataset carregado! Já tem {len(resultados)} linhas.")
    
    textos_humanos = [linha['Text'] for linha in resultados if linha['Label'] == 'Human']
    print(f"Encontrados {len(textos_humanos)} textos Humanos para usar como base!")

except FileNotFoundError:
    print("ERRO: Ficheiro base não encontrado!")
    exit()

print("\nA iniciar a reescrita de dados sintéticos...")

for label, nome_modelo in modelos_para_gerar.items():
    textos_validos = sum(1 for linha in resultados if linha['Label'] == label)
    
    print(f"\n--- A GERAR PARA A CLASSE: {label} ---")
    print(f"Já existem {textos_validos} textos desta IA no ficheiro. Faltam gerar {max(0, OBJETIVO_POR_MODELO - textos_validos)}.")

    erros_consecutivos = 0

    while textos_validos < OBJETIVO_POR_MODELO:
        
        texto_base_escolhido = random.choice(textos_humanos)

        if label == "Google":
            texto_gerado = gerar_gemma(texto_base_escolhido)
            time.sleep(2.5)  # Gemma 3: 30 RPM → ~2.5s entre requests
        elif label == "Mistral":
            texto_gerado = gerar_mistral(texto_base_escolhido)
            time.sleep(1.5)
        else:
            texto_gerado = gerar_groq(nome_modelo, texto_base_escolhido)
            time.sleep(1.5)

        if texto_gerado:
            print(f"\n--- TEXTO DA IA ({label}) ---")
            print(texto_gerado)
            print("--------------------------------")
            erros_consecutivos = 0
            word_count = len(texto_gerado.split())
            
            if 80 <= word_count <= 160:
                textos_validos += 1
                texto_limpo = texto_gerado.replace('\n', ' ').replace('\r', ' ')

                novo_id = f"ID-{len(resultados) + 1}"

                resultados.append({
                    "ID":    novo_id,
                    "Text":  texto_limpo,
                    "Label": label  
                })
                print(f"[{label} - {textos_validos}/{OBJETIVO_POR_MODELO}] Sucesso! ({word_count} palavras) Guardado como {novo_id}")

                pd.DataFrame(resultados).to_csv(NOME_FICHEIRO, sep=";", index=False)
        else:
            erros_consecutivos += 1
            print(f"Aviso: Erro ao gerar texto ({erros_consecutivos}/3)")
            time.sleep(5) 
            
            if erros_consecutivos >= 3:
                print(f"\n⚠️ LIMITE DIÁRIO ATINGIDO PARA {label}! A guardar progresso e a passar à próxima IA...")
                break 

print("\nGeração concluída! A guardar o ficheiro final...")
df_final = pd.DataFrame(resultados)
df_final.to_csv(NOME_FICHEIRO, sep=";", index=False)
print(f"Ficheiro {NOME_FICHEIRO} atualizado com sucesso. ESTÁ PRONTO PARA A TAREFA 2!")