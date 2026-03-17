import os
import time
import pandas as pd
import random
import requests
from dotenv import load_dotenv

load_dotenv()
from groq import Groq
import google.generativeai as genai

CONTENT_TYPE_JSON = "application/json"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
gemini_model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemma-3-27b-it")

if not GROQ_API_KEY:
    print("Aviso: GROQ_API_KEY em falta. A classe 'Meta' será ignorada.")

if not GEMINI_API_KEY:
    print("Aviso: GEMINI_API_KEY em falta. A classe 'Google' será ignorada.")
if not ANTHROPIC_API_KEY:
    print("Aviso: ANTHROPIC_API_KEY em falta. A classe 'Anthropic' será ignorada.")

# 2. Funções de Geração 
def gerar_groq(modelo, texto_original):
    if not groq_client:
        return None
    prompt = (
        "You are an expert scientist. Rewrite the following scientific text in your own words. "
        "The new text MUST be strictly between 100 and 120 words long. "
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

def gerar_anthropic(texto_original):
    if not ANTHROPIC_API_KEY:
        return None
    prompt = (
        "You are an expert scientist. Rewrite the following scientific text in your own words. "
        "The new text MUST be strictly between 100 and 120 words long. "
        "Maintain a formal, scientific and encyclopedia-style tone. "
        "Do not include titles, introductory remarks, concluding remarks, or the word count. "
        f"Original text to rewrite:\n{texto_original}"
    )

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": CONTENT_TYPE_JSON,
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    data = {
        "model": "claude-haiku-4-5",
        "max_tokens": 250,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            content = response.json().get("content", [])
            if content and content[0].get("type") == "text":
                return content[0].get("text", "").strip()
            print(f"Erro Anthropic: resposta sem texto esperado - {response.text}")
            return None
        print(f"Erro Anthropic (Status {response.status_code}): {response.text}")
        return None
    except Exception as e:
        print(f"Erro Anthropic Python: {e}")
        return None

def truncar_para_limite(texto, max_palavras=120):
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
    if not gemini_model:
        return None
    prompt = (
        "You are an expert scientist. Write a single paragraph that rewrites the following scientific text in your own words. "
        "CRITICAL RULES: (1) Write ONLY ONE paragraph with NO line breaks. "
        "(2) The paragraph MUST contain between 100 and 120 words — count carefully. "
        "(3) Use a formal, scientific, encyclopedia-style tone. "
        "(4) Do NOT include titles, headers, bullet points, word count, or any commentary. "
        "(5) Output ONLY the rewritten paragraph, nothing else.\n\n"
        f"Text to rewrite:\n{texto_original}"
    )
    try:
        response = gemini_model.generate_content(prompt)
        texto = response.text.strip()
        primeiro_paragrafo = texto.split('\n\n')[0].replace('\n', ' ').strip()
        return truncar_para_limite(primeiro_paragrafo, max_palavras=120)
    except Exception as e:
        print(f"Erro Gemini: {e}")
        return None

modelos_para_gerar = {
    "Meta":    "llama-3.1-8b-instant",

    "Google":  "gemma-3-27b-it",
    "Anthropic": "claude-haiku-4-5",
}

OBJETIVO_POR_MODELO = 1000

NOME_FICHEIRO = "../data/dataset_limpo.csv" 
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

chave_necessaria_por_label = {
    "Meta": GROQ_API_KEY,
    "Google": GEMINI_API_KEY,
    "Anthropic": ANTHROPIC_API_KEY,
}


for label, nome_modelo in modelos_para_gerar.items():
    if not chave_necessaria_por_label.get(label):
        print(f"\n--- A IGNORAR A CLASSE: {label} (API key em falta) ---")
        continue

    textos_validos = sum(1 for linha in resultados if linha['Label'] == label and 100 <= len(str(linha['Text']).split()) <= 120)
    
    print(f"\n--- A GERAR PARA A CLASSE: {label} ---")
    print(f"Já existem {textos_validos} textos desta IA no ficheiro. Faltam gerar {max(0, OBJETIVO_POR_MODELO - textos_validos)}.")

    erros_consecutivos = 0

    while textos_validos < OBJETIVO_POR_MODELO:
        
        texto_base_escolhido = random.choice(textos_humanos)

        if label == "Google":
            texto_gerado = gerar_gemma(texto_base_escolhido)
            time.sleep(2.5)  # Gemma 3: 30 RPM → ~2.5s entre requests
        elif label == "Anthropic":
            texto_gerado = gerar_anthropic(texto_base_escolhido)
            time.sleep(2.0)
        else:
            texto_gerado = gerar_groq(nome_modelo, texto_base_escolhido)
            time.sleep(1.5)

        if texto_gerado:
            print(f"\n--- TEXTO DA IA ({label}) ---")
            print(texto_gerado)
            print("--------------------------------")
            erros_consecutivos = 0
            word_count = len(texto_gerado.split())
            
            if 100 <= word_count <= 120:
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
                print(f"\nLIMITE DIÁRIO ATINGIDO PARA {label}! A guardar progresso e a passar à próxima IA...")
                break 

print("\nGeração concluída! A guardar o ficheiro final...")
df_final = pd.DataFrame(resultados)
df_final.to_csv(NOME_FICHEIRO, sep=";", index=False)
print(f"Ficheiro {NOME_FICHEIRO} atualizado com sucesso. ESTÁ PRONTO PARA A TAREFA 2!")