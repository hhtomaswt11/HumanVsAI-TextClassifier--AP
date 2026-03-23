import os
import time
import csv
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_KEY")

if not GROQ_API_KEY:
    print("Erro: GROQ_KEY não encontrada no ficheiro .env ou no ambiente.")
    exit(1)

client = Groq(api_key=GROQ_API_KEY)
# Using the same model as in previous script but updated to the client structure
MODEL_NAME = "llama-3.1-8b-instant"

def truncar_para_limite(texto, max_palavras=160):
    palavras = texto.split()
    if len(palavras) <= max_palavras:
        return texto
    texto_cortado = ' '.join(palavras[:max_palavras])
    for sep in ['. ', '! ', '? ']:
        ultimo = texto_cortado.rfind(sep)
        if ultimo != -1:
            return texto_cortado[:ultimo + 1].strip()
    return texto_cortado.strip()

def gerar_meta(topic):
    prompt = (
        "You are an expert scientist. Write a single encyclopedic paragraph about the topic below in your own words. "
        "CRITICAL RULES: (1) Write ONLY ONE paragraph with NO line breaks. "
        "(2) The paragraph MUST contain between 80 and 120 words — count carefully. "
        "(3) Use a formal, scientific, encyclopedia-style tone. "
        "(4) Do NOT include titles, headers, bullet points, word count, or any commentary. "
        "(5) Output ONLY the paragraph, nothing else.\n\n"
        f"Topic:\n{topic}"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=250,
        )
        texto = response.choices[0].message.content.strip()
        # Garantir que é apenas um parágrafo e remover quebras de linha
        primeiro_paragrafo = texto.split('\n\n')[0].replace('\n', ' ').strip()
        return truncar_para_limite(primeiro_paragrafo, max_palavras=160)
    except Exception as e:
        print(f"Erro Meta para o tópico '{topic}': {e}")
        return None

TOPICS_FILE = "temas_meta.txt"
OUTPUT_FILE = "dataset_meta1.csv"
RETRIES_PER_TOPIC = 3

if not os.path.exists(TOPICS_FILE):
    print(f"Erro: {TOPICS_FILE} não encontrado.")
    exit(1)

with open(TOPICS_FILE, 'r', encoding='utf-8') as f:
    topics = [l.strip() for l in f if l.strip()]

print(f"Iniciar geração para {len(topics)} tópicos usando {MODEL_NAME}...")

fieldnames = ['text', 'label']
needs_header = not os.path.exists(OUTPUT_FILE)

with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if needs_header:
        writer.writeheader()

    for idx, topic in enumerate(topics):
        texto_gerado = None
        for attempt in range(RETRIES_PER_TOPIC):
            texto_gerado = gerar_meta(topic)
            
            if texto_gerado:
                wc = len(texto_gerado.split())
                if 70 <= wc <= 140:
                    break
                else:
                    print(f"[Meta] Tentativa {attempt+1} para '{topic}': texto fora do intervalo ({wc} palavras), tentando novamente...")
                    texto_gerado = None
                    time.sleep(2)
            else:
                time.sleep(2)

        if texto_gerado:
            texto_limpo = texto_gerado.replace('\n', ' ').replace('\r', ' ').strip()
            row = {
                'text': texto_limpo,
                'label': 'Meta',
            }
            writer.writerow(row)
            csvfile.flush()
            print(f"[{idx+1}/{len(topics)}] Gerado para '{topic}' ({len(texto_limpo.split())} palavras)")
        else:
            print(f"Falhou gerar texto para '{topic}' após {RETRIES_PER_TOPIC} tentativas.")
        
        time.sleep(2)

print(f"Geração concluída. Resultados guardados em {OUTPUT_FILE}")