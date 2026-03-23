import os
import time
import csv
import json
import requests
import uuid
from dotenv import load_dotenv

# 1. Configuração de Ambiente
load_dotenv()
API_ENDPOINT = "https://api.iaedu.pt/agent-chat//api/v1/agent/cmamvd3n40000c801qeacoad2/stream"
API_KEY = os.environ.get("AIEDU_KEY")
CHANNEL_ID = os.environ.get("CHANNEL_ID")

# 2. Funções Auxiliares
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

def gerar_iaedu(topic):
    prompt = (
        "You are an expert scientist. Write a single encyclopedic paragraph about the topic below in your own words. "
        "CRITICAL RULES: (1) Write ONLY ONE paragraph with NO line breaks. "
        "(2) The paragraph MUST contain between 80 and 120 words — count carefully. "
        "(3) Use a formal, scientific, encyclopedia-style tone. "
        "(4) Do NOT include titles, headers, bullet points, word count, or any commentary. "
        "(5) Output ONLY the paragraph, nothing else.\n\n"
        f"Topic:\n{topic}"
    )
    
    headers = {
        "x-api-key": API_KEY,
    }
    
    # Generate a unique thread_id for each generation request to ensure independence
    thread_id = f"gen_{uuid.uuid4().hex[:8]}"
    
    data = {
        "channel_id": CHANNEL_ID,
        "thread_id": thread_id,
        "user_info": "{}",
        "message": prompt
    }
    
    # multipart/form-data via files parameter in requests
    form_files = {k: (None, v) for k, v in data.items()}
    
    try:
        response = requests.post(API_ENDPOINT, headers=headers, files=form_files, stream=True)
        if response.status_code != 200:
            print(f"Erro IAEDU API (Status {response.status_code}): {response.text}")
            return None
            
        full_content = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    if chunk.get("type") == "message":
                        # The final message object contains the full content
                        content = chunk.get("content", {})
                        if isinstance(content, dict):
                            full_content = content.get("content", "").strip()
                        else:
                            # Fallback if structure is different
                            full_content = str(content).strip()
                except json.JSONDecodeError:
                    continue
        
        if not full_content:
            return None

        # Garantir que é apenas um parágrafo e remover quebras de linha
        primeiro_paragrafo = full_content.split('\n\n')[0].replace('\n', ' ').strip()
        return truncar_para_limite(primeiro_paragrafo, max_palavras=160)
        
    except Exception as e:
        print(f"Erro IAEDU para o tópico '{topic}': {e}")
        return None

# 3. Execução Principal
TOPICS_FILE = "temas_openai.txt"
OUTPUT_FILE = "dataset_openai1.csv"
RETRIES_PER_TOPIC = 3

if not os.path.exists(TOPICS_FILE):
    print(f"Erro: {TOPICS_FILE} não encontrado. Saindo.")
    exit(1)

with open(TOPICS_FILE, 'r', encoding='utf-8') as f:
    topics = [l.strip() for l in f if l.strip()]

print(f"Iniciando geração para {len(topics)} tópicos usando IAEDU API...")

fieldnames = ['text', 'label']
needs_header = not os.path.exists(OUTPUT_FILE)

with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if needs_header:
        writer.writeheader()

    for idx, topic in enumerate(topics):
        texto_gerado = None
        for attempt in range(RETRIES_PER_TOPIC):
            texto_gerado = gerar_iaedu(topic)
            
            if texto_gerado:
                wc = len(texto_gerado.split())
                if 70 <= wc <= 140:
                    break
                else:
                    print(f"[IAEDU] Tentativa {attempt+1} para '{topic}': texto fora do intervalo ({wc} palavras), tentando novamente...")
                    texto_gerado = None
                    time.sleep(2)
            else:
                time.sleep(2)

        if texto_gerado:
            texto_limpo = texto_gerado.replace('\n', ' ').replace('\r', ' ').strip()
            row = {
                'text': texto_limpo,
                'label': 'OpenAI',
            }
            writer.writerow(row)
            csvfile.flush()
            print(f"[{idx+1}/{len(topics)}] Gerado para '{topic}' ({len(texto_limpo.split())} palavras)")
        else:
            print(f"Falhou gerar texto para '{topic}' após {RETRIES_PER_TOPIC} tentativas.")
        
        # Buffer time between requests
        time.sleep(2)

print(f"Geração concluída. Resultados guardados em {OUTPUT_FILE}")
