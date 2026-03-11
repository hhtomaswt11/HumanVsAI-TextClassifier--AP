import pandas as pd
import re
import os

def limpar_dataset(input_file, output_file):
    print(f"A carregar o dataset: {input_file}...")
    df = pd.read_csv(input_file, sep=';', on_bad_lines='skip')
    
    total_inicial = len(df)
    print(f"Total de linhas originais: {total_inicial}")

    def limpar_prefixos(texto):
        if pd.isna(texto):
            return texto
        
        texto_str = str(texto).strip()
        
    
        padroes = [
            # Cobre "This/The research paper/article/study investigates/explores/presents/focuses on/delves into..."
            # Cobre também se tiver um título pelo meio ex: titled "Nome do artigo"
            r"^(This|The) (research )?(paper|article|thesis|study) (titled [\"'].+?[\"'] )?(investigates|explores|delves into|focuses on|presents|aims to( \w+)?|provides( an overview of)?|analyses|analyzes|compares|studies|reports on|evaluates) (the |a |an )?",
            
            # Cobre "In this paper/article, we investigate/explore/discuss..."
            r"^In this (research )?(paper|article|study), we (investigate|explore|discuss|analyze|analyse|present|propose) (the |a |an )?"
        ]
        
        texto_str = re.sub(r'^"+', '', texto_str)

        for padrao in padroes:
            # Substitui o padrão por nada, ignorando maiúsculas/minúsculas
            texto_str = re.sub(padrao, "", texto_str, flags=re.IGNORECASE)
            
        # Capitalizar a primeira letra para o modelo não estranhar
        if len(texto_str) > 0:
            texto_str = texto_str[0].upper() + texto_str[1:]
            
        return texto_str

    df['Text'] = df['Text'].apply(limpar_prefixos)

    def is_valid_text(texto):
        if pd.isna(texto):
            return False
            
        texto_str = str(texto).strip()
        
        # Aceitamos textos que terminem de forma "fechada"
        # Adicionei o parêntese ")" porque alguns abstracts terminam com "(abridged)" ou anos.
        return texto_str.endswith(('.', '!', '?', '"', "'", "”", "’", ")"))

    df_limpo = df[df['Text'].apply(is_valid_text)].copy()
    
    total_final = len(df_limpo)
    removidos = total_inicial - total_final
    
    print(f"\nTextos truncados removidos: {removidos}")
    print(f"Total de linhas no dataset limpo: {total_final}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_limpo.to_csv(output_file, sep=';', index=False)
    
    print(f"\n[SUCESSO] Dataset limpo guardado em: {output_file}")
    
    return df_limpo

if __name__ == "__main__":
    caminho_input = '../data/dataset_final.csv'
    caminho_output = '../data/dataset_limpo.csv'
    
    dataset_limpo = limpar_dataset(caminho_input, caminho_output)