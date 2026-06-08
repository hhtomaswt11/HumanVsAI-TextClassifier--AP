# Projeto de AP: Deteção de Texto Humano vs IA

Este repositório contém o trabalho desenvolvido para a unidade curricular de **Aprendizagem Profunda**. O objetivo do projeto é classificar textos entre 5 fontes: **Humano, Google, Meta, Anthropic e OpenAI**.

## Grupo 
* **Aluno 1:** Luís Pinto da Cunha (pg60280)
* **Aluno 2:** Nuno Filipe Leite Oliveira Araújo (pg61218)
* **Aluno 3:** Rodrigo Miguel Granja Ferreira (pg60392)
* **Aluno 4:** Tomás Henrique Alves Melo (pg60018)




## Organização do Repositório

O projeto está estruturado de forma modular:

*   **`data/`**: Contém os datasets em formato CSV utilizados para treino e validação (original, limpo e exemplos).
*   **`models/`**: Implementações de diversos modelos de aprendizagem profunda:
    *   **`numpy_models/`**: Implementação Numpy de raiz da arquitetura de redes neuronais (camadas, ativações, otimizadores, etc.).
    *   **`pytorch_models/`**: Implementações utilizando a framework PyTorch (MLP, RNN, LSTM, GRU).
    *   **`transformers_models/`**: Modelos baseados em Transformers.
    *   **`llm_models/`**: Avaliação de Large Language Models para classificação de texto.
*   **`notebooks/`**: Notebooks Jupyter para experimentação e apresentação de resultados detalhados.
*   **`scripts/`**: Scripts Python para recolha de dados via APIs (Google, OpenAI, Meta, Anthropic), limpeza automática e fusão de datasets.
*   **`Subm1/`, `Subm2/`, `Subm3/`**: Resultados e notebooks de submissão para as diferentes etapas do projeto.
*   **`Apresentacao/`**: Apresentação do projeto (vídeo).
*   **`subm*.csv`**: Datasets usados na predição das 3 submissões.


## Avaliação Final 

**Nota:** **18/20**
