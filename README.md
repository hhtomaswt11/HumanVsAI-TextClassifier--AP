# Projeto de AP: Deteção de Texto Humano vs IA

Este repositório contém o trabalho desenvolvido para a unidade curricular de **Aprendizagem Profunda**. O objetivo do projeto é classificar textos entre 5 autores: **Humano, Google, Meta, Anthropic e OpenAI**.

## 👥 Grupo 
* **Aluno 1:** Rodrigo Miguel Granja Ferreira (pg60392)
* **Aluno 2:** Tomás Henrique Alves Melo (pg60018)
* **Aluno 3:** Luís Pinto da Cunha (pg60280)
* **Aluno 4:** Nuno Filipe Leite Oliveira Araújo (pg61218)


## 📂 Organização do Repositório

O projeto está estruturado de forma modular para separar a lógica de processamento, os modelos e a experimentação:

* **`data/`**: Contém os datasets em formato CSV (original, limpo e exemplos).
* **`models/`**: 
    * **`numpy_models/`**: Implementação de raiz da rede neuronal (camadas, ativações, otimizadores, etc.).
    * **`pytorch_models/`**: Implementações utilizando a framework PyTorch.
* **`notebooks/`**: Notebooks Jupyter para experimentação e apresentação de resultados (Tarefa 2 e Tarefa 3).
* **`scripts/`**: Scripts Python utilitários para geração e limpeza automática de dados.
