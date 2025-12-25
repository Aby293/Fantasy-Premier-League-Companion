# Fantasy Premier League Graph-RAG System

## Overview
The Fantasy Premier League Graph-RAG System is an advanced AI-powered platform designed to assist Fantasy Premier League (FPL) enthusiasts by leveraging Retrieval-Augmented Generation (RAG) with a Neo4j graph database, LangChain, and large language models (LLMs) like Gemma, Llama, and Mistral. The system integrates machine learning models for player performance prediction, a knowledge graph for structured FPL data representation, and an interactive Streamlit interface for querying and visualization.

## Features
- **Data Cleaning & Preparation:**
  - Processes raw FPL data (see `cleaned_fpl.csv`, `fpl_two_seasons.csv`, `two_seasons_cleaned.csv`).
  - Fixture data available in `fixtures.csv`.
- **Machine Learning Models:**
  - Neural network model (`ffnn_model.keras`), Random Forest Model (`random_forest_model_compressed.pkl`), and XGBoost model (`xgb_model.json`) for predicting player points.
  - Model explanations and visualizations (see `Figures`).
- **Knowledge Graph Construction:**
  - Scripts for creating and populating a Neo4j knowledge graph with FPL entities (`MS2/Create_kg.py`).
  - Sample queries for graph exploration (`MS2/queries.txt`).
- **Retrieval-Augmented Generation (RAG):**
  - LangChain-based RAG system with vector embeddings and LLM integration for natural language queries.
  - Support for multiple LLMs (Gemma, Llama, Mistral) via HuggingFace API.
  - Graph-aware retrieval and response generation.
- **Interactive Analysis:**
  - Jupyter notebooks for data exploration, model training, and evaluation (`fantasy-pl.ipynb`, `MS3/notebook.ipynb`, `MS3/db_editing.ipynb`).
  - Evaluation results in `MS3/evaluation_results.csv`.
- **Web Interface:**
  - Streamlit-based app for user interaction, graph visualization, and query execution (`MS3/interface.py`).
  - Real-time query history, intent classification, and debug information.

## Directory Structure
- `Figures/`: Visualizations and model explanations.
- `models/`: Pre-trained machine learning models.
- `MS2/`: Knowledge graph creation scripts, configurations, and data.
- `MS3/`: RAG system backend (`app.py`), Streamlit interface (`interface.py`), evaluation, and notebooks.

## How to Run the System
### Prerequisites
- **Python:** 3.8+ recommended.
- **Neo4j Database:** Install and run Neo4j locally (default URI: `neo4j://localhost:7687`). Update credentials in `MS2/config.txt` and `MS3/config.txt`.
- **GPU (Optional):** For faster LLM inference, ensure CUDA-compatible GPU and PyTorch with CUDA support.

### 1. Environment Setup
Install required packages:
```bash
pip install -r requirements.txt
```
If `requirements.txt` is missing, install key dependencies:
```bash
pip install streamlit langchain langchain-community langchain-huggingface neo4j pandas spacy transformers torch plotly networkx scikit-learn xgboost keras tensorflow huggingface-hub
python -m spacy download en_core_web_sm
```

### 2. Data Preparation
- Place your FPL data files in the root or `MS2/` as needed.
- Use the provided notebooks to explore and clean data.

### 3. Knowledge Graph Setup
- Ensure Neo4j is running.
- Run the knowledge graph creation script:
  ```bash
  python MS2/Create_kg.py
  ```
- Verify graph population using `MS2/queries.txt`.

### 4. Machine Learning Models
- Models are stored in `models/`. Use the notebooks to load and evaluate them.
- Visual explanations are in `Figures/`.

### 5. Web Application
- Start the Streamlit interface:
  ```bash
  streamlit run MS3/interface.py
  ```
- Access the app via the provided local URL (e.g., http://localhost:8501).
- Interact with the RAG system: enter natural language queries about FPL, view graph visualizations, and explore results.

### 6. Interactive Notebooks
- Open and run `fantasy-pl.ipynb` or `MS3/notebook.ipynb` in Jupyter for data analysis, model evaluation, and system setup.

## License
See `LICENSE` for details.

## Contributors
Built by Aby, Zeina, Habiba, Ehab.