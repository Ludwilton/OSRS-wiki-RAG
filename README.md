# OSRS-wiki-RAG

A Retrieval-Augmented Generation (RAG) chatbot for Old School RuneScape wiki content.

![botdemo](./assets/bot_demo-webp.webp)

### 1. Install Ollama

Download and install from [https://ollama.com/](https://ollama.com)

### 2. Start Ollama and Pull Required Models

```bash

# In a new terminal, pull the required models
ollama pull qwen3:8b  # or your preferred chat model
ollama pull mxbai-embed-large  # for embeddings
```

### 3. Clone the Repository

```bash
git clone https://github.com/Ludwilton/OSRS-wiki-RAG.git
cd OSRS-wiki-RAG
```

### 4. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


## Usage

### 1. Scrape Wiki Data, Process and Clean Articles,  Create Vector Embeddings

```bash
python main.py
```

### 2. Run the Chatbot

```bash
streamlit run chatbot.py
```

The web interface will be available at `http://localhost:8501`

## Project Structure

```
OSRS-wiki-RAG/
├── chatbot.py              # Main Streamlit chatbot application
├── wikiscraper.py          # Wiki content scraping
├── article_cleanup.py      # Article preprocessing and cleaning
├── chunking_articles.py    # Text chunking and vector embedding creation
├── requirements.txt        # Python dependencies
├── .env                    # Environment configuration
├── chroma_db/             # ChromaDB vector database
├── datasets/
│   ├── osrs_articles/     # Raw scraped articles
│   └── clean_articles/    # Processed and cleaned articles
└── README.md
```


### LLM models

You can use different models by updating the `.env` file, make sure to unset the variable when swapping, generated response quality will vary significantly depending on model & system prompt.Defaults to qwen3:8b.


basD on [Local-RAG-with-Ollama](https://github.com/ThomasJanssen-tech/Local-RAG-with-Ollama)