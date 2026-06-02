# GitHub Repositories Recommender

A graph-based GitHub repository recommendation system that will use repository metadata, embeddings, and graph relationships to suggest relevant repositories.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- sentence-transformers
- NetworkX
- node2vec
- Streamlit
- matplotlib
- Plotly

## Dataset

Dataset: https://www.kaggle.com/datasets/donbarbos/github-repos

The dataset is not committed to GitHub because of its size. Download it separately and place raw files in `data/raw/`.

## Kaggle Credentials

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` and add your Kaggle username and API key:

```text
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

The `.env` file contains private API credentials and must never be committed to GitHub. It is ignored by Git so each user can keep their own local Kaggle token outside version control.

## Dataset download

Install project dependencies:

```bash
pip install -r requirements.txt
```

Download the Kaggle dataset:

```bash
python src/download_dataset.py
```

KaggleHub downloads the dataset to a local cache folder and prints the downloaded path. Downloaded files are not moved into this repository. Kaggle credentials may be required if Kaggle asks for authentication.

## Local LLM baseline with Ollama

Install Ollama:

```bash
brew install ollama
```

Download the default local model:

```bash
ollama pull mistral
```

Start the Ollama server:

```bash
ollama serve
```

Run the Streamlit demo:

```bash
streamlit run app.py
```

The demo includes a local LLM comparison using Ollama. If Ollama is not running, the app shows a warning instead of crashing.
