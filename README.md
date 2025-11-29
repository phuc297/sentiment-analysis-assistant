# Sentiment Analysis Assistant

## Installation and Setup
This project uses `uv` for dependency management. Ensure you have `uv` installed on your system.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sentiment-analysis-assistant.git
cd sentiment-analysis-assistant
```

### 2. Create and Activate Virtual Environment

Use `uv` to create a virtual environment and install dependencies.

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate # (in Linux)
venv\Scripts\Activate # (in Windows)
```



### 3. Install Dependencies

Use `uv sync` to install all required packages defined in `pyproject.toml`.

```bash
uv sync
```

## Usage

### 1. Start the FastAPI Backend

The backend runs on Uvicorn and serves the sentiment analysis model via an API.

```bash
uvicorn src.api:app
```

### 2. Start the Streamlit Frontend

The frontend is the interactive user interface that communicates with the FastAPI backend.

```bash
streamlit run ./src/streamlit.py
```
