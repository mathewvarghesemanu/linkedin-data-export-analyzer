# LinkedIn Data Analysis

Streamlit webapp that analyses your LinkedIn post history using local LLMs via Ollama.

## Features

- **Monthly / Yearly charts** — stacked bar charts of posts over time, split by original vs reshared
- **Word cloud** — frequency map of all post commentary text
- **Technical terms cloud** — noun/proper-noun extraction via NLTK POS tagging
- **Topic classification** — each post classified into one of 11 topic buckets (LLM, cached)
- **Company mentions** — top-30 organisations extracted from posts (LLM, cached)

## Setup

### 1. Install uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Create environment and install dependencies

```bash
uv sync
```

This creates `.venv/` and installs all dependencies from `pyproject.toml`.

### 3. Run the app

```bash
uv run streamlit run webapp.py
```

Requires [Ollama](https://ollama.com) running locally. Default model: `qwen2.5:14b`. Any Ollama model can be typed in or downloaded from the sidebar.

## Web App

Opens at `http://localhost:8501` after `uv run streamlit run webapp.py`.

**Sidebar controls:**

| Control | Description |
|---|---|
| Your name | Scopes LLM cache — each name gets isolated cache files |
| Model name | Type any Ollama model name (e.g. `qwen2.5:14b`, `llama3.2`) |
| ⬇️ button | Downloads model from Ollama in the background with live progress |
| Installed models | Click to select; ✕ to delete |
| Chart toggles | Enable/disable each section independently |

**Main panels** (all toggleable from sidebar):

| Panel | What it shows |
|---|---|
| Posts per Month | Stacked bar chart split by Original vs Reshared |
| Posts per Year | Same data aggregated annually |
| Word Cloud | Frequency map of all post text (filler words filtered) |
| Technical Terms Cloud | Nouns/proper nouns only via NLTK POS tagging |
| Topic Classification | Each post bucketed into 1 of 11 topics via LLM |
| Company Mentions | Top 30 organisations extracted from posts via LLM |

LLM-backed panels (topic + companies) classify on first run then read from cache — subsequent loads are instant.

## Data

Place your LinkedIn export in `data/`:

```
data/
└── Complete_LinkedInDataExport_<date>.zip   ← extracted folder, not the .zip file
```

The app reads `Shares.csv` from the most recent `Complete_*.zip` folder.

### Downloading Your LinkedIn Archive

1. linkedin.com → profile picture → **Settings & Privacy**
2. **Data Privacy** → **Get a copy of your data**
3. Select **Download larger data archive** → **Request archive**
4. Wait for email (up to 24 hours), download within 72 hours
5. Extract the zip and place the folder in `data/`

## Notebooks

- [notebooks/main.ipynb](notebooks/main.ipynb) — exploratory analysis
- [notebooks/vetting.ipynb](notebooks/vetting.ipynb) — vetting / filtering logic

## Caching

LLM results (topic classification, company extraction) are cached per user in `cache/`. Each unique name in the sidebar gets its own cache files so results aren't mixed.

## Sources

- [LinkedIn Help — Downloading your account data](https://www.linkedin.com/help/linkedin/answer/a1339364/downloading-your-account-data)
