# LinkedIn Data Analysis

Streamlit webapp that analyses your LinkedIn post history using local LLMs via Ollama.

## Features

- **Monthly / Yearly charts** — stacked bar charts of posts over time, split by original vs reshared
- **Word cloud** — frequency map of all post commentary text
- **Technical terms cloud** — noun/proper-noun extraction via NLTK POS tagging
- **Topic classification** — each post classified into one of 11 configurable topic buckets (LLM, cached)
- **Company mentions** — top-30 organisations extracted from posts (LLM, cached)

Model, data/cache locations, and topic buckets are all configurable via [config/settings.json](config/settings.json).

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

### 3. Install and start Ollama

Download from [ollama.com](https://ollama.com), then pull the model set in `config/settings.json` (default `gemma4:latest`):

```bash
ollama pull gemma4:latest
```

Any Ollama model works — change the `model` value in config, type a name in the app sidebar, or pull from the sidebar at runtime.

### 4. Run the app

```bash
uv run streamlit run webapp.py
```

Requires Ollama running locally. Default model: `gemma4:latest` (from `config/settings.json`). Any Ollama model can be typed in or downloaded from the sidebar.

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

## Configuration

App settings live in [config/settings.json](config/settings.json):

| Key | Description |
|---|---|
| `model` | Default Ollama model (e.g. `gemma4:latest`). Overridable from the sidebar. |
| `data_dir` | Folder scanned for the LinkedIn export (default `data`) |
| `cache_dir` | Folder for per-user LLM caches (default `cache`) |
| `topics` | The topic buckets posts are classified into — edit this list to retune classification |

## Data

Two ways to load your LinkedIn export:

**Upload it (easiest):** drop the raw `Complete_LinkedInDataExport_*.zip` straight into the **Data** uploader in the sidebar. The app extracts it into `data/` on first upload and reuses that extraction on subsequent runs — re-uploading the same file won't unzip it again. (Streamlit caps uploads at 200 MB by default; complete exports are normally well under that.)

**Place it manually:** drop an already-extracted export folder into `data/`:

```
data/
└── Complete_LinkedInDataExport_<date>.zip   ← extracted folder, not the .zip file
```

Either way, the app reads `Shares.csv` from the most recent `Complete_*.zip` folder.

### Downloading Your LinkedIn Archive

1. linkedin.com → profile picture → **Settings & Privacy**
2. **Data Privacy** → **Get a copy of your data**
3. Select **Download larger data archive** → **Request archive**
4. Wait for email (up to 24 hours), download within 72 hours
5. Extract the zip and place the folder in `data/`

## Notebooks

- [notebooks/main.ipynb](notebooks/main.ipynb) — exploratory analysis
- [notebooks/writing_style.ipynb](notebooks/writing_style.ipynb) — profiles your writing voice from your original posts. Computes quantitative style metrics (length, sentence rhythm, paragraph structure, emoji/hashtag/question habits, vocabulary richness), visualizes the distributions, surfaces signature words and opening hooks, then uses the local Ollama model to synthesize a reusable style guide. Saves [outputs/writing_style.md](outputs/) and `outputs/writing_style_metrics.json`.

### Stripping outputs from commits

Notebook outputs are kept out of git via [nbstripout](https://github.com/kynan/nbstripout) (a dev dependency). A git clean filter strips cell outputs and execution counts when notebooks are staged, while leaving your working copy untouched. The filter is committed in [.gitattributes](.gitattributes), but git requires each clone to register it locally (one-time, per checkout):

```bash
git config filter.nbstripout.clean "uv run --no-sync nbstripout"
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true
```

After that, `git add`/`git commit` on any `*.ipynb` automatically drops outputs from what's committed.

## Caching

LLM results (topic classification, company extraction) are cached per user in `cache/`. Each unique name in the sidebar gets its own cache files so results aren't mixed.

## Sources

- [LinkedIn Help — Downloading your account data](https://www.linkedin.com/help/linkedin/answer/a1339364/downloading-your-account-data)
