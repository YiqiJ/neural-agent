# neuro-chat-agent — terminal agent

Tiny terminal agent to summarize your neuroscience dataset. Beginner-friendly and no UI required.

## Quick start (Windows PowerShell)

1) Create a virtual environment and install dependencies

```
python -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -U pip ; pip install -r requirements.txt
```

2) (Optional) Set your OpenAI API key if you want LLM-assisted path extraction

```
$env:OPENAI_API_KEY = "sk-..."
```

3) Run the agent with a natural-language query

```
python main.py "tell me the data summary of the dataset {data/oleg_data.parquet}"
```

Or start interactive mode and type your request:

```
python main.py
```

Example input in interactive mode:

```
> tell me the data summary of the dataset {data/oleg_data.parquet}
```

## What it does

- Parses your natural-language request to find a .parquet path (via simple heuristics or OpenAI tool-calling if configured).
- Loads the dataset using `src/data.py::read_data`.
- Computes a per-mouse summary via `src/data.py::dataset_summary`.
- Prints a readable table to the terminal.

## Project layout

- `src/data.py` — data loading and summary utilities (already provided)
- `src/agent.py` — minimal agent that parses queries and calls data functions
- `src/__init__.py` — marks `src` as a Python package
- `main.py` — small CLI entry point
- `data/` — includes `oleg_data.parquet` example

## Notes

- OpenAI is optional; without an API key the agent falls back to simple regex extraction for the dataset path.
- If you have very large datasets, ensure you have enough RAM and that `pyarrow` is installed (included in `requirements.txt`).
