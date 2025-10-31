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


## Project layout


## Notes


## MCP server (optional)

This repo includes a minimal Model Context Protocol (MCP) server exposing a dataset utility:

- Tool `list_mice(dataset_root: str, pattern?: str, include_full_paths?: bool)` returns subject directories (e.g., `Subject-<id>` or `Mouse-<id>`).

Install dependencies if not already:

```powershell
python -m pip install -r requirements.txt
```

Quick one-off check (prints JSON):

```powershell
python -m src.tools.mcp_server --dataset-root data --once
```

Start the MCP server on stdio:

```powershell
python -m src.tools.mcp_server
```

You can connect with an MCP-compatible client to call `list_mice`.
