from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Optional, List, Dict

from dotenv import load_dotenv
from openai import OpenAI

from src.tools.data import summarize_dataset, summarize_single_mouse
from src.tools.single_cell import compute_trial_avg_responses

load_dotenv()

class Agent:
    def __init__(self, model: str | None = None, system_prompt: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.system_prompt = system_prompt or "You are a concise, friendly terminal assistant."
        self.history: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def reset(self):
        self.history = [{"role": "system", "content": self.system_prompt}]
    # ---------- chat ----------
    def chat(self, query: str) -> str:
        ql = (query or "").strip().lower()
        if re.search(r"\b(what('?s)?|whats|what is)\s+your\s+name\b", ql):
            return "Friendly Neural Data Agent."
        if not self.client:
            return "Set OPENAI_API_KEY (env or .env) to enable chat."
        try:
            messages = [*self.history, {"role": "user", "content": query}]
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.6,
            )
            content = (resp.choices[0].message.content or "").strip()
            self.history.append({"role": "user", "content": query})
            self.history.append({"role": "assistant", "content": content})
            # optional: truncate to last N turns
            if len(self.history) > 25:
                self.history = [self.history[0], *self.history[-24:]]
            return content
        except Exception as e:
            return f"Chat error: {e}"

    # ---------- router ----------
    def handle(self, query: str) -> str:
        q = (query or "").strip()
        if not self.client:
            return "Set OPENAI_API_KEY (env or .env) to enable the agent."

        try:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "summarize_dataset",
                        "description": "Summarize a .parquet dataset at the given path.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Absolute or relative path to a .parquet file.",
                                }
                            },
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "summarize_single_mouse",
                        "description": "Summarize dataset statistics for a single mouse ID.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Absolute or relative path to a .parquet file.",
                                },
                                "mouse_id": {
                                    "type": "string",
                                    "description": "The mouse ID to summarize (matches values in the dataset).",
                                },
                            },
                            "required": ["path", "mouse_id"],
                            "additionalProperties": False,
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "compute_trial_avg_responses",
                        "description": "Compute and plot trial average responses for a single mouse ID.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Absolute or relative path to a .parquet file."
                                },
                                "mouse_id": {
                                    "type": "string",
                                    "description": "The mouse ID to summarize (matches values in the dataset)."
                                },
                                "sort": {
                                    "type": "string",
                                    "enum": ["peak_time", "peak_amp"],
                                    "description": "Row sorting method for heatmap."
                                },
                                "percentile_clip": {
                                    "type": "number",
                                    "description": "Percentile for robust color clipping (e.g., 1 => [1,99])."
                                },
                                "vmin": { "type": "number", "description": "Lower color limit." },
                                "vmax": { "type": "number", "description": "Upper color limit." },
                                "cmap": { "type": "string", "description": "Matplotlib colormap name." },
                                "save_path": { "type": "string", "description": "Path to save the plot." },
                                "plot": { "type": "boolean", "description": "Show the plot window." }
                            },
                            "required": ["path", "mouse_id"],
                            "additionalProperties": False
                        }
                    }
                },
            ]

            messages = [*self.history, {"role": "user", "content": q}]
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
            )
            msg = resp.choices[0].message
            tcs = getattr(msg, "tool_calls", None)

            # If the model called a tool, execute and return its result
            if tcs:
                for tc in tcs:
                    if not tc.function:
                        continue
                    name = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")

                    if name == "summarize_dataset":
                        path = args.get("path")
                        if not path:
                            return "Please include a .parquet file path, e.g. data/dataset.parquet"
                        result = summarize_dataset(path)

                    elif name == "summarize_single_mouse":
                        path = args.get("path")
                        mouse_id = args.get("mouse_id")
                        if not path or not mouse_id:
                            return "Please include both 'path' and 'mouse_id', e.g. path=data/dataset.parquet mouse_id=mouse_01"
                        result = summarize_single_mouse(path, mouse_id)

                    elif name == "compute_trial_avg_responses":
                        path = args.pop("path", None)
                        mouse_id = args.pop("mouse_id", None)
                        if not path or not mouse_id:
                            return "Please include both 'path' and 'mouse_id', e.g. path=data/dataset.parquet mouse_id=mouse_01"
                        # Pass through any optional kwargs (sort, percentile_clip, vmin, vmax, cmap, save_path, plot)
                        result = compute_trial_avg_responses(path, mouse_id, **args)

                    else:
                        continue

                    # record turn
                    self.history.append({"role": "user", "content": q})
                    self.history.append({"role": "assistant", "content": result})
                    if len(self.history) > 25:
                        self.history = [self.history[0], *self.history[-24:]]
                    return result

            # Otherwise, normal chat
            content = (msg.content or "").strip()
            self.history.append({"role": "user", "content": q})
            self.history.append({"role": "assistant", "content": content})
            if len(self.history) > 25:
                self.history = [self.history[0], *self.history[-24:]]
            return content

        except Exception as e:
            return f"Router error: {e}"

def main():
    import argparse
    agent = Agent()
    parser = argparse.ArgumentParser(description="Class-based terminal agent")
    parser.add_argument("query", nargs="*")
    args = parser.parse_args()
    if args.query:
        print(agent.handle(" ".join(args.query)))
        return
    print("Type your request (or 'exit' to quit, 'reset' to clear chat memory).")
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        low = user.lower()
        if low in {"exit", "quit"}:
            break
        if low in {"reset", "clear"}:
            agent.reset()
            print("Chat history cleared.")
            continue
        print(agent.handle(user))

if __name__ == "__main__":
    main()