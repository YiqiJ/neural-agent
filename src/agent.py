from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, List, Dict

from dotenv import load_dotenv
from openai import OpenAI

from .data import read_data, dataset_summary

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

    # ---------- tools ----------
    @staticmethod
    def extract_dataset_path(query: str) -> Optional[str]:
        if not query:
            return None
        m = re.search(r"\{([^}]+)\}", query)
        if m:
            return m.group(1).strip()
        m = re.search(r"['\"]([^'\"]+)['\"]", query)
        if m:
            return m.group(1).strip()
        m = re.search(r"([A-Za-z]:\\[^\s]+\.parquet|/[^\s]+\.parquet|[^\s]+\.parquet)", query)
        if m:
            return m.group(1).strip()
        return None

    def _llm_try_extract_path(self, query: str) -> Optional[str]:
        if not self.client:
            return None
        try:
            tools = [{
                "type": "function",
                "function": {
                    "name": "extract_path",
                    "description": "Extract the dataset file path from the user's request.",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                },
            }]
            messages = [
                {"role": "system", "content": "Extract a .parquet path if present; otherwise do nothing."},
                {"role": "user", "content": query},
            ]
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
            )
            msg = resp.choices[0].message
            tcs = getattr(msg, "tool_calls", None)
            if not tcs:
                return None
            import json
            for tc in tcs:
                if tc.function and tc.function.name == "extract_path":
                    args = json.loads(tc.function.arguments or "{}")
                    return args.get("path")
            return None
        except Exception:
            return None

    # ---------- actions ----------
    def summarize_dataset(self, path_str: str) -> str:
        p = Path(path_str).expanduser().resolve()
        if not p.exists():
            return f"Error: File not found at {p}"
        try:
            df = read_data(p)
            s = dataset_summary(df)
            float_cols = [c for c in ["trial_length_mean","trial_length_std","trial_length_min","trial_length_max"] if c in s.columns]
            if float_cols:
                s[float_cols] = s[float_cols].astype(float).round(3)
            return s.to_string(index=False)
        except Exception as e:
            return f"Error while summarizing dataset: {e}"

    # ---------- chat ----------
    def chat(self, query: str) -> str:
        ql = (query or "").strip().lower()
        if re.search(r"\b(what('?s)?|whats|what is)\s+your\s+name\b", ql):
            return "GitHub Copilot"
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
        ql = (query or "").lower()
        summary_keywords = ["summary", "summarize", "summarise", "overview", "describe"]
        is_summary = any(kw in ql for kw in summary_keywords)
        if is_summary:
            path = self._llm_try_extract_path(query) or self.extract_dataset_path(query)
            if not path:
                return "Please include a .parquet file path, e.g. data/dataset.parquet"
            return self.summarize_dataset(path)
        return self.chat(query)

def main():
    import argparse
    agent = Agent()
    parser = argparse.ArgumentParser(description="Class-based terminal agent")
    parser.add_argument("query", nargs="*")
    args = parser.parse_args()
    if args.query:
        print(agent.handle(" ".join(args.query)))
        return
    print("Type your request (or 'exit' to quit, 'reset' to clear).")
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