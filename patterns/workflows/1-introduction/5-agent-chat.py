"""Chat agent demo replicating the n8n "Build Your First AI Agent" workflow.

This script showcases a durable, maintainable Python implementation of the
workflow. It exposes a simple command-line chat interface where the model can
use tools to fetch weather data or headlines from an RSS feed. Recent
conversation context is stored to mimic short‑term memory.

Requirements:
    * An OpenAI API key available as the ``OPENAI_API_KEY`` environment variable.
    * Dependencies listed in ``patterns/workflows/requirements.txt``.

Run the script and start chatting::

    python patterns/workflows/1-introduction/5-agent-chat.py

Type ``exit`` to stop the conversation.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import feedparser
import requests
from openai import OpenAI


class ToolError(RuntimeError):
    """Raised when a tool cannot complete its task."""


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def get_weather(city: str) -> Dict[str, Any]:
    """Return current temperature for ``city`` in Celsius.

    A two‑step call is performed: first the city is geocoded using the official
    Open‑Meteo geocoding endpoint, then the coordinates are used to retrieve the
    current weather. Network errors raise ``ToolError`` so the agent can report
    them gracefully.
    """

    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        geo_resp.raise_for_status()
        data = geo_resp.json()
    except requests.RequestException as exc:  # pragma: no cover - network
        raise ToolError(f"Geocoding request failed: {exc}") from exc

    results = data.get("results") or []
    if not results:
        raise ToolError(f"No results found for city '{city}'.")

    latitude = results[0]["latitude"]
    longitude = results[0]["longitude"]

    try:
        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,weathercode",
            },
            timeout=10,
        )
        weather_resp.raise_for_status()
        current = weather_resp.json().get("current", {})
    except requests.RequestException as exc:  # pragma: no cover - network
        raise ToolError(f"Forecast request failed: {exc}") from exc

    return {
        "city": city,
        "latitude": latitude,
        "longitude": longitude,
        "temperature_c": current.get("temperature_2m"),
        "weathercode": current.get("weathercode"),
    }


def get_news(feed_url: str, limit: int = 5) -> Dict[str, Any]:
    """Return the latest ``limit`` headlines from ``feed_url``.

    ``feedparser`` is used because it handles many edge cases around broken
    feeds and different encodings. Only the title and link of each entry are
    returned to keep responses short and deterministic.
    """

    parsed = feedparser.parse(feed_url)
    items = [
        {"title": entry.get("title", ""), "link": entry.get("link", "")}
        for entry in parsed.entries[:limit]
    ]
    return {
        "feed": parsed.feed.get("title", "Unnamed feed"),
        "items": items,
    }


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------


class ChatAgent:
    """Minimal chat agent with tool usage and short‑term memory."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.system_message = (
            "You are a helpful demo AI agent. Use tools for weather and news when "
            "appropriate."
        )
        self.memory: List[Dict[str, str]] = []

    # --------------------- Helper methods ---------------------------------

    def _build_messages(self, user_input: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_message}]
        messages.extend(self.memory[-30:])  # keep last 30 exchanges
        messages.append({"role": "user", "content": user_input})
        return messages

    def _tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a given city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_news",
                    "description": "Get the latest headlines from an RSS feed URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "feed_url": {"type": "string"},
                            "limit": {"type": "integer", "default": 5},
                        },
                        "required": ["feed_url"],
                    },
                },
            },
        ]

    def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name == "get_weather":
            return get_weather(**arguments)
        if name == "get_news":
            return get_news(**arguments)
        raise ToolError(f"Unknown tool '{name}'.")

    # ------------------------- Public API ----------------------------------

    def chat(self, user_input: str) -> str:
        """Return the agent's reply to ``user_input``."""

        messages = self._build_messages(user_input)
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=self._tools()
        )
        message = response.choices[0].message

        if message.tool_calls:
            # Execute each tool call then call the model again with results
            tool_messages: List[Dict[str, Any]] = []
            for call in message.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                result = self._execute_tool(call.function.name, args)
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.function.name,
                        "content": json.dumps(result),
                    }
                )
            messages.append(message)
            messages.extend(tool_messages)
            follow_up = self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            final = follow_up.choices[0].message.content or ""
        else:
            final = message.content or ""

        # Update memory with the latest exchange
        self.memory.extend(
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": final},
            ]
        )
        self.memory = self.memory[-30:]
        return final


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def main() -> None:
    agent = ChatAgent()
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        try:
            reply = agent.chat(user_input)
            print(f"Agent: {reply}\n")
        except ToolError as err:
            print(f"Tool error: {err}\n")
        except Exception as err:  # pragma: no cover - for robustness
            print(f"Unexpected error: {err}\n")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
