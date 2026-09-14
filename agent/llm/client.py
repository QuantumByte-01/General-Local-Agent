from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types

from agent.config import DEFAULT_MODELS
from agent.state import ChatMessage


@dataclass
class LlmTurn:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    finish_reason: str = ""
    truncated: bool = False


def _schema_from_json(schema: dict[str, Any]) -> types.Schema:
    type_map = {
        "object": types.Type.OBJECT,
        "string": types.Type.STRING,
        "integer": types.Type.INTEGER,
        "number": types.Type.NUMBER,
        "boolean": types.Type.BOOLEAN,
        "array": types.Type.ARRAY,
    }
    json_type = str(schema.get("type") or "object").lower()
    gem_type = type_map.get(json_type, types.Type.OBJECT)
    props = None
    items = None
    if "properties" in schema and isinstance(schema["properties"], dict):
        props = {k: _schema_from_json(v if isinstance(v, dict) else {"type": "string"}) for k, v in schema["properties"].items()}
    if "items" in schema and isinstance(schema["items"], dict):
        items = _schema_from_json(schema["items"])
    return types.Schema(
        type=gem_type,
        description=schema.get("description"),
        properties=props,
        required=list(schema.get("required") or []) or None,
        items=items,
    )


def tools_to_gemini(tool_defs: list[dict[str, Any]]) -> list[types.Tool]:
    decls = []
    for spec in tool_defs:
        decls.append(
            types.FunctionDeclaration(
                name=spec["name"],
                description=spec.get("description") or spec["name"],
                parameters=_schema_from_json(spec.get("parameters") or {"type": "object", "properties": {}}),
            )
        )
    if not decls:
        return []
    return [types.Tool(function_declarations=decls)]


def messages_to_contents(messages: list[ChatMessage]) -> list[types.Content]:
    contents: list[types.Content] = []
    index = 0
    while index < len(messages):
        msg = messages[index]
        if msg.role == "system":
            index += 1
            continue
        if msg.role == "assistant":
            parts: list[types.Part] = []
            if msg.content:
                parts.append(types.Part(text=msg.content))
            for call in msg.tool_calls:
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            name=call["name"],
                            args=call.get("arguments") or {},
                        )
                    )
                )
            if parts:
                contents.append(types.Content(role="model", parts=parts))
            index += 1
            continue
        if msg.role == "tool":
            parts = []
            while index < len(messages) and messages[index].role == "tool":
                tool_msg = messages[index]
                parts.append(
                    types.Part.from_function_response(
                        name=tool_msg.tool_name or "tool",
                        response={"result": tool_msg.content},
                    )
                )
                index += 1
            contents.append(types.Content(role="user", parts=parts))
            continue
        contents.append(types.Content(role="user", parts=[types.Part(text=msg.content)]))
        index += 1
    return contents


class GeminiClient:
    def __init__(self, api_keys: list[str], models: list[str]) -> None:
        self.api_keys = [k for k in api_keys if k] or [""]
        self.models = models or list(DEFAULT_MODELS)
        self.latched_model: str | None = None
        self.latched_key_index: int | None = None

    def _client(self, key: str) -> genai.Client:
        key = (key or "").strip()
        if key:
            return genai.Client(api_key=key)
        return genai.Client()

    async def generate(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        tool_defs: list[dict[str, Any]],
        max_output_tokens: int,
    ) -> LlmTurn:
        contents = messages_to_contents(messages)
        gem_tools = tools_to_gemini(tool_defs)
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=gem_tools or None,
            max_output_tokens=max_output_tokens,
            temperature=0.3,
        )
        errors: list[str] = []
        model_order = [self.latched_model] + [m for m in self.models if m != self.latched_model] if self.latched_model else list(self.models)
        model_order = [m for m in model_order if m]
        key_indices = list(range(len(self.api_keys)))
        if self.latched_key_index is not None:
            key_indices = [self.latched_key_index] + [i for i in key_indices if i != self.latched_key_index]

        last_exc: Exception | None = None
        for key_i in key_indices:
            try:
                client = self._client(self.api_keys[key_i])
            except Exception as exc:
                errors.append(f"client[{key_i}]: {exc}")
                last_exc = exc
                continue
            for model in model_order:
                try:
                    generate = getattr(getattr(client, "aio", None), "models", None)
                    if generate is not None:
                        resp = await client.aio.models.generate_content(
                            model=model,
                            contents=contents,
                            config=config,
                        )
                    else:
                        resp = await asyncio.to_thread(
                            client.models.generate_content,
                            model=model,
                            contents=contents,
                            config=config,
                        )
                    self.latched_model = model
                    self.latched_key_index = key_i
                    return self._parse(resp, model)
                except Exception as exc:
                    last_exc = exc
                    errors.append(f"{model} key#{key_i + 1}: {exc}")
                    continue
        joined = "; ".join(errors)[:800]
        raise RuntimeError(joined or str(last_exc) or "LLM call failed")

    def _parse(self, resp: Any, model: str) -> LlmTurn:
        text_parts: list[str] = []
        calls: list[dict[str, Any]] = []
        finish = ""
        truncated = False
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            cand = candidates[0]
            finish = str(getattr(cand, "finish_reason", "") or "")
            truncated = "MAX_TOKENS" in finish.upper()
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            for i, part in enumerate(parts):
                fn = getattr(part, "function_call", None)
                if fn and getattr(fn, "name", None):
                    args = dict(getattr(fn, "args", None) or {})
                    calls.append(
                        {
                            "id": getattr(fn, "id", None) or f"call_{i}",
                            "name": fn.name,
                            "arguments": args,
                        }
                    )
                elif getattr(part, "text", None):
                    text_parts.append(part.text)
        text = "".join(text_parts).strip()
        if not text and not calls:
            text = (getattr(resp, "text", None) or "").strip()
        return LlmTurn(text=text, tool_calls=calls, model=model, finish_reason=finish, truncated=truncated)
