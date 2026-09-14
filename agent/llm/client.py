from __future__ import annotations

import asyncio
import json
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
    def __init__(
        self,
        api_keys: list[str],
        models: list[str],
        *,
        thinking_budget: int | None = 0,
        stream: bool = True,
        timeout_ms: int = 60_000,
    ) -> None:
        self.api_keys = [k for k in api_keys if k] or [""]
        self.models = models or list(DEFAULT_MODELS)
        self.latched_model: str | None = None
        self.latched_key_index: int | None = None
        self.thinking_budget = thinking_budget
        self.stream = stream
        self.timeout_ms = timeout_ms
        self._clients: dict[int, Any] = {}
        self._gem_tools_key: tuple[str, ...] | None = None
        self._gem_tools: list[types.Tool] | None = None
        self._thinking_ok = True

    def _client(self, key_index: int) -> Any:
        if key_index in self._clients:
            return self._clients[key_index]
        key = (self.api_keys[key_index] or "").strip()
        http_options = None
        try:
            http_options = types.HttpOptions(timeout=self.timeout_ms)
        except Exception:
            http_options = None
        if key and http_options is not None:
            client = genai.Client(api_key=key, http_options=http_options)
        elif key:
            client = genai.Client(api_key=key)
        elif http_options is not None:
            client = genai.Client(http_options=http_options)
        else:
            client = genai.Client()
        self._clients[key_index] = client
        return client

    def _tool_payload(self, tool_defs: list[dict[str, Any]]) -> list[types.Tool] | None:
        key = tuple(spec["name"] for spec in tool_defs)
        if self._gem_tools_key == key and self._gem_tools is not None:
            return self._gem_tools
        payload = tools_to_gemini(tool_defs)
        self._gem_tools_key = key
        self._gem_tools = payload
        return payload

    def _config(self, system: str, tool_defs: list[dict[str, Any]], max_output_tokens: int):
        kwargs: dict[str, Any] = {
            "system_instruction": system,
            "tools": self._tool_payload(tool_defs) or None,
            "max_output_tokens": max_output_tokens,
            "temperature": 0.2,
        }
        if self._thinking_ok and self.thinking_budget is not None:
            try:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
            except Exception:
                self._thinking_ok = False
        return types.GenerateContentConfig(**kwargs)

    async def warmup(self) -> None:
        """Create the HTTP client so the first user turn skips TLS setup."""
        try:
            self._client(0)
        except Exception:
            pass

    async def generate(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        tool_defs: list[dict[str, Any]],
        max_output_tokens: int,
        on_text: Any | None = None,
    ) -> LlmTurn:
        contents = messages_to_contents(messages)
        config = self._config(system, tool_defs, max_output_tokens)
        errors: list[str] = []
        model_order = [self.latched_model] + [m for m in self.models if m != self.latched_model] if self.latched_model else list(self.models)
        model_order = [m for m in model_order if m]
        key_indices = list(range(len(self.api_keys)))
        if self.latched_key_index is not None:
            key_indices = [self.latched_key_index] + [i for i in key_indices if i != self.latched_key_index]

        last_exc: Exception | None = None
        for key_i in key_indices:
            try:
                client = self._client(key_i)
            except Exception as exc:
                errors.append(f"client[{key_i}]: {exc}")
                last_exc = exc
                continue
            for model in model_order:
                try:
                    turn = await self._call(client, model, contents, config, on_text)
                    self.latched_model = model
                    self.latched_key_index = key_i
                    return turn
                except Exception as exc:
                    msg = str(exc)
                    if "thinking" in msg.lower() and self._thinking_ok:
                        self._thinking_ok = False
                        config = self._config(system, tool_defs, max_output_tokens)
                        try:
                            turn = await self._call(client, model, contents, config, on_text)
                            self.latched_model = model
                            self.latched_key_index = key_i
                            return turn
                        except Exception as exc2:
                            last_exc = exc2
                            errors.append(f"{model} key#{key_i + 1}: {exc2}")
                            continue
                    last_exc = exc
                    errors.append(f"{model} key#{key_i + 1}: {exc}")
                    continue
        joined = "; ".join(errors)[:800]
        raise RuntimeError(joined or str(last_exc) or "LLM call failed")

    async def stream_generate(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        tool_defs: list[dict[str, Any]],
        max_output_tokens: int,
    ):
        q: asyncio.Queue = asyncio.Queue()

        async def on_text(delta: str) -> None:
            await q.put(("text", delta))

        async def produce() -> None:
            try:
                turn = await self.generate(
                    system=system,
                    messages=messages,
                    tool_defs=tool_defs,
                    max_output_tokens=max_output_tokens,
                    on_text=on_text,
                )
                await q.put(("turn", turn))
            except Exception as exc:
                await q.put(("error", exc))

        task = asyncio.create_task(produce())
        try:
            while True:
                kind, payload = await q.get()
                yield kind, payload
                if kind in {"turn", "error"}:
                    break
        finally:
            await task

    async def _call(self, client: Any, model: str, contents: Any, config: Any, on_text: Any | None) -> LlmTurn:
        aio_models = getattr(getattr(client, "aio", None), "models", None)
        if self.stream and aio_models is not None and hasattr(aio_models, "generate_content_stream"):
            stream = aio_models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )
            if asyncio.iscoroutine(stream):
                stream = await stream
            text_parts: list[str] = []
            calls: list[dict[str, Any]] = []
            finish = ""

            async def _handle_chunk(chunk: Any) -> None:
                nonlocal finish
                piece = _chunk_text(chunk)
                if piece:
                    text_parts.append(piece)
                    if on_text:
                        maybe = on_text(piece)
                        if asyncio.iscoroutine(maybe):
                            await maybe
                parsed = self._parse(chunk, model)
                for call in parsed.tool_calls:
                    if call not in calls:
                        calls.append(call)
                if parsed.finish_reason:
                    finish = parsed.finish_reason

            if hasattr(stream, "__aiter__"):
                async for chunk in stream:
                    await _handle_chunk(chunk)
            else:
                for chunk in stream:
                    await _handle_chunk(chunk)
            truncated = "MAX_TOKENS" in finish.upper()
            text = "".join(text_parts).strip()
            uniq: list[dict[str, Any]] = []
            seen: set[str] = set()
            for call in calls:
                key = str(call.get("id") or "") or f"{call['name']}:{json_dumps(call.get('arguments'))}"
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(call)
            return LlmTurn(text=text, tool_calls=uniq, model=model, finish_reason=finish, truncated=truncated)

        if aio_models is not None:
            resp = await aio_models.generate_content(model=model, contents=contents, config=config)
        else:
            resp = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=contents,
                config=config,
            )
        turn = self._parse(resp, model)
        if on_text and turn.text:
            maybe = on_text(turn.text)
            if asyncio.iscoroutine(maybe):
                await maybe
        return turn

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


def _chunk_text(chunk: Any) -> str:
    try:
        piece = getattr(chunk, "text", None)
    except Exception:
        piece = None
    return str(piece) if piece else ""


def json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except Exception:
        return str(value)

