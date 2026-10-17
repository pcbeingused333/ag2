# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Gemini 3 can send a function call's thought signature on a part of its own.

The signature then arrives on a part with no text and no function call, before
the functionCall part. It has to end up on that function call, or the call is
replayed without it and Gemini 3 rejects the next request.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fast_depends.use import SerializerCls
from google.genai import types

from ag2 import Context
from ag2.config.gemini.gemini_client import GeminiClient
from ag2.config.gemini.mappers import convert_messages

SIGNATURE = b"\x12\x34signature"


def _response(*parts: types.Part) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))]
    )


def _call(name: str, call_id: str, **args: str) -> types.Part:
    return types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args))


@pytest.fixture
def client() -> GeminiClient:
    with patch("ag2.config.gemini.gemini_client.genai.Client"):
        return GeminiClient(model="gemini-3-pro-preview", api_key="test-key")


@pytest.fixture
def context() -> Context:
    return Context(stream=AsyncMock())


def _replayed_signatures(response) -> list[bytes | None]:
    contents = convert_messages([response], SerializerCls)
    return [part.thought_signature for content in contents for part in content.parts or () if part.function_call]


@pytest.mark.asyncio
async def test_stream_attaches_standalone_signature_to_the_next_call(client: GeminiClient, context: Context) -> None:
    async def stream():
        yield _response(types.Part(thought_signature=SIGNATURE))
        yield _response(_call("get_weather", "c1", city="Paris"))

    result = await client._process_stream(stream(), context)

    assert [call.thought_signature for call in result.tool_calls.calls] == [SIGNATURE]
    assert _replayed_signatures(result) == [SIGNATURE]


@pytest.mark.asyncio
async def test_response_attaches_standalone_signature_to_the_next_call(client: GeminiClient, context: Context) -> None:
    response = _response(types.Part(thought_signature=SIGNATURE), _call("get_weather", "c1", city="Paris"))

    result = await client._process_response(response, context)

    assert [call.thought_signature for call in result.tool_calls.calls] == [SIGNATURE]
    assert _replayed_signatures(result) == [SIGNATURE]


@pytest.mark.asyncio
async def test_standalone_signature_goes_to_the_first_parallel_call_only(
    client: GeminiClient, context: Context
) -> None:
    async def stream():
        yield _response(
            types.Part(thought_signature=SIGNATURE),
            _call("get_weather", "c1", city="Paris"),
            _call("get_weather", "c2", city="London"),
        )

    result = await client._process_stream(stream(), context)

    assert [call.thought_signature for call in result.tool_calls.calls] == [SIGNATURE, None]


@pytest.mark.asyncio
async def test_signature_on_the_call_itself_is_kept(client: GeminiClient, context: Context) -> None:
    own = b"own-signature"
    response = _response(
        types.Part(thought_signature=SIGNATURE),
        types.Part(
            function_call=types.FunctionCall(id="c1", name="get_weather", args={"city": "Paris"}),
            thought_signature=own,
        ),
    )

    result = await client._process_response(response, context)

    assert [call.thought_signature for call in result.tool_calls.calls] == [own]
