# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError

from vllm_ascend.patch.platform.patch_request_validation import validation_exception_handler


def response(errors, log=False):
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(args=SimpleNamespace(log_error_stack=log))))
    return asyncio.run(validation_exception_handler(request, RequestValidationError(errors)))


@pytest.mark.parametrize("size", [1024, 1048576])
def test_responses_invalid_union_does_not_echo_input(size):
    secret = "REJECTED_PRIVATE_PAYLOAD_" + "x" * size
    with pytest.raises(ValidationError) as caught:
        ResponsesRequest.model_validate(
            {"model": "v41", "input": [{"role": "assistant", "content": [{"type": "text", "text": secret}]}]}
        )
    result = response(caught.value.errors())
    assert result.status_code == 400
    assert len(result.body) < 16384
    assert b"REJECTED_PRIVATE_PAYLOAD" not in result.body
    assert b"additional errors omitted" in result.body
    assert json.loads(result.body)["error"]["param"].startswith("input")


def test_context_parameter_and_logging_do_not_render_exception():
    class Unprintable:
        def __str__(self):
            raise AssertionError("payload or exception context rendered")

    error = {
        "loc": ("body", "input"),
        "msg": "Invalid value",
        "input": Unprintable(),
        "ctx": {"error": VLLMValidationError("Invalid input", parameter="input")},
    }
    result = response([error], log=True)
    assert json.loads(result.body)["error"]["param"] == "input"


def test_bounded_location_message_and_many_errors():
    error = {"loc": ("body", "z" * 100000), "msg": "m" * 100000, "input": "private"}
    result = response([error] * 10000)
    assert len(result.body) < 16384
    assert b"9992 additional errors omitted" in result.body


def test_empty_and_string_errors():
    assert response([]).status_code == 400
    assert b"Unsupported Media Type" in response(["Unsupported Media Type"]).body


def test_fastapi_rejects_invalid_body_and_accepts_valid_body():
    app = FastAPI()
    app.state.args = SimpleNamespace(log_error_stack=False)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    @app.post("/v1/responses")
    async def endpoint(request: ResponsesRequest):
        return {"model": request.model}

    async def check():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            bad = await client.post(
                "/v1/responses",
                json={
                    "model": "v41",
                    "input": [{"role": "user", "content": [{"type": "text", "text": "x" * 1048576}]}],
                },
            )
            assert bad.status_code == 400 and len(bad.content) < 16384
            good = await client.post("/v1/responses", json={"model": "v41", "input": "hello"})
            assert good.status_code == 200 and good.json() == {"model": "v41"}

    asyncio.run(check())
