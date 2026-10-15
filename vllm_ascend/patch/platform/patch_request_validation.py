# SPDX-License-Identifier: Apache-2.0
"""Bound validation diagnostics without rendering rejected request payloads.

Responses inputs use large unions. One invalid item can produce hundreds of
errors referencing the same input; formatting those dictionaries synchronously
can stall health checks and metrics in the API event loop.
"""

import sys
from http import HTTPStatus

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from vllm.entrypoints.serve.utils import server_utils
from vllm.exceptions import VLLMValidationError

MAX_ERRORS = 8
MAX_FIELD_CHARS = 256
MAX_LOCATION_PARTS = 16


def _text(value):
    # Do not stringify arbitrary objects, containers, or exception contexts.
    return value[:MAX_FIELD_CHARS] if isinstance(value, str) else ""


def _location(error):
    loc = error.get("loc")
    if not isinstance(loc, (list, tuple)):
        return ""
    parts = [
        _text(part) if isinstance(part, str) else part
        for part in loc[:MAX_LOCATION_PARTS]
        if isinstance(part, (str, int))
    ]
    return server_utils.clean_loc_for_param(parts)[:MAX_FIELD_CHARS]


async def validation_exception_handler(req, exc: RequestValidationError):
    errors = exc.errors()
    param = None
    for error in errors:
        if not isinstance(error, dict):
            continue
        context = error.get("ctx")
        context_error = context.get("error") if isinstance(context, dict) else None
        if isinstance(context_error, VLLMValidationError):
            param = _text(context_error.parameter) or None
            break
    if param is None and errors and isinstance(errors[0], dict):
        param = _location(errors[0]) or None

    count = len(errors)
    label = "error" if count == 1 else "errors"
    lines = [f"{count} validation {label}:"]
    for error in errors[:MAX_ERRORS]:
        if isinstance(error, dict):
            location = _location(error)
            message = _text(error.get("msg")) or "Invalid request value"
            lines.append(f"  {location}: {message}")
        else:
            lines.append("  " + (_text(error) or "Invalid request value"))
    if count > MAX_ERRORS:
        lines.append(f"  {count - MAX_ERRORS} additional errors omitted")
    if req.app.state.args.log_error_stack:
        # Logging the exception would render its input dictionaries again.
        server_utils.logger.warning("Request validation failed: %s errors; param=%s", count, param)
    response = server_utils.create_error_response(
        message="\n".join(lines),
        err_type="BadRequestError",
        status_code=HTTPStatus.BAD_REQUEST,
        param=param,
    )
    return JSONResponse(content=response.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


def install_validation_handler():
    server_utils.validation_exception_handler = validation_exception_handler
    # Handle both frontend import orders without importing the API server here.
    api_server = sys.modules.get("vllm.entrypoints.openai.api_server")
    if api_server is not None:
        api_server.validation_exception_handler = validation_exception_handler


install_validation_handler()
