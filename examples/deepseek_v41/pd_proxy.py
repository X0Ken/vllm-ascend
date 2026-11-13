# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-prefill then D-generation using real named-plane KV metadata; preserve both roles."""

import contextlib
import hmac
import json
import os
import pathlib
import time
from types import MappingProxyType

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

CONFIG = MappingProxyType(json.loads(pathlib.Path(os.environ["QC_PD_PROXY_CONFIG"]).read_text()))
KEY = pathlib.Path(CONFIG["key_file"]).read_text().strip()


@contextlib.asynccontextmanager
async def lifespan(app):
    app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(1800, connect=20), trust_env=False)
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    for role in ["prefill_url", "decode_url"]:
        r = await app.state.client.get(CONFIG[role] + "/health")
        r.raise_for_status()
    return {"status": "ok"}


@app.post("/v1/completions")
@app.post("/v1/chat/completions")
async def completions(request: Request):
    if not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + KEY):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    data = await request.json()
    headers = {
        "Authorization": "Bearer " + KEY,
        "X-data-parallel-rank": request.headers.get("X-data-parallel-rank", "0"),
    }
    prefill = dict(data)
    prefill.update(stream=False, max_tokens=1, min_tokens=1, ignore_eos=True)
    prefill.pop("stream_options", None)
    prefill["kv_transfer_params"] = {"do_remote_decode": True, "do_remote_prefill": False}
    pstart = time.monotonic()
    p = await app.state.client.post(CONFIG["prefill_url"] + request.url.path, json=prefill, headers=headers)
    pend = time.monotonic()
    pms = 1000 * (pend - pstart)
    if p.status_code != 200:
        return Response(p.content, status_code=p.status_code, media_type="application/json")
    pobj = p.json()
    pusage = pobj.get("usage")
    assert pusage is not None, "Missing P-side usage"
    params = pobj.get("kv_transfer_params")
    if not params or not params.get("do_remote_prefill"):
        return JSONResponse({"error": "P did not publish real KV transfer metadata"}, status_code=502)
    data["kv_transfer_params"] = params
    # Preserve the P clock separately from D's stream and end-to-end TTFT.
    dstart = time.monotonic()
    dr = await app.state.client.send(
        app.state.client.build_request("POST", CONFIG["decode_url"] + request.url.path, json=data, headers=headers),
        stream=True,
    )
    if dr.status_code != 200:
        content = await dr.aread()
        await dr.aclose()
        return Response(content, status_code=dr.status_code, media_type="application/json")
    if not data.get("stream"):
        d = json.loads(await dr.aread())
        await dr.aclose()
        d["prefill_usage"] = pusage
        d["prefill_http_ms"] = pms
        return JSONResponse(d)

    async def events():
        dfirst = None
        try:
            async for line in dr.aiter_lines():
                if line.startswith("data:") and line[5:].strip() != "[DONE]":
                    d = json.loads(line[5:])
                    if dfirst is None and any(c.get("token_ids") for c in d.get("choices", [])):
                        dfirst = 1000 * (time.monotonic() - dstart)
                    d["prefill_usage"] = pusage
                    d["prefill_http_ms"] = pms
                    d["prefill_request_id"] = params.get("remote_request_id")
                    d["prefill_start_monotonic"] = pstart
                    d["prefill_end_monotonic"] = pend
                    if dfirst is not None:
                        d["decode_phase_first_token_ms"] = dfirst
                    line = "data: " + json.dumps(d, ensure_ascii=False)
                yield (line + "\n").encode()
        finally:
            await dr.aclose()

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/metrics")
async def metrics():
    r = await app.state.client.get(CONFIG["decode_url"] + "/metrics")
    return Response(r.content, status_code=r.status_code, media_type="text/plain")


@app.post("/reset_prefix_cache")
async def reset_prefix_cache(request: Request):
    if not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + KEY):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    results = []
    for role in ["prefill_url", "decode_url"]:
        r = await app.state.client.post(
            CONFIG[role] + "/reset_prefix_cache",
            headers={"Authorization": "Bearer " + KEY},
            params={"reset_external": "false"},
        )
        if r.status_code != 200:
            return Response(r.content, status_code=r.status_code, media_type="application/json")
        results.append({"role": role, "response": r.json()})
    return {"success": True, "roles": results}
