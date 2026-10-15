# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Detect scheduled multimodal spans independently of encoder cache misses."""


def has_scheduled_multimodal_tokens(requests, num_scheduled_tokens):
    for req_id, count in num_scheduled_tokens.items():
        if count <= 0:
            continue
        request = requests[req_id]
        start = request.num_computed_tokens
        end = start + count
        for feature in request.mm_features or ():
            position = feature.mm_position
            if position.offset < end and position.offset + position.length > start:
                return True
    return False
