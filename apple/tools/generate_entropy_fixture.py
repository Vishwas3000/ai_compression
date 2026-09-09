#!/usr/bin/env python3
"""Generate the small native-decoder fixture with the official me-tANS encoder."""

import json
from pathlib import Path

import numpy as np
import torch

from src.codec.entropy_coding.lib_wrappers.mans import ans


pmf = np.ones((1, 63), dtype=np.int64)
cumulative = np.cumsum(pmf, axis=1)
cdf = ((cumulative * 255 + (cumulative[:, -1:] >> 1)) // cumulative[:, -1:]).astype(np.uint8)
values = np.array([[0, 1, 2, 3, 7, 15, 31, 62, 5, 12, 24, 48, 60]], dtype=np.uint8)
encoder = ans.ANSEncoder(1024, 1)
encoder.encode_factorize(cdf, values.copy())
size = encoder.close()
memory = np.empty(size, dtype=np.uint8)
thread_sizes = np.empty(1, dtype=np.uint32)
encoder.get_memory(memory)
encoder.get_thread_sizes(thread_sizes)
fixture = {
    "factorized": {
        "cdf": cdf[0].tolist(),
        "values": values[0].tolist(),
        "memory": memory.tolist(),
        "thread_sizes": thread_sizes.tolist(),
    }
}

cache = torch.load(
    Path("src/codec/entropy_coding/lib_wrappers/mans/cache.pt"), weights_only=False
)
indexes = np.zeros(13, dtype=np.uint8)
values = np.array([0, 1, -1, 2, -2, 8, -8, 100, -100, 32766, -32767, 0, 3], dtype=np.int16)
masks = np.ones(13, dtype=np.bool_)
encoder = ans.ANSEncoder(1024, 1)
encoder.set_sgm_transitions(
    cache["encode_transition_file"].data,
    cache["bound_file"].data,
    cache["state_map_file"].data,
)
encoder.encode_sgm(indexes, values.copy(), masks)
size = encoder.close()
memory = np.empty(size, dtype=np.uint8)
encoder.get_memory(memory)
fixture["sgm"] = {
    "transitions": cache["decode_transition_file"][0].tolist(),
    "bound": int(cache["bound_file"][0]),
    "indexes": indexes.tolist(),
    "masks": masks.astype(np.uint8).tolist(),
    "values": values.tolist(),
    "memory": memory.tolist(),
}
print(json.dumps(fixture))
