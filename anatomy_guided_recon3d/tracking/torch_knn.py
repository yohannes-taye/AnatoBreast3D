"""Exact Euclidean KNN for inference when the upstream CUDA extension is absent.

This changes the neighbour-search implementation, not TAPIP3D's network or
weights. It supports Windows and older CUDA GPUs. Native pointops remains an
option on workstations where its extension has been compiled.
"""
from __future__ import annotations


def knnquery(nsample, xyz, new_xyz, offset, new_offset):
    import torch

    if new_xyz is None:
        new_xyz = xyz
    context_ends, query_ends = offset.tolist(), new_offset.tolist()
    if len(context_ends) != len(query_ends):
        raise ValueError("KNN batch offsets differ")
    indices, distances = [], []
    context_start = query_start = 0
    for context_end, query_end in zip(context_ends, query_ends):
        context = xyz[context_start:context_end].float()
        queries = new_xyz[query_start:query_end].float()
        if len(queries) and len(context) < nsample:
            raise ValueError("A point-cloud batch has fewer context points than the requested neighbours")
        # Bound the distance matrix independently of the number of query points.
        chunk_size = max(1, min(512, 4_000_000 // max(1, len(context))))
        for chunk in queries.split(chunk_size):
            d = torch.cdist(chunk[None], context[None], compute_mode="donot_use_mm_for_euclid_dist")[0]
            values, idx = d.topk(nsample, dim=-1, largest=False, sorted=True)
            indices.append((idx + context_start).to(torch.int32))
            distances.append(values)
        context_start, query_start = context_end, query_end
    if not indices:
        return (torch.empty((0, nsample), dtype=torch.int32, device=xyz.device),
                torch.empty((0, nsample), dtype=torch.float32, device=xyz.device))
    return torch.cat(indices), torch.cat(distances)


def install() -> None:
    """Install the inference-only shim inside the isolated worker process."""
    import sys
    import types
    import importlib

    root = importlib.import_module("third_party")
    parent = root
    # Upstream's functions/__init__.py imports a compiled top-level pointops2
    # package. Bypass that initializer as well as the extension module itself.
    for name in ("third_party.pointops2", "third_party.pointops2.functions"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
        setattr(parent, name.rsplit(".", 1)[1], package)
        parent = package
    module = types.ModuleType("third_party.pointops2.functions.pointops")
    module.knnquery = knnquery
    sys.modules[module.__name__] = module
    package.pointops = module
