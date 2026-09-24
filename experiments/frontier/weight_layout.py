"""Owned, lossless BF16 weight banks installed before graph capture."""

import gc
import math
import time
import weakref

import torch
from cuda.bindings import driver as cuda

from .compression import Allocation, checked

_PROCESS_OWNERS = []


def granularity():
    torch.cuda.init()
    checked(cuda.cuInit(0))
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = torch.cuda.current_device()
    prop.allocFlags.compressionType = 0
    return {
        "minimum_bytes": checked(
            cuda.cuMemGetAllocationGranularity(
                prop,
                cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        ),
        "recommended_bytes": checked(
            cuda.cuMemGetAllocationGranularity(
                prop,
                cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
        ),
        "interpretation": "CUDA VMM allocation granularities only. Neither physical page size nor TLB mapping is inferred.",
    }


def original_model(engine):
    model = engine.base.model
    return model.original if hasattr(model, "original") else model


def execution_order(model):
    groups = ["net.encoder.embeddings"]
    for i in range(len(model.net.encoder.layers)):
        groups += [
            f"net.encoder.layers.{i}.{field}"
            for field in ("attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo")
        ]
    groups.append("net.type_emb")
    for i in range(len(model.net.head.layers)):
        groups += [
            f"net.head.layers.{i}.{field}"
            for field in (
                "self_attn.in_proj",
                "self_attn.out_proj",
                "linear1",
                "linear2",
            )
        ]
    groups += ["net.scorer.1", "net.scorer.3", "net.act_head.0", "net.act_head.2"]
    parameters = list(model.named_parameters())
    selected = []
    seen = set()
    for group in groups:
        for name, parameter in parameters:
            matches = (
                name in (group + "_weight", group + "_bias")
                if group.endswith(".in_proj")
                else name.startswith(group + ".")
            )
            if matches and parameter.dtype == torch.bfloat16:
                assert id(parameter) not in seen
                selected.append((name, parameter))
                seen.add(id(parameter))
    expected = {id(p) for _, p in parameters if p.dtype == torch.bfloat16}
    if seen != expected:
        raise ValueError(
            "Execution ordering missed BF16 parameters: "
            + repr(
                [
                    n
                    for n, p in parameters
                    if p.dtype == torch.bfloat16 and id(p) not in seen
                ]
            )
        )
    return selected


def untouched(model):
    return {
        **{
            "parameter:" + name: (p.data_ptr(), p._version, list(p.shape), str(p.dtype))
            for name, p in model.named_parameters()
            if p.dtype != torch.bfloat16
        },
        **{
            "buffer:" + name: (p.data_ptr(), list(p.shape), str(p.dtype))
            for name, p in model.named_buffers()
        },
    }


class WeightLayout:
    def __init__(self):
        self.owner = None
        self.bank = None
        self.references = []
        self.model_reference = None
        self.report = {}

    @torch.inference_mode()
    def install(self, engine, kind):
        if engine.adapter.graphs or engine.base.graphs:
            raise RuntimeError("Weight placement must precede all graph captures")
        if kind not in ("torch-bank", "vmm-bank", "vmm-2m-aligned"):
            raise ValueError(kind)
        model = original_model(engine)
        self.model_reference = weakref.ref(model)
        selected = execution_order(model)
        if model.training or any(p.requires_grad for _, p in selected):
            raise ValueError("Only frozen inference parameters may be moved")
        before = untouched(model)
        alignment = 2 * 1024 * 1024 if kind == "vmm-2m-aligned" else 256
        end = 0
        rows = []
        for name, parameter in selected:
            assert parameter.is_cuda and parameter.is_contiguous()
            offset = math.ceil(end / alignment) * alignment
            rows.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "offset_bytes": offset,
                    "bytes": parameter.nbytes,
                    "original_pointer": parameter.data_ptr(),
                }
            )
            end = offset + parameter.nbytes
        # Reserve alignment slack; the 2 MiB option aligns absolute addresses too.
        slack = alignment - 1
        elements = math.ceil((end + slack) / 2)
        torch.cuda.synchronize()
        started = time.perf_counter()
        if kind == "torch-bank":
            self.bank = torch.empty(elements, device="cuda", dtype=torch.bfloat16)
            allocation = {
                "logical_bytes": elements * 2,
                "allocated_bytes": elements * 2,
                "compression_granted": None,
                "kind": "torch.empty",
            }
        else:
            self.owner = Allocation((elements,), torch.bfloat16, compressed=False)
            self.bank = self.owner.tensor
            allocation = self.owner.report()
            assert allocation["compression_granted"] == 0
        self.bank.zero_()
        base_padding = (-self.bank.data_ptr()) % alignment
        torch.cuda.synchronize()
        allocated = time.perf_counter()
        mismatches = 0
        for row, (name, parameter) in zip(rows, selected):
            absolute_offset = base_padding + row["offset_bytes"]
            view = self.bank.narrow(0, absolute_offset // 2, parameter.numel()).view(
                parameter.shape
            )
            view.copy_(parameter)
            mismatches += int(
                (view.view(torch.int16) != parameter.view(torch.int16)).sum()
            )
            row.update(bank_offset_bytes=absolute_offset, pointer=view.data_ptr())
            assert view.data_ptr() % alignment == 0
            # Preserve Parameter identity for original/compiled module aliases.
            parameter.data = view
            assert parameter.data_ptr() == view.data_ptr()
            self.references.append((name, weakref.ref(parameter)))
        torch.cuda.synchronize()
        done = time.perf_counter()
        assert mismatches == 0
        assert untouched(model) == before
        self.report = {
            "kind": kind,
            "alignment_bytes": alignment,
            "allocation": allocation,
            "bank_pointer": self.bank.data_ptr(),
            "base_padding_bytes": base_padding,
            "parameter_count": len(rows),
            "parameter_bytes": sum(r["bytes"] for r in rows),
            "bank_used_bytes": end + base_padding,
            "weight_bit_mismatches": mismatches,
            "allocation_and_zero_seconds": allocated - started,
            "copy_verify_replace_seconds": done - allocated,
            "setup_seconds": done - started,
            "unchanged_buffers_and_non_bf16_parameters": True,
            "graph_count_at_install": 0,
            "parameters": rows,
            "order": "Encoder projections in execution order, followed by head projections, scorer, and action head; embeddings/type embedding first where BF16.",
        }
        return self.report

    def close(self):
        # The caller has closed and discarded its engine before entering here.
        torch.cuda.synchronize()
        torch._dynamo.reset()
        from torch._inductor.utils import clear_caches

        clear_caches()
        gc.collect()
        live = [name for name, ref in self.references if ref() is not None]
        model_live = (
            self.model_reference is not None and self.model_reference() is not None
        )
        if live or model_live:
            raise RuntimeError(
                f"Refusing unmap while model/parameters remain alive: {model_live}, {live}"
            )
        self.bank = None
        if self.owner is not None:
            self.owner.close()
            self.owner = None
        self.report["cleanup"] = {
            "model_dead": True,
            "all_parameters_dead": True,
            "compiler_cache_reset": True,
            "inductor_memory_caches_cleared": True,
            "unmapped_after_graph_and_model_destruction": True,
        }

    def retain_until_process_exit(self):
        """Keep VMM mapped while compiler-retained Python objects may exist.

        Only use in the dedicated one-candidate worker process. Allocation has
        no destructor that unmaps memory; CUDA context teardown at process exit
        reclaims it after the worker has synchronized and closed its graphs.
        """
        torch.cuda.synchronize()
        _PROCESS_OWNERS.append(self)
        self.report["cleanup"] = {
            "ownership": "dedicated worker process lifetime",
            "manual_unmap_performed": False,
            "bank_retained_until_process_exit": True,
            "worker_streams_synchronized": True,
            "candidate_engine_closed_before_retention": True,
            "storage_kind": "vmm" if self.owner is not None else "torch",
            "reason": "Compiled method references outlive model close and in-memory cache resets. Keep the mapping alive until CUDA context teardown instead of guessing reference lifetimes.",
        }
