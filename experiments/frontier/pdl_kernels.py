"""Isolated PDL clones of retained exact kernels, without changing source files."""

import ast
import contextlib
import hashlib
import inspect
import linecache
import re
import textwrap
from pathlib import Path
from unittest.mock import patch

import torch
import triton as tr
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from . import engine, head_gemm, head_install, matmul, mlp_geglu, reduce_norm, tma

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {}


def audit_compiled(compiled):
    """Check emitted PTX before the JIT returns a kernel for its first launch."""
    ptx = compiled.asm["ptx"]
    kernel_name = compiled.name
    if not kernel_name.startswith("pdl_"):
        return None
    mode = kernel_name.split("_")[1]
    wait = ptx.find("griddepcontrol.wait")
    if wait < 0 or "griddepcontrol.launch_dependents" not in ptx:
        raise RuntimeError(f"Missing dependency instructions: {kernel_name}")
    if not compiled.metadata.launch_pdl:
        raise RuntimeError(f"Missing launch_pdl metadata: {kernel_name}")
    # Pointer provenance for instructions before the unconditional wait. Weight
    # pointers are parameter 1; tensor descriptors expand the first argument
    # to five PTX parameters, making the second descriptor parameter 5.
    weight = 5 if kernel_name.endswith("tma_tma") else 1
    taint = {}
    reads = []
    before = ptx[:wait]
    labels_before = set(re.findall(r"(?m)^\s*([\w$]+):", before))
    for target in re.findall(r"\bbra(?:\.uni)?\s+([\w$]+)", before):
        if target not in labels_before:
            raise RuntimeError(f"Control flow may bypass wait: {kernel_name}: {target}")
    register = re.compile(r"%[a-zA-Z]+\d+")
    for number, raw in enumerate(before.splitlines(), 1):
        line = raw.split("//", 1)[0].strip()
        if not line or line.startswith((".", "$", "{")):
            continue
        line = re.sub(r"^@!?%\w+\s+", "", line)
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        op, operands = parts
        if op.startswith("st.global"):
            raise RuntimeError(f"Global store before wait: {kernel_name}:{number}")
        if op.startswith("ld.global") or (
            op.startswith("cp.async") and ".global" in op
        ):
            addresses = re.findall(r"\[([^\]]+)\]", operands)
            address = addresses[1] if op.startswith("cp.async") else addresses[0]
            origins = set().union(
                *(taint.get(r, set()) for r in register.findall(address))
            )
            if mode != "preload" or origins != {weight}:
                raise RuntimeError(
                    f"Unproven pre-wait read {kernel_name}:{number}: {origins}: {line}"
                )
            reads.append({"line": number, "instruction": line, "parameter": weight})
        if op.startswith(("st.", "cp.", "bar.", "mbarrier.", "bra", "griddep")):
            continue
        first, _, rest = operands.partition(",")
        destinations = register.findall(first)
        origins = set().union(*(taint.get(r, set()) for r in register.findall(rest)))
        origins.update(int(v) for v in re.findall(r"_param_(\d+)", rest))
        for dest in destinations:
            taint[dest] = origins
    return {
        "name": kernel_name,
        "wait_count": ptx.count("griddepcontrol.wait"),
        "trigger_count": ptx.count("griddepcontrol.launch_dependents"),
        "wait_before_first_global_load": not reads,
        "pre_wait_weight_reads": reads,
        "dependent_loads_after_wait": True,
        "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
        "cubin_sha256": hashlib.sha256(compiled.asm["cubin"]).hexdigest(),
        "n_regs": getattr(compiled, "n_regs", None),
        "shared_bytes": compiled.metadata.shared,
    }


def _compile(tree, namespace, tag):
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + "\n"
    filename = str(__file__) + ":" + tag
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102 - inspected local AST with fixed transformations
    return source


def _kernel(original, mode, tag):
    tree = ast.parse(textwrap.dedent(original.src))
    fn = tree.body[0]
    fn.decorator_list = []
    fn.name = "pdl_" + mode + "_" + tag
    if mode == "preload" and original.__name__ in {
        "_matmul",
        "_tma",
        "_project",
        "_gemm",
    }:
        # Preserve the existing K loop and FP32 accumulator order, but load
        # its first weight tile before waiting for producer activation data.
        loop = next(node for node in fn.body if isinstance(node, ast.For))
        if original.__name__ == "_matmul":
            setup = ast.parse("""k_first = part * steps * BK + kk
w_first = tl.load(W + n[None, :] * K + k_first[:, None], (n[None, :] < N) & (k_first[:, None] < K), 0)
gdc_wait()
gdc_launch_dependents()
""").body
            load = next(
                n
                for n in loop.body
                if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "w"
            )
            repl = ast.If(
                test=ast.Compare(
                    left=ast.Name("i", ast.Load()),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(0)],
                ),
                body=ast.parse("w = w_first").body,
                orelse=[load],
            )
            loop.body[loop.body.index(load)] = repl
        elif original.__name__ == "_gemm":
            setup = ast.parse("""k_first = k0 + kk
w_first = tl.load(W + n[None, :] * K + k_first[:, None], (n[None, :] < N) & (k_first[:, None] < K), 0)
gdc_wait()
gdc_launch_dependents()
""").body
            load = next(
                n
                for n in loop.body
                if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "w"
            )
            repl = ast.If(
                test=ast.Compare(
                    left=ast.Name("step", ast.Load()),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(0)],
                ),
                body=ast.parse("w = w_first").body,
                orelse=[load],
            )
            loop.body[loop.body.index(load)] = repl
        elif original.__name__ == "_tma":
            setup = ast.parse("""b_first = W.load([n0, start])
gdc_wait()
gdc_launch_dependents()
""").body
            load = next(
                n
                for n in loop.body
                if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "b"
            )
            loop.body[loop.body.index(load)] = ast.If(
                test=ast.Compare(
                    left=ast.Name("i", ast.Load()),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(0)],
                ),
                body=ast.parse("b = b_first").body,
                orelse=[load],
            )
        else:
            setup = ast.parse("""if ACCESS == 1:
    b_first = W.load([n0, 0]).T
else:
    wn_first = n // 2 + (n % 2) * 2624 if ACCESS == 2 else n
    b_first = tl.load(W + wn_first[None, :] * 1024 + kk[:, None], n[None, :] < 5248, 0)
gdc_wait()
gdc_launch_dependents()
""").body
            branch = loop.body[0]
            for nodes in [branch.body, branch.orelse]:
                load = next(
                    n
                    for n in nodes
                    if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id == "b"
                )
                nodes[nodes.index(load)] = ast.If(
                    test=ast.Compare(
                        left=ast.Name("start", ast.Load()),
                        ops=[ast.Eq()],
                        comparators=[ast.Constant(0)],
                    ),
                    body=ast.parse("b = b_first").body,
                    orelse=[load],
                )
        index = fn.body.index(loop)
        fn.body[index:index] = setup
    else:
        fn.body[0:0] = ast.parse("gdc_wait()").body
        if mode == "early":
            fn.body[1:1] = ast.parse("gdc_launch_dependents()").body
        else:
            fn.body.extend(ast.parse("gdc_launch_dependents()").body)
    namespace = dict(
        original.fn.__globals__,
        gdc_wait=gdc_wait,
        gdc_launch_dependents=gdc_launch_dependents,
        __name__=__name__,
    )
    source = _compile(tree, namespace, fn.name)
    kernel = tr.jit(namespace[fn.name])
    globals()[fn.name] = kernel
    return kernel, source


class _LaunchOptions(ast.NodeTransformer):
    def __init__(self, kernels):
        self.kernels = kernels

    def visit_Call(self, node):
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.kernels
        ):
            if any(k.arg == "launch_pdl" for k in node.keywords):
                raise ValueError("Unexpected existing PDL option")
            node.keywords.append(
                ast.keyword(arg="launch_pdl", value=ast.Constant(True))
            )
        return node


def _helper(original, kernels, mode, tag):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    fn = tree.body[0]
    fn.decorator_list = []
    _LaunchOptions(kernels).visit(tree)
    namespace = dict(original.__globals__, **kernels)
    source = _compile(tree, namespace, "helper_" + mode + "_" + tag)
    return namespace[fn.name], source


def create(mode):
    if mode in VARIANTS:
        return VARIANTS[mode]
    if mode not in {"late", "early", "preload"}:
        raise ValueError(mode)
    import laya_blackwell.kernels as base_kernels

    specs = [
        ("matmul", matmul, ["_matmul", "_reduce"], "matmul"),
        ("tma", tma, ["_tma", "_reduce"], "matmul"),
        ("mlp", mlp_geglu, ["_project"], "project"),
        ("head", head_gemm, ["_gemm", "_reduce"], "gemm"),
        ("rope", base_kernels, ["_rope"], "rope_qkv"),
    ]
    variant = {"kernels": {}, "helpers": {}, "sources": {}}
    for tag, module, names, helper_name in specs:
        kernels = {}
        for name in names:
            kernel, source = _kernel(getattr(module, name), mode, tag + name)
            kernels[name] = kernel
            variant["kernels"][tag + name] = kernel
            variant["sources"][tag + name] = source
        helper, source = _helper(getattr(module, helper_name), kernels, mode, tag)
        variant["helpers"][tag] = helper
        variant["sources"]["helper_" + tag] = source
    VARIANTS[mode] = variant
    return variant


@torch.library.custom_op(
    "laya_frontier_pdl::matrix", mutates_args=(), device_types="cuda"
)
def matrix(
    x: torch.Tensor,
    w: torch.Tensor,
    config: list[int],
    partial_bf16: bool,
    return_partials: bool,
    mode: str,
) -> torch.Tensor:
    return VARIANTS[mode]["helpers"]["matmul"](
        x, w, tuple(config), partial_bf16=partial_bf16, return_partials=return_partials
    )


@matrix.register_fake
def _(x, w, config, partial_bf16, return_partials, mode):
    shape = (*x.shape[:-1], w.shape[0])
    if return_partials:
        shape = (config[3], *shape)
    return torch.empty(shape, device=x.device, dtype=x.dtype)


@torch.library.custom_op(
    "laya_frontier_pdl::rope", mutates_args=(), device_types="cuda"
)
def rope(
    x: torch.Tensor, c: torch.Tensor, s: torch.Tensor, fp32: bool, mode: str
) -> torch.Tensor:
    return VARIANTS[mode]["helpers"]["rope"](x, c, s, fp32=fp32)


@rope.register_fake
def _(x, c, s, fp32, mode):
    return torch.empty_like(x)


@contextlib.contextmanager
def installed(mode):
    import laya_blackwell.model as model_module

    variant = create(mode)

    def selected_matrix(
        x,
        weight,
        config,
        scale=None,
        group=64,
        quant_mode=None,
        partial_bf16=False,
        return_partials=False,
    ):
        if scale is not None or quant_mode is not None:
            raise ValueError("PDL probe only supports retained BF16 matrices")
        return matrix(
            x, weight, [int(v) for v in config], partial_bf16, return_partials, mode
        )

    def selected_rope(x, c, s, *, fp32=False):
        return rope(x, c, s, fp32, mode)

    with contextlib.ExitStack() as stack:
        prior_hook = tr.knobs.runtime.jit_post_compile_hook

        def audited_compile(**kwargs):
            function = kwargs["fn"].jit_function
            if function.__name__.startswith("pdl_"):
                compiled = function.device_caches[kwargs["compile"]["device"]][0][
                    kwargs["key"]
                ]
                audit_compiled(compiled)
            if prior_hook is not None:
                return prior_hook(**kwargs)

        stack.enter_context(
            patch.object(tr.knobs.runtime, "jit_post_compile_hook", audited_compile)
        )
        for module, attr, value in [
            (engine, "matmul", selected_matrix),
            (reduce_norm, "matmul", selected_matrix),
            (tma, "matmul", variant["helpers"]["tma"]),
            (mlp_geglu, "project", variant["helpers"]["mlp"]),
            (head_install, "gemm", variant["helpers"]["head"]),
            (model_module, "rope_qkv", selected_rope),
        ]:
            stack.enter_context(patch.object(module, attr, value))
        yield variant


def evidence(mode, directory):
    variant = VARIANTS[mode]
    directory.mkdir(parents=True, exist_ok=True)
    result = {"generated_source_sha256": {}, "compiled": []}
    for name, source in variant["sources"].items():
        (directory / (name + ".py")).write_text(source)
        result["generated_source_sha256"][name] = hashlib.sha256(
            source.encode()
        ).hexdigest()
    for name, function in variant["kernels"].items():
        seen = set()
        for cache in function.device_caches.values():
            for compiled in cache[0].values():
                if compiled.hash in seen:
                    continue
                seen.add(compiled.hash)
                ptx = compiled.asm["ptx"]
                stem = name + "-" + compiled.hash[:12]
                (directory / (stem + ".ptx")).write_text(ptx)
                (directory / (stem + ".ttir")).write_text(compiled.asm["ttir"])
                row = {
                    **audit_compiled(compiled),
                    "tag": name,
                    "hash": compiled.hash,
                    "launch_pdl": compiled.metadata.launch_pdl,
                }
                result["compiled"].append(row)
    if not result["compiled"]:
        raise RuntimeError("No PDL kernel compilation captured")
    return result
