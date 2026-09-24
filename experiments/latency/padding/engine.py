"""Change only batch/sequence padding before any graph capture."""

from types import MethodType

from experiments.native.engine import ExperimentalEngine

POLICIES = {"stock", "batch-exact", "batch-compatible", "sequence-32", "dense-32"}


def shape_for(base, prepared, policy):
    if policy not in POLICIES:
        raise ValueError(f"Unknown shape policy: {policy}")
    count = len(prepared.items)
    if not 0 < count <= base.max_questions:
        raise ValueError("Shape requires a nonempty, bounded prepared request")
    length = max(len(item["ids"]) for item in prepared.items)
    options = max(2, max(len(item["markers"]) for item in prepared.items))
    max_len = base.agent.cfg.get("max_len", 512)
    if length > max_len:
        raise ValueError("Prepared sequence exceeds the model maximum")
    batch = (
        count
        if policy in {"batch-exact", "batch-compatible", "dense-32"}
        else min(base.max_questions, 1 << (count - 1).bit_length())
    )
    if policy in {"sequence-32", "dense-32"}:
        length = min(max_len, max(32, (length + 31) // 32 * 32))
    else:
        length = next(
            (size for size in base.sequence_buckets if length <= size <= max_len),
            max_len,
        )
    # Keep option padding unchanged so output width and head work are comparable.
    return batch, length, 1 << (options - 1).bit_length()


class DenseEngine(ExperimentalEngine):
    def __init__(self, *, shape_policy="dense-32", **kwargs):
        if shape_policy not in POLICIES:
            raise ValueError(f"Unknown shape policy: {shape_policy}")
        super().__init__(**kwargs)
        self.shape_policy = shape_policy

        def shape(base, prepared):
            return shape_for(base, prepared, shape_policy)

        self.base._shape = MethodType(shape, self.base)
        if shape_policy == "batch-compatible":

            def graph_key(base, prepared):
                candidate_shape = base._shape(prepared)
                stock_batch, stock_length, _ = shape_for(base, prepared, "stock")
                unmasked = len(prepared.items) == stock_batch and all(
                    len(item["ids"]) == stock_length for item in prepared.items
                )
                return (*candidate_shape, unmasked)

            self.base._graph_key = MethodType(graph_key, self.base)
