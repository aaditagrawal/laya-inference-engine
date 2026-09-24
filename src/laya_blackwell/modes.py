"""Select a public execution mode without loading unused native extensions."""


def create_engine(*, mode="balanced", **options):
    """Create the ordinary BF16 engine or the opt-in compiled SM120 engine.

    Fast mode requires ``laya-blackwell build-fast`` before construction. It
    uses additional GPU tables and compiles the short request shape on first
    use. Both engines expose predict, warmup, close and context management.
    """
    if mode == "balanced":
        from .engine import BlackwellEngine

        return BlackwellEngine(**options)
    if mode == "fast":
        from .fast import FastEngine

        return FastEngine(**options)
    raise ValueError("mode must be balanced or fast")
