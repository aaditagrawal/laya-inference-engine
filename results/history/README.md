Laya benchmark history

The main [warm-latency table](warm-progression.csv) follows the development stages
on the RTX 5070 Ti. Each latency is for a complete request. Sixteen-question
columns measure the whole batch, not one question.

GitHub renders each CSV as a table. Start with the warm-latency table above, then
use [paired comparisons](paired-gains.csv), [startup](startup.csv),
[concurrent serving](serving.csv), [HTTP](http.csv), or the
[RTX A6000 comparison](rtx-a6000.csv).

Each CSV corresponds to a table in that view. Timing boundaries, hardware,
validation limits and source paths are embedded in [tables.json](tables.json).
Startup, HTTP, concurrent serving and the RTX A6000 comparison have separate tables.

The history combines saved runs from 23–24 September 2026. It is not one paired
experiment. Use paired-gains.csv to assess measured incremental gains. Alternative
branches are not cumulative steps, and rejected candidates are not default-engine
performance claims. The final AOT startup matrix is the fully offline repeated
run; preliminary AOT loader diagnostics are intentionally excluded.

No benchmarks were rerun to build these tables. Source SHA256 values in
tables.json identify the inputs used. Regenerate the tables from the repository
root with `uv run --no-sync python scripts/summarize_history.py`.
