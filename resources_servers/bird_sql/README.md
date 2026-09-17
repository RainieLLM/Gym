# BIRD SQL Resource Server

Execution-based text-to-SQL verification on the BIRD dev split.

Binary reward: `1.0` if the predicted SQL produces the same result set as the
ground-truth query on the per-`db_id` SQLite database, `0.0` otherwise. No LLM
judge.

## Dataset

- **1534 tasks** across 11 SQLite databases from the BIRD dev split
- Each task has a `difficulty` label: `simple`, `moderate`, or `challenging`

## Database download

SQLite databases are downloaded automatically on first server startup:

```
https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
```

They are cached at `resources_servers/bird_sql/.bird_sql/dev_20240627/dev_databases/`
and re-used on subsequent runs.

To trigger the download manually:

```python
from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql
ensure_bird_sql()
```

## Example usage

### Running servers

```bash
gym env start \
    --model-type vllm_model \
    --resources-server bird_sql
```

Requires `policy_base_url` / `policy_api_key` / `policy_model_name` in
`env.yaml` (or passed as CLI overrides).

### Collecting rollouts (5-example smoke test)

```bash
gym eval run --no-serve \
    --agent bird_sql_simple_agent \
    --input resources_servers/bird_sql/data/example.jsonl \
    --output results/bird_sql_rollouts.jsonl \
    --num-repeats 1
```

For a full BIRD dev run, see `benchmarks/birdbench/README.md`.

## Verification flow

1. Extract SQL from the last ` ```sql ... ``` ` code block. Keep comments and
   internal line breaks so SQLite can read the query as written.
2. Execute the reference SQL against the per-`db_id` SQLite database.
3. If the reference query succeeds, execute the predicted SQL against that database.
4. Compare result sets via unordered set equality.
5. Return reward `1.0` on match, `0.0` otherwise (or on any extraction /
   execution error).

### Result codes

The `failure_reason` field explains the result of each verifier call.
The binary reward stays the same for every case. SQL error text does not appear
in the response.

| Code | Meaning |
| --- | --- |
| `none` | Both queries ran and their result sets match. |
| `no_sql_extracted` | The model response has no SQL code block. No query ran. |
| `result_mismatch` | Both queries ran, but their result sets differ. |
| `execution_error` | The predicted SQL caused a SQLite error. |
| `execution_timeout` | The predicted SQL exceeded the time limit. |
| `gold_execution_error` | The reference SQL caused a SQLite error. |
| `gold_execution_timeout` | The reference SQL exceeded the time limit. |
| `unknown_error` | The verifier could not classify another error. |

The verifier runs the reference SQL first. If it fails, the predicted SQL does
not run. `diagnostics/total` counts all rollouts in a metrics report.
`diagnostics/<code>` counts each result code. The `none` code marks correct
answers. `diagnostics/unclassified` counts older rollouts without a known code.
The accuracy metrics keep their original denominator.

For older results, `execution_error` also covered a result mismatch or a
predicted SQL timeout. Those cases now have separate codes. The scores and the
result-set comparison rule do not change.

## Input JSONL format

Each task must provide:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "<question + schema dump>"}
    ]
  },
  "question": "...",
  "gt_sql": "SELECT ...",
  "sql_context": "CREATE TABLE ...;",
  "db_id": "california_schools",
  "difficulty": "simple",
  "id": 0
}
```

`question` / `gt_sql` / `db_id` are used by the server; `difficulty` / `id`
are passed through to metrics and output.

## Metrics

Overall:

- `pass@k/accuracy` — fraction of tasks with at least one correct rollout in k tries
- `pass@1[avg-of-k]/accuracy` — mean accuracy across k rollouts, averaged over tasks

Per-difficulty subsets (via `compute_subset_metrics(field="difficulty")`):

- `simple/pass@1[avg-of-k]/accuracy`
- `moderate/pass@1[avg-of-k]/accuracy`
- `challenging/pass@1[avg-of-k]/accuracy`
