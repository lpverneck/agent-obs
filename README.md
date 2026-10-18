# agent-obs

## To-dos

- OK - Add baggage user_id ou session_id
- OK - Add metrics and logs
- Add cost metric for example
- OK - Capturar prompts
- Add more tools
- OK - Add evals pipeline
- ajustar endereço de host das aplicações
- talvez criar uma branch com somente instrumentação manual/controlada



## Metrics

- `agent_obs_gen_ai_client_token_usage_sum`
- `agent_obs_kb_search_documents_total`
- `agent_obs_eval_score_sum` / `agent_obs_eval_score_count`

## Evals

Shared scorers in `agent/evals.py`, used both offline and online. Each scorer returns
`None` when the sample lacks what it needs, so each mode runs the ones that apply.

| Scorer | Type | Needs | Mode |
|---|---|---|---|
| `tool_selection` | deterministic | expected sources | offline |
| `source_recall` | deterministic | expected sources | offline |
| `correctness` | LLM judge | expected answer | offline |
| `answer_relevance` | LLM judge | query + answer | offline, online |
| `groundedness` | LLM judge | retrieved documents | online |

Each score is emitted as a `gen_ai.evaluation.result` event (GenAI semconv, to Loki) and
recorded in the `eval.score` histogram (to Prometheus), both tied to the evaluated trace.

### Offline experiments

Runs `agent/datasets/kb_agent.jsonl` against the running agent, prints a report compared
with the previous experiment and saves it to `agent/experiments/`.

```bash
cd agent
uv run opentelemetry-instrument --service_name eval-runner python experiment.py --name baseline
```

Each case is a trace in the `eval-runner` service, with the agent spans nested under it and
`session.id` set to the experiment id.

### Online evaluation

`/invoke` scores a share of the requests (`EVAL_ONLINE_SAMPLE_RATE`) in the background,
in an `evaluate kb_agent` span of the request trace. Requests from `user_id=eval-runner`
are skipped.

### Queries

- Loki: `{service_name=~"agent-service|eval-runner"} | gen_ai_evaluation_name != ""`
- Prometheus: `sum by (gen_ai_evaluation_name, eval_mode) (agent_obs_eval_score_sum) / sum by (gen_ai_evaluation_name, eval_mode) (agent_obs_eval_score_count)`
