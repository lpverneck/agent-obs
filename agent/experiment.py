import argparse
import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
import httpx
from opentelemetry import propagate, trace
from opentelemetry.trace import StatusCode
from evals import evaluate


AGENT_URL = os.getenv(
    "AGENT_URL", f"http://localhost:{os.getenv('AGENT_PORT', '8002')}"
)
JAEGER_UI_URL = os.getenv(
    "JAEGER_UI_URL", f"http://localhost:{os.getenv('JAEGER_UI_PORT', '16686')}"
)
# requests from this user are skipped by the agent's online evaluation
EVAL_USER_ID = "eval-runner"
EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
# shared knowledge item dataset in the repo `data/` dir, also used to seed the knowledge base
DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "rag_sample.csv"

tracer = trace.get_tracer(__name__)


def load_cases(path: Path) -> list[dict]:
    """One case per knowledge item: its sample question, identified by the document id."""
    with path.open(newline="") as file:
        return [
            {
                # same id rule as app/api.py, so the case id is the seeded document id
                "id": re.sub(r"[^a-z0-9]+", "-", row["ki_topic"].lower()).strip("-"),
                # some sample questions are wrapped in literal quotes
                "query": row["sample_question"].strip().strip('"'),
            }
            for row in csv.DictReader(file)
        ]


def run_case(case: dict, experiment_id: str) -> dict:
    """Ask the agent one dataset question and score its answer, in a trace of its own."""
    attributes = {"eval.experiment.id": experiment_id, "eval.case.id": case["id"]}
    with tracer.start_as_current_span(
        f"eval_case {case['id']}", attributes=attributes
    ) as span:
        row = {
            "id": case["id"],
            "query": case["query"],
            "trace_id": trace.format_trace_id(span.get_span_context().trace_id),
        }
        try:
            # propagate the trace context so the agent spans join this case's trace
            headers = {}
            propagate.inject(headers)
            start = time.perf_counter()
            # the experiment id as session id tags every agent span via baggage
            response = httpx.post(
                f"{AGENT_URL}/invoke",
                json={
                    "query": case["query"],
                    "session_id": experiment_id,
                    "user_id": EVAL_USER_ID,
                },
                headers=headers,
                timeout=120.0,
            )
            response.raise_for_status()
            row["latency_s"] = round(time.perf_counter() - start, 2)
            answer = response.json()
            score = evaluate(
                case["query"], answer["answer"], mode="offline", **attributes
            )
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            return {**row, "error": f"{type(exc).__qualname__}: {exc}"}
        return {
            **row,
            "answer": answer["answer"],
            "sources": answer["sources"],
            "scores": {score.name: score.model_dump()},
        }


def summarize(rows: list[dict]) -> dict[str, float]:
    """Mean of each scorer across the cases where it applied, plus mean latency."""
    values: dict[str, list[float]] = {}
    for row in rows:
        for name, score in row.get("scores", {}).items():
            values.setdefault(name, []).append(score["value"])
        if "latency_s" in row:
            values.setdefault("latency_s", []).append(row["latency_s"])
    return {name: round(mean(scores), 3) for name, scores in values.items()}


def latest_experiment() -> dict | None:
    """Most recent saved experiment, used as the baseline for comparison."""
    files = sorted(
        EXPERIMENTS_DIR.glob("*.json"), key=lambda path: path.stat().st_mtime
    )
    return json.loads(files[-1].read_text()) if files else None


def print_report(experiment: dict, baseline: dict | None) -> None:
    rows, summary = experiment["cases"], experiment["summary"]
    columns = [name for name in ("answer_relevance", "latency_s") if name in summary]
    id_width = max(len(row["id"]) for row in rows) + 2
    id_width = max(id_width, len("baseline") + 2)

    def line(label: str, cells: list[str]) -> str:
        return label.ljust(id_width) + "".join(
            cell.rjust(len(column) + 2) for cell, column in zip(cells, columns)
        )

    def cell(row: dict, column: str) -> str:
        if column == "latency_s":
            return f"{row['latency_s']:.2f}" if "latency_s" in row else "-"
        score = row.get("scores", {}).get(column)
        return f"{score['value']:.2f}" if score else "-"

    errors = [row for row in rows if "error" in row]
    print(
        f"\nExperiment {experiment['experiment_id']} "
        f"({len(rows)} cases, {len(errors)} errors)\n"
    )
    print(line("case", columns))
    for row in rows:
        print(line(row["id"], [cell(row, column) for column in columns]))
    print(line("mean", [f"{summary[column]:.2f}" for column in columns]))
    if baseline:
        deltas = [
            f"{summary[column] - baseline['summary'][column]:+.2f}"
            if column in baseline["summary"]
            else "-"
            for column in columns
        ]
        print(line("baseline", deltas))
        print(f"\n(baseline: {baseline['experiment_id']})")

    for row in errors:
        print(f"\nerror in {row['id']}: {row['error']}")
    print("\nTraces:")
    for row in rows:
        print(f"  {row['id']}: {JAEGER_UI_URL}/trace/{row['trace_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an offline agent experiment.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="CSV file with ki_topic and sample_question columns.",
    )
    parser.add_argument(
        "--name",
        default="experiment",
        help="Experiment name, e.g. the change under test.",
    )
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    experiment_id = f"{args.name}-{datetime.now():%Y%m%d-%H%M%S}"
    baseline = latest_experiment()

    rows = []
    for case in cases:
        print(f"running {case['id']}...")
        rows.append(run_case(case, experiment_id))

    experiment = {
        "experiment_id": experiment_id,
        "dataset": str(args.dataset),
        "agent_url": AGENT_URL,
        "summary": summarize(rows),
        "cases": rows,
    }
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    (EXPERIMENTS_DIR / f"{experiment_id}.json").write_text(
        json.dumps(experiment, indent=2)
    )
    print_report(experiment, baseline)


if __name__ == "__main__":
    main()
