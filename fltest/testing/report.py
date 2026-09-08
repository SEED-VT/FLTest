"""Reporting for runs and test outcomes (JSON file + console summary)."""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TestOutcome:
    """A single PASS/FAIL check produced by a tester."""

    test_type: str          # "differential" | "metamorphic"
    name: str               # what was checked
    status: str             # "PASS" | "FAIL" | "SKIP"
    detail: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


def summarize(outcomes: List[TestOutcome]) -> Dict[str, int]:
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    return counts


def write_report(
    path: str | Path,
    title: str,
    runs: Optional[List[Dict[str, Any]]] = None,
    outcomes: Optional[List[TestOutcome]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a JSON report and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": title,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "runs": runs or [],
        "outcomes": [asdict(o) for o in (outcomes or [])],
        "summary": summarize(outcomes or []),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def print_outcomes(title: str, outcomes: List[TestOutcome]) -> bool:
    """Print a console table of outcomes. Returns True if all passed (no FAILs)."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    symbol = {"PASS": "✓", "FAIL": "✗", "SKIP": "–"}
    for o in outcomes:
        print(f"  [{symbol.get(o.status, '?')} {o.status}] {o.test_type}: {o.name}")
        if o.detail:
            print(f"        {o.detail}")
    counts = summarize(outcomes)
    print(f"{'-' * 70}")
    print(f"  PASS={counts['PASS']}  FAIL={counts['FAIL']}  SKIP={counts['SKIP']}")
    print(f"{'=' * 70}\n")
    return counts["FAIL"] == 0


# ---------------------------------------------------------------------------
# Run-matrix console table
# ---------------------------------------------------------------------------

#: Resolved run parameters that may become table columns, in display order.
_PARAM_COLUMNS = [
    ("framework", "framework"),
    ("dataset", "dataset"),
    ("data_distribution", "distribution"),
    ("dirichlet_alpha", "alpha"),
    ("classes_per_partition", "cls/client"),
    ("model_name", "model"),
    ("aggregation", "aggregation"),
    ("num_clients", "clients"),
    ("num_rounds", "rounds"),
    ("client_epochs", "epochs"),
    ("client_lr", "lr"),
    ("client_batch_size", "batch"),
    ("optimizer", "optim"),
    ("seed", "seed"),
    ("attacks", "attack"),
    ("defenses", "defense"),
]

#: Metrics shown first when present; anything else in ``final`` follows alphabetically.
_METRIC_ORDER = [
    "accuracy",
    "loss",
    "attack_success_rate",
    "model_replacement_scale",
    "per_client_acc_mean",
    "per_client_acc_min",
    "reconstruction_mse",
    "reconstruction_psnr",
    "label_recovery",
]

#: Fingerprint metric kept out of the table; it is still written to the JSON report.
_METRICS_HIDDEN = {"gm_weight_sum"}

#: Short column headers, so a run with several plugin metrics still fits a terminal.
_METRIC_HEADERS = {
    "attack_success_rate": "asr",
    "model_replacement_scale": "mr-scale",
    "per_client_acc_mean": "pc-mean",
    "per_client_acc_min": "pc-min",
    "reconstruction_mse": "rec-mse",
    "reconstruction_psnr": "rec-psnr",
    "label_recovery": "label-rec",
    "membership_inference_auc": "mia-auc",
    "membership_loss_gap": "mia-gap",
}

#: What each shortened column means, printed under the table for the metrics in play. A
#: reader should not have to open the source to learn what `asr` or `mia-auc` is.
_METRIC_GLOSS = {
    "accuracy": "top-1 accuracy of the global model on the held-out test set",
    "loss": "mean cross-entropy on the same test set",
    "attack_success_rate": "attack success rate, the share of triggered inputs predicted as the attacker's target label",
    "model_replacement_scale": "model-replacement scale applied to the malicious client delta",
    "per_client_acc_mean": "per-client accuracy, averaged over clients",
    "per_client_acc_min": "per-client accuracy of the worst-served client",
    "reconstruction_mse": "pixel mean-squared error of the DLG reconstruction, lower means a better reconstruction and worse privacy",
    "reconstruction_psnr": "peak signal-to-noise ratio of that reconstruction, higher means a better reconstruction",
    "label_recovery": "share of the victim's labels the attack recovered",
    "membership_inference_auc": "membership inference AUC, the chance a training sample looks more member-like than a held-out one; 0.5 is no leakage",
    "membership_loss_gap": "held-out loss minus training loss, the overfitting gap membership inference exploits",
}

#: Width the fixed-settings block wraps at, independent of how wide the table is.
_WRAP = 96


def _fmt_plugins(value) -> str:
    """Render an attack/defense list as names, or ``none`` when empty."""
    if not value:
        return "none"
    return ",".join(p.get("name", "?") if isinstance(p, dict) else str(p) for p in value)


def _fmt_plugins_verbose(value) -> str:
    """Render an attack/defense list with its parameters, for the fixed-settings block."""
    if not value:
        return "none"
    out = []
    for p in value:
        if not isinstance(p, dict):
            out.append(str(p))
            continue
        params = ", ".join(f"{k}={v}" for k, v in (p.get("params") or {}).items())
        targets = p.get("target_clients")
        label = p.get("name", "?")
        if params:
            label += f"({params})"
        if targets:
            label += f" on clients {list(targets)}"
        out.append(label)
    return "; ".join(out)


def _fmt_param(key: str, value) -> str:
    if key in ("attacks", "defenses"):
        return _fmt_plugins(value)
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _fmt_metric(key: str, value) -> str:
    if value is None:
        return "-"
    if key == "reconstruction_psnr":
        return f"{value:.1f}"
    return f"{value:.4f}"


def print_run_matrix(
    name: str,
    results: List[Any],
    total_duration: float = 0.0,
    report_path: Optional[Any] = None,
) -> None:
    """Print the run matrix as an aligned table.

    Parameters shared by every run are printed once above the table, and only the
    parameters that actually differ become columns. A fuzzed grid therefore shows what
    varies across its cells instead of repeating the same values on every row.
    """
    if not results:
        print(f"\nRUN MATRIX: {name}\n  no runs were produced by this config.")
        return

    params = [getattr(r, "params", {}) or {} for r in results]
    available = [(k, h) for k, h in _PARAM_COLUMNS if any(k in p for p in params)]

    def _cell(p, key):
        return _fmt_param(key, p.get(key))

    varying = [(k, h) for k, h in available if len({_cell(p, k) for p in params}) > 1]
    fixed = [(k, h) for k, h in available if (k, h) not in varying]

    # Metric columns: the known order first, then anything else a plugin recorded.
    present = {k for r in results for k in (r.final or {})} - _METRICS_HIDDEN
    metrics = [m for m in _METRIC_ORDER if m in present]
    metrics += sorted(present - set(metrics))

    failed = [r for r in results if r.status != "success"]

    headers = ["run"] + [h for _, h in varying]
    headers += [_METRIC_HEADERS.get(m, m) for m in metrics] + ["time"]
    if failed:
        headers.insert(1, "status")

    rows = []
    for r, p in zip(results, params):
        row = [r.run_name]
        if failed:
            row.append("ok" if r.status == "success" else r.status)
        row += [_cell(p, k) for k, _ in varying]
        row += [_fmt_metric(m, (r.final or {}).get(m)) for m in metrics]
        row.append(f"{r.duration_seconds:.1f}s")
        rows.append(row)

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    # Left-align the label columns, right-align the numeric ones.
    numeric_from = len(headers) - len(metrics) - 1

    def _line(cells):
        out = []
        for i, c in enumerate(cells):
            out.append(c.rjust(widths[i]) if i >= numeric_from else c.ljust(widths[i]))
        return "  " + "  ".join(out).rstrip()

    total_width = max(len(_line(headers)), 60)
    rule = "=" * total_width

    print(f"\n{rule}\nRUN MATRIX: {name}")
    if fixed:
        settings = []
        for key, header in fixed:
            value = params[0].get(key)
            if key in ("attacks", "defenses"):
                settings.append(f"{header}={_fmt_plugins_verbose(value)}")
                continue
            shown = _fmt_param(key, value)
            if shown == "-":
                continue  # knob does not apply to this run, e.g. alpha under IID
            settings.append(f"{header}={shown}")
        body = "  ".join(settings)
        for i, chunk in enumerate(textwrap.wrap(body, _WRAP) or [""]):
            print(f"{'same for all runs:' if i == 0 else ' ' * 18} {chunk}")
    print(rule)
    print(_line(headers))
    print("-" * total_width)
    for row in rows:
        print(_line(row))
    print(rule)

    # Per-round trace of the headline metric, so the table shows how a run got where it
    # did rather than only where it ended.
    traced = [r for r in results if len(getattr(r, "history", None) or {}) > 1]
    if traced and metrics:
        headline = metrics[0]
        print(f"per-round {headline}:")
        for r in traced:
            rounds = sorted(r.history, key=lambda k: int(k))
            values = [r.history[k].get(headline) for k in rounds]
            shown = [_fmt_metric(headline, v) for v in values if v is not None]
            if len(shown) > 12:
                shown = shown[:6] + ["..."] + shown[-5:]
            print(f"  {r.run_name:<{widths[0]}}  " + " -> ".join(shown))
        print(rule)

    # Legend, so the shortened headers and the less obvious metrics explain themselves.
    glossed = [m for m in metrics if m in _METRIC_GLOSS]
    if glossed:
        print("columns:")
        for metric in glossed:
            print(f"  {_METRIC_HEADERS.get(metric, metric):<10} {_METRIC_GLOSS[metric]}")
        print(f"  {'time':<10} wall-clock duration of the run")
        print("metrics are from the final round; per-round history is in the JSON report.")
        print(rule)

    summary = f"{len(results)} run{'s' if len(results) != 1 else ''}"
    if failed:
        summary += f", {len(results) - len(failed)} succeeded, {len(failed)} failed"
    if total_duration:
        summary += f", {total_duration:.1f}s total"
    print(summary)
    for r in failed:
        # Some libraries raise a message that opens with a blank line, which leaves the
        # first line holding only the exception name. Carry the next line with content up
        # so the reason is readable rather than a bare "ImportError:".
        lines = [line.strip() for line in (r.error or "").splitlines() if line.strip()]
        reason = lines[0] if lines else "no error recorded"
        if reason.endswith(":") and len(lines) > 1:
            reason = f"{reason} {lines[1]}"
        print(f"  {r.run_name} [{r.framework}] failed: {reason}")
    if report_path:
        print(f"Report: {report_path}")
