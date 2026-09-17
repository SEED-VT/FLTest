"""The run-matrix table shows what differs between runs and flags failures."""

import types

from fltest.testing.report import print_run_matrix


def _result(name, framework="reference", status="success", error=None, **final):
    """A stand-in for RunResult carrying only the fields the renderer reads."""
    return types.SimpleNamespace(
        run_id=name, run_name=name, framework=framework, status=status, error=error,
        duration_seconds=1.0,
        params={
            "framework": framework, "dataset": "mnist", "data_distribution": "iid",
            "dirichlet_alpha": None, "model_name": "MLP", "num_clients": 4,
            "num_rounds": 3, "seed": 786, "attacks": [], "defenses": [],
        },
        final=final, history={},
    )


def test_shared_parameters_print_once_and_differences_become_columns(capsys):
    a = _result("plain", accuracy=0.9, loss=0.3)
    b = _result("robust", accuracy=0.8, loss=0.4)
    b.params["defenses"] = [{"name": "median", "params": {}}]

    print_run_matrix("demo", [a, b], total_duration=2.0)
    out = capsys.readouterr().out

    assert "same for all runs:" in out
    assert "dataset=mnist" in out          # shared, so stated once above the table
    assert "defense" in out                # differs, so it is a column
    assert "median" in out
    assert "dirichlet_alpha" not in out    # not applicable under IID, so not shown
    assert "2 runs" in out


def test_plugin_metrics_appear_with_short_headers(capsys):
    r = _result("backdoor", accuracy=0.89, loss=0.35, attack_success_rate=0.47,
                per_client_acc_min=0.36, gm_weight_sum=-51.6)
    print_run_matrix("demo", [r])
    out = capsys.readouterr().out

    assert "asr" in out and "0.4700" in out
    assert "pc-min" in out
    assert "gm_weight_sum" not in out      # fingerprint stays in the JSON report only


def test_failed_run_is_reported_with_status_and_reason(capsys):
    ok = _result("good", accuracy=0.9, loss=0.3)
    bad = _result("bad", status="failed", error="RuntimeError: backend exploded\n  stack…")

    print_run_matrix("demo", [ok, bad], total_duration=2.0)
    out = capsys.readouterr().out

    assert "status" in out and "failed" in out
    assert "1 succeeded, 1 failed" in out
    assert "RuntimeError: backend exploded" in out


def test_empty_matrix_does_not_crash(capsys):
    print_run_matrix("demo", [])
    assert "no runs" in capsys.readouterr().out


def test_aggregation_rule_is_named_in_the_output(capsys):
    """A robust-aggregation defense replaces FedAvg, so the table has to say which ran."""
    plain = _result("plain", accuracy=0.9)
    robust = _result("robust", accuracy=0.9)
    plain.params["aggregation"] = "fedavg"
    robust.params["aggregation"] = "median"

    print_run_matrix("demo", [plain, robust])
    out = capsys.readouterr().out
    assert "aggregation" in out and "fedavg" in out and "median" in out


def test_shortened_columns_are_explained(capsys):
    r = _result(
        "r", accuracy=0.9, attack_success_rate=0.4,
        membership_inference_auc=0.66, model_replacement_scale=4.0,
    )
    print_run_matrix("demo", [r])
    out = capsys.readouterr().out

    assert "columns:" in out
    assert "attack success rate" in out          # asr
    assert "membership inference AUC" in out     # mia-auc
    assert "0.5 is no leakage" in out
    assert "mr-scale" in out
    assert "scale applied to the malicious client delta" in out


def test_per_round_trace_shows_progress(capsys):
    r = _result("r", accuracy=0.9)
    r.history = {1: {"accuracy": 0.5}, 2: {"accuracy": 0.7}, 3: {"accuracy": 0.9}}
    print_run_matrix("demo", [r])
    out = capsys.readouterr().out

    assert "per-round accuracy:" in out
    assert "0.5000 -> 0.7000 -> 0.9000" in out


def test_single_round_run_has_no_trace(capsys):
    r = _result("r", accuracy=0.9)
    r.history = {1: {"accuracy": 0.9}}
    print_run_matrix("demo", [r])
    assert "per-round accuracy:" not in capsys.readouterr().out


def test_failure_reason_skips_a_leading_blank_line(capsys):
    """Some libraries raise messages that begin with a newline.

    Reporting the first line verbatim then prints the exception name and nothing else,
    which is how an unusable `ImportError:` reached a user.
    """
    bad = _result("bad", status="failed",
                  error="ImportError: \n\nAutoModel requires the PyTorch library\nmore detail")
    print_run_matrix("demo", [bad])
    out = capsys.readouterr().out
    assert "AutoModel requires the PyTorch library" in out
    assert "failed: ImportError:\n" not in out


def test_non_numeric_metrics_do_not_become_columns(capsys):
    """A plugin may record a list or a dict, which cannot be a fixed-width cell.

    FLDetector records per-client scores and the ids it flagged. Formatting either as a
    float raises TypeError, which would crash the CLI after a run had already finished.
    """
    r = _result("fld", accuracy=0.9)
    r.final.update(
        fldetector_detected_count=2,
        fldetector_detected_clients=[0, 1],
        fldetector_scores={0: 0.1, 1: 0.9},
    )
    print_run_matrix("demo", [r])          # must not raise
    out = capsys.readouterr().out

    assert "fld-flagged" in out            # the numeric one is a column
    assert "fldetector_scores" not in out  # the structured ones stay in the JSON report
    assert "fldetector_detected_clients" not in out
