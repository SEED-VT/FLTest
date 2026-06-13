"""FLTest command-line interface.

Subcommands:
  fltest run <conf>          run the orchestrated experiment matrix
  fltest diff <conf>         differential test (cross-framework parity or determinism)
  fltest metamorphic <conf>  metamorphic relations (clients/rounds/attack/dp)
  fltest pitfalls <conf>     pitfall check + counter-experiment recommendations
  fltest list                list available frameworks / attacks / defenses / metrics
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

# Quieter third-party logs by default (Flower/Ray/HF are noisy).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("RAY_DEDUP_LOGS", "0")


def _quiet(verbose: bool) -> None:
    if not verbose:
        logging.disable(logging.INFO)


@click.group()
@click.version_option(version="0.2.0", prog_name="fltest")
def cli() -> None:
    """FLTest — a testbed for privacy & robustness of Privacy-Preserving FL."""


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="reports", help="Directory for the JSON report.")
@click.option("--verbose", "-v", is_flag=True, help="Show framework/Ray logs.")
def run(config_path: str, output: str, verbose: bool) -> None:
    """Run the orchestrated experiment matrix from CONFIG_PATH."""
    _quiet(verbose)
    from fltest.core.config import load_config
    from fltest.core.orchestrator import Orchestrator
    from fltest.testing.report import write_report

    config = load_config(config_path)
    matrix = Orchestrator(verbose=True).run(config)

    click.echo("\n" + "=" * 70 + f"\nRUN MATRIX: {config.name}\n" + "=" * 70)
    for r in matrix.results:
        acc = r.final.get("accuracy")
        acc_s = f"acc={acc:.4f}" if acc is not None else r.status
        extra = " ".join(
            f"{k}={v:.4f}" for k, v in r.final.items()
            if k not in ("accuracy", "loss", "gm_weight_sum") and isinstance(v, (int, float))
        )
        click.echo(f"  {r.run_name:<14} [{r.framework:<9}] {acc_s} {extra}")
    click.echo("=" * 70)

    report_path = write_report(
        Path(output) / f"{config.name}_run.json", title=f"FLTest run: {config.name}",
        runs=[r.to_dict() for r in matrix.results],
        extra={"total_duration": round(matrix.total_duration, 2)},
    )
    click.echo(f"Report: {report_path}")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="reports", help="Directory for the JSON report.")
@click.option("--verbose", "-v", is_flag=True, help="Show framework/Ray logs.")
def diff(config_path: str, output: str, verbose: bool) -> None:
    """Differential test: cross-framework parity (default) or determinism."""
    _quiet(verbose)
    from fltest.core.config import load_config
    from fltest.core.orchestrator import Orchestrator
    from fltest.testing.differential import DifferentialTester
    from fltest.testing.report import print_outcomes, write_report

    config = load_config(config_path)
    spec = config.testing.differential
    mode = spec.mode if spec else "cross_framework"
    metric = spec.metric if spec else "accuracy"
    tol = spec.tolerance if spec else 0.05

    tester = DifferentialTester(Orchestrator(verbose=True))
    if mode == "determinism":
        report = tester.determinism(config, metric=metric, eps=tol)
    else:
        matrix = Orchestrator(verbose=True).run(config)
        report = tester.cross_framework(matrix, metric=metric, tolerance=tol)

    ok = print_outcomes(f"DIFFERENTIAL ({mode}, metric={metric}, tol={tol})", report.outcomes)
    write_report(Path(output) / f"{config.name}_differential.json",
                 title=f"FLTest differential: {config.name}", outcomes=report.outcomes,
                 extra={"mode": mode, "metric": metric, "tolerance": tol})
    sys.exit(0 if ok else 1)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="reports", help="Directory for the JSON report.")
@click.option("--verbose", "-v", is_flag=True, help="Show framework/Ray logs.")
def metamorphic(config_path: str, output: str, verbose: bool) -> None:
    """Metamorphic test: check input/output relations defined under testing.metamorphic."""
    _quiet(verbose)
    from fltest.core.config import load_config
    from fltest.core.orchestrator import Orchestrator
    from fltest.testing.metamorphic import MetamorphicTester
    from fltest.testing.report import print_outcomes, write_report

    config = load_config(config_path)
    relations = config.testing.metamorphic
    if not relations:
        click.echo("No metamorphic relations defined under testing.metamorphic.", err=True)
        sys.exit(2)

    report = MetamorphicTester(Orchestrator(verbose=True)).check(config, relations)
    ok = print_outcomes("METAMORPHIC", report.outcomes)
    write_report(Path(output) / f"{config.name}_metamorphic.json",
                 title=f"FLTest metamorphic: {config.name}", outcomes=report.outcomes)
    sys.exit(0 if ok else 1)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def pitfalls(config_path: str) -> None:
    """Run the pitfall checker and print counter-experiment recommendations."""
    from fltest.core.config import load_config
    from fltest.pitfalls import check_config, recommend

    config = load_config(config_path)
    findings = check_config(config)

    click.echo("\n" + "=" * 70 + f"\nPITFALL CHECK: {config.name}\n" + "=" * 70)
    if not findings:
        click.echo("  No pitfalls detected. (Still consider sweeping attacks/distributions.)")
    for f in findings:
        click.echo(f"\n  [{f.severity.upper():<6}] {f.title}  ({f.pitfall})")
        click.echo(f"      {f.message}")
        click.echo(f"      → {f.recommendation}")

    recs = recommend(findings)
    if recs:
        click.echo("\n" + "-" * 70 + "\nCOUNTER-EXPERIMENTS (merge into your config):\n" + "-" * 70)
        for r in recs:
            if r["counter_experiment"]:
                click.echo(f"\n# {r['title']} ({r['severity']})")
                click.echo(r["counter_experiment"])
    click.echo("=" * 70)


@cli.command(name="list")
def list_components() -> None:
    """List available frameworks, attacks, defenses, and metrics."""
    import fltest.frameworks  # noqa: F401
    import fltest.attacks  # noqa: F401
    import fltest.defenses  # noqa: F401
    import fltest.metrics  # noqa: F401
    from fltest.core.registry import ATTACKS, DEFENSES, FRAMEWORKS, METRICS

    click.echo(f"Frameworks: {FRAMEWORKS.names()}")
    click.echo(f"Attacks:    {ATTACKS.names()}")
    click.echo(f"Defenses:   {DEFENSES.names()}")
    click.echo(f"Metrics:    {METRICS.names()}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
