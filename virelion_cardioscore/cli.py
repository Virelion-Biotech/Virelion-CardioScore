"""
Command-line interface for Virelion CardioScore.

Usage examples
--------------
  cardioscore demo
  cardioscore run --config path/to/config.yaml --output-dir results/
  cardioscore version
"""

from __future__ import annotations

from pathlib import Path

import click

from virelion_cardioscore import __version__
from virelion_cardioscore.analysis.pipeline import CardioScorePipeline
from virelion_cardioscore.io.synthetic import load_synthetic_dataset


@click.group()
@click.version_option(__version__, prog_name="cardioscore")
def main() -> None:
    """Virelion CardioScore – CiPA-aligned cardiotoxicity scoring for iPSC-CM MEA data."""
    pass


@main.command()
@click.option("--n-compounds", default=4, show_default=True, help="Number of synthetic compounds")
@click.option("--n-concentrations", default=6, show_default=True, help="Concentration points")
@click.option("--seed", default=42, show_default=True, help="Random seed")
@click.option("--output-dir", default="./outputs", show_default=True, type=click.Path())
def demo(n_compounds: int, n_concentrations: int, seed: int, output_dir: str) -> None:
    """Generate synthetic data and run the full CardioScore pipeline."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    click.echo("Generating synthetic MEA dataset …")
    dataset = load_synthetic_dataset(
        n_compounds=n_compounds,
        n_concentrations=n_concentrations,
        seed=seed,
    )
    click.echo(f"  Compounds : {dataset.compounds}")
    click.echo(f"  Wells     : {len(dataset.features)}")

    click.echo("Running CardioScore pipeline …")
    pipeline = CardioScorePipeline.from_defaults()
    result = pipeline.run(dataset)

    summary_path = out / "summary.csv"
    json_path = out / "scores.json"
    html_path = out / "report.html"

    result.to_csv(summary_path)
    result.to_json(json_path)
    result.to_html(html_path)

    click.echo("\n=== CardioScore Summary ===")
    click.echo(result.summary_table.to_string(index=False))
    click.echo(f"\nQC log ({len(result.qc_log)} entries):")
    for line in result.qc_log[:8]:
        click.echo(f"  • {line}")
    if len(result.qc_log) > 8:
        click.echo(f"  … and {len(result.qc_log) - 8} more")

    click.echo(f"\nOutputs written to {out.resolve()}/")
    click.echo(f"  • {summary_path.name}")
    click.echo(f"  • {json_path.name}")
    click.echo(f"  • {html_path.name}")


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Pipeline YAML config")
@click.option("--output-dir", default="./outputs", show_default=True, type=click.Path())
@click.option("--features", type=click.Path(exists=True), help="Optional pre-extracted feature CSV")
def run(config: str, output_dir: str, features: str | None) -> None:
    """Run CardioScore on a configured experiment (or feature table)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pipeline = CardioScorePipeline.from_config(config)

    if features:
        import pandas as pd

        df = pd.read_csv(features)
        result = pipeline.run(df)
    else:
        click.echo("No --features supplied; running synthetic demo data with the given config.")
        dataset = load_synthetic_dataset()
        result = pipeline.run(dataset)

    result.to_csv(out / "summary.csv")
    result.to_json(out / "scores.json")
    result.to_html(out / "report.html")

    click.echo(result.summary_table.to_string(index=False))
    click.echo(f"\nResults written to {out.resolve()}")


@main.command()
def version() -> None:
    """Print version and exit."""
    click.echo(f"virelion-cardioscore {__version__}")


if __name__ == "__main__":
    main()
