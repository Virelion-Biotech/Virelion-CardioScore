"""
Command-line interface for Virelion CardioScore.
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd

from virelion_cardioscore import __version__
from virelion_cardioscore.analysis.pipeline import CardioScorePipeline
from virelion_cardioscore.io.raw_trace import RawTraceSchemaError, load_raw_traces_to_feature_table
from virelion_cardioscore.io.synthetic import load_synthetic_dataset
from virelion_cardioscore.preprocessing.beat_detection import BeatDetectionConfig
from virelion_cardioscore.preprocessing.filtering import FilterConfig


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
    dataset = load_synthetic_dataset(n_compounds=n_compounds, n_concentrations=n_concentrations, seed=seed)
    click.echo(f"Compounds: {dataset.compounds}")
    click.echo(f"Wells: {len(dataset.features)}")

    result = CardioScorePipeline.from_defaults().run(dataset)
    result.to_csv(out / "summary.csv")
    result.to_json(out / "scores.json")
    result.to_html(out / "report.html")
    click.echo(result.summary_table.to_string(index=False))
    click.echo(f"Outputs written to {out.resolve()}/")


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Pipeline YAML config")
@click.option("--output-dir", default="./outputs", show_default=True, type=click.Path())
@click.option("--features", type=click.Path(exists=True), help="Optional pre-extracted feature CSV")
@click.option("--raw-traces", type=click.Path(exists=True), help="Raw per-electrode voltage trace CSV")
def run(config: str, output_dir: str, features: str | None, raw_traces: str | None) -> None:
    """Run CardioScore on a configured experiment."""
    if features and raw_traces:
        raise click.UsageError("Pass only one of --features or --raw-traces, not both.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipeline = CardioScorePipeline.from_config(config)

    if raw_traces:
        preprocessing_cfg = pipeline.config.get("preprocessing", {})
        beat_cfg = pipeline.config.get("beat_detection", {})
        filter_config = FilterConfig(
            highpass_hz=float(preprocessing_cfg.get("highpass_hz", 0.5)),
            lowpass_hz=float(preprocessing_cfg.get("lowpass_hz", 40.0)),
            notch_hz=float(preprocessing_cfg.get("notch_hz", 50.0)),
            notch_q=float(preprocessing_cfg.get("notch_q", 30.0)),
            detrend=bool(preprocessing_cfg.get("detrend", True)),
        )
        beat_config = BeatDetectionConfig.from_dict(beat_cfg)
        try:
            df = load_raw_traces_to_feature_table(raw_traces, filter_config=filter_config, beat_config=beat_config)
        except RawTraceSchemaError as exc:
            raise click.ClickException(str(exc)) from exc
        result = pipeline.run(df)
    elif features:
        result = pipeline.run(pd.read_csv(features))
    else:
        result = pipeline.run(load_synthetic_dataset())

    reporting = pipeline.config.get("reporting", {})
    generated = []
    if reporting.get("generate_csv", True):
        result.to_csv(out / "summary.csv")
        generated.append("summary.csv")
    if reporting.get("generate_json", True):
        result.to_json(out / "scores.json")
        generated.append("scores.json")
    if reporting.get("generate_html", True):
        result.to_html(out / "report.html")
        generated.append("report.html")

    click.echo(result.summary_table.to_string(index=False))
    click.echo(f"Results written to {out.resolve()}")
    click.echo(f"Generated: {', '.join(generated) if generated else 'no report files (disabled by config)'}")


@main.command()
def version() -> None:
    """Print version and exit."""
    click.echo(f"virelion-cardioscore {__version__}")


if __name__ == "__main__":
    main()
