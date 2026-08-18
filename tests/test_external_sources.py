from pathlib import Path

import yaml


def test_external_source_registry_has_required_provenance_fields():
    registry = Path(__file__).parents[1] / "benchmarks" / "external_sources.yaml"
    payload = yaml.safe_load(registry.read_text(encoding="utf-8"))
    sources = payload["sources"]
    assert len(sources) >= 3
    required = {"id", "status", "evidence_level", "title", "year", "doi", "url", "benchmark_scope"}
    for source in sources:
        assert required.issubset(source)
        assert source["doi"]
        assert source["url"].startswith("https://")
        assert source["evidence_level"] in {
            "raw_mea_dataset",
            "processed_mea_summary",
            "published_mea_summary",
        }
        assert source["benchmark_scope"]


def test_external_registry_does_not_mark_published_summary_as_raw_data():
    registry = Path(__file__).parents[1] / "benchmarks" / "external_sources.yaml"
    sources = yaml.safe_load(registry.read_text(encoding="utf-8"))["sources"]
    for source in sources:
        if source["evidence_level"] == "published_mea_summary":
            assert source["status"] == "published_summary_only"
