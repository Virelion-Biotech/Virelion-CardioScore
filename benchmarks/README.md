# Reference benchmark harness

CardioScore benchmarks are deliberately reference-data driven. Do not invent expected scores from the implementation being tested.

A benchmark manifest is YAML or JSON and contains one or more dataset entries:

```yaml
datasets:
  - name: example_reference
    features: path/to/features.csv
    config: path/to/pipeline.yaml
    score_tolerance: 0.02
    expected:
      - compound: Compound_A
        cardioscore: 0.42
        risk_class: Moderate
```

Run:

```bash
cardioscore benchmark --manifest benchmarks/reference_manifest.yaml
```

The command exits non-zero when any observed score exceeds its tolerance or its risk class differs from the locked reference.

Reference files should come from an independently reviewed analysis, published benchmark, or frozen internal adjudication. The repository does not ship fabricated real-data reference scores.
