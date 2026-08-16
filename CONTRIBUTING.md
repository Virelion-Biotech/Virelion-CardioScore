# Contributing to Virelion CardioScore

Thank you for your interest in improving early cardiac safety assessment tools.

## Ground rules

- Keep the project GPL-3.0-or-later compatible.
- Prefer transparent, interpretable methods over black-box models for the core scoring path.
- Document scientific assumptions clearly (especially endpoint weights and risk thresholds).
- Do not claim regulatory validation unless a formal study supports it.
- Write tests for new functionality.

## Development setup

```bash
git clone https://github.com/Virelion-Biotech/Virelion-CardioScore.git
cd Virelion-CardioScore
python -m venv .venv
source .venv/bin/activate   # or Windows equivalent
pip install -e ".[dev]"
pytest
```

## Suggested contribution areas

1. **MEA format readers** – Axion, Multi Channel Systems, or other common exporters.
2. **Additional endpoints** – conduction velocity proxies, early-afterdepolarization detectors, etc.
3. **Concentration–response modeling** – optional curve fitting when data support it.
4. **Validation datasets** – public or de-identified MEA series that can serve as benchmarks.
5. **Documentation & tutorials** – notebooks that walk through real analysis workflows.
6. **Report improvements** – richer plots, PDF export, LIMS-friendly exports.

## Pull request process

1. Open an issue first for larger changes so we can discuss design.
2. Keep PRs focused; one logical change per PR is ideal.
3. Ensure `pytest` passes and that the demo CLI still runs.
4. Update the CHANGELOG and relevant docs.
5. Reference any related issues.

## Code style

- Python 3.10+ type hints where practical.
- `ruff` for linting (config in `pyproject.toml`).
- Prefer small, well-named functions over deep nesting.

## Scientific responsibility

This framework is used to help prioritize compounds for further evaluation.  
Misleading risk classifications can waste resources or, worse, miss real liabilities.  
Please treat endpoint definitions, weights, and thresholds with appropriate care and document any changes thoroughly.

## Questions

Open a GitHub issue or contact the maintainers via the organization page.
