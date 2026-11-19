# Test fixtures

`data/` holds tiny samples of the real sources, laid out like a `KDP_DATA_ROOT`, so `tests/test_builds.py` can run every
build script unchanged. Regenerate them from full downloads with:

```bash
python -I tests/fixtures/make_fixtures.py <datasets root>
```

Sources and licenses of the samples:
- EIA-930 (CISO, ERCO, PJM; 7 days of 2018, 21 days of 2025): U.S. Energy Information Administration, public domain.
- World Development Indicators (6 economies + World, 42 indicators): World Bank, CC BY 4.0.
- San Francisco Street Tree Inventory (300 trees): San Francisco Public Works via DataSF, ODC-PDDL.
