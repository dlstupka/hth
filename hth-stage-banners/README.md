# HTH stage banners

This package is based on the latest publication-summary plumbing update and adds
visible GitHub Actions log banners for the stage names already defined in
`_core-hth.yml`.

Added banners:

- `STAGE_PREPROCESS`
- `STAGE_DETECT_CURRENT`
- `STAGE_DETECT_CANDIDATES`
- `STAGE_VALIDATE_GEOMETRY`
- `STAGE_VALIDATE_OUTPUTS`
- `STAGE_PUBLISH_PRODUCTION`
- `STAGE_PUBLISH_TEST`

Conditional banners use the same mode/publication conditions as their stages,
so irrelevant stages appear as skipped rather than printing misleading output.

Install from the HTH repository root:

```bash
unzip hth-stage-banners.zip
cp -a hth-stage-banners/. .
python -m unittest tests.test_write_run_summary -v
git diff --check
```
