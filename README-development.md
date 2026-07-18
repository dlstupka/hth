# HTH Development Guide

## Requirements

- Python 3.12+
- Git
- ZIP/UnZIP
- dependencies from `requirements.txt`

## Validate a change

```bash
python -m unittest discover -s tests -v
python -m compileall -q hth
git diff --check
git status
```

## Apply a packaged update

```bash
unzip hth-update.zip
cp -a hth-update/. .
python -m unittest discover -s tests -v
git diff --check
git diff
```

The overlay pattern deliberately preserves unrelated repository files while
replacing only files carried by the package.

## Commit discipline

```bash
git add <reviewed-files>
git status
git commit -m "Describe the architectural or behavioral change"
git push
```

Prefer explicit `git add` followed by `git commit -m`. Avoid relying on
`git commit -a` for new files.

## Workflow design rules

- Keep entry workflows thin.
- Put shared behavior in `.github/workflows/_core-hth.yml`.
- Use canonical `STAGE_*` names in workflow logs and documentation.
- Add timestamps and elapsed timing through `hth/stage_timing.py`.
- Put summary formatting and JSON parsing in Python, not shell/YAML.
- Add or update tests and documentation with every durable interface change.
