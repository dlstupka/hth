# Hidden Texas History Research Framework (HTH) v0.6.1

HTH is an open, reproducible digital-humanities framework for acquiring,
preprocessing, analyzing, validating, and publishing historical document
collections. The first collection is **HTH-0001: San Antonio Baptisms,
1788–1824 and 1858–1898**.

The framework keeps source evidence, processing code, generated build artifacts,
and curated publications separate so that every published result can be traced
back to an exact source commit and pipeline commit.

## Current pipeline

```text
Source repository
        ↓
STAGE_PREPROCESS
        ↓
STAGE_DETECT_CURRENT
        ↓
STAGE_DETECT_CANDIDATES
        ↓
STAGE_VALIDATE_GEOMETRY
        ↓
STAGE_VALIDATE_OUTPUTS
        ↓
STAGE_PUBLISH_PRODUCTION or STAGE_PUBLISH_TEST
        ↓
Publication Manifest + Pipeline Health
```

GitHub Actions logs use these exact stage names. Each stage records its UTC
start time, UTC completion time, status, and elapsed duration. Pipeline Health
summarizes those timings so runtime changes can be spotted without reading the
full log.

## Version 0.6

HTH framework v0.6.1 extends the maintainable pipeline foundation:

- reusable GitHub Actions core with thin production, test, and calibration wrappers;
- detector registry with independent detector failure handling;
- physical-page analysis, candidate generation, and geometry validation;
- production and test publication layouts;
- Publication Manifest for provenance and publication identity;
- Pipeline Health for counts, output validation, timestamps, and stage timing;
- stable stage banners using the canonical `STAGE_*` names;
- unit tests for summary plumbing and detector registration.

The AutoHotkey capture utility retains its own **v0.5** version identity. The
framework release and acquisition-tool release are intentionally tracked
separately.

## Repository map

```text
.github/workflows/       reusable core and entry workflows
config/                  versioned pipeline configuration
hth/                     Python pipeline modules
hth/geometry/            detector registry and detector implementations
tests/                   unit tests
tools/                   reference-collection review tools
docs/                   design, operating, and project documentation
docs/README.md            documentation index
CHANGELOG.md              framework release history
```

## Run tests

From the repository root:

```bash
python -m unittest discover -s tests -v
git diff --check
```

## Apply an HTH update package

```bash
unzip hth-update.zip
cp -a hth-update/. .
python -m unittest discover -s tests -v
git diff --check
```

Always review `git diff` before committing.

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md)
- [Project status](docs/project-status.md)
- [Detector regression](docs/regression.md)
- [Development and update workflow](docs/development.md)

## Acknowledgments

HTH was conceived and developed by **Dan Stupka**. The framework architecture,
detector implementations, workflow design, and engineering were developed
collaboratively with **OpenAI ChatGPT**. Detector-level metadata separately
records each implementation's origin, algorithmic foundation, authorship,
version, and source repository.

The detector registry is the single source of truth for those fields. Detector
implementations return algorithm results and diagnostics only; the registry
injects provenance, timing, validation, and exception isolation uniformly.
Detailed field definitions and version policy are documented in
[multi-detector geometry documentation](docs/multidetector-geometry.md).

## Contact

Project maintainer: **Dan Stupka**  
Email: **stupka@gmail.com**

Historical corrections, source leads, reproducibility reports, and technical
contributions are welcome.
