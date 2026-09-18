# Contrast and tonal-range normalization

The tonal stage follows the immutable photometric collection and treats every
traditional normalization question as an evidence problem. It never assumes a
contrast transform is useful merely because one is available, and it never
drops a page when correction is unsafe or unnecessary.

**HTH normalize collection** is the top-level orchestrator. One dispatch runs
canonical normalization, the complete photometric sequence, and the following
tonal sequence in dependency order. Individual stage workflows remain available
for audit and recovery:

1. **HTH normalize tonal assess** measures percentile endpoints, usable tonal
   span, median luminance, and endpoint clipping on every page. Each page is
   classified as `correction-candidate`, `preserve`, or `review`; all three
   routes continue downstream.
2. **HTH normalize tonal method assessment** compares bounded global
   percentile-stretch strengths on a deterministic development partition.
   A method must improve tonal span while preserving high-frequency detail,
   endpoint occupancy, and median luminance.
3. **HTH normalize tonal method validation** applies the selected method to
   every held-out candidate. One unsafe candidate blocks automatic correction.
4. **HTH normalize tonal integration** reproduces every evidenced route,
   writes a lossless collection, publishes an immutable release, and records
   Canonical Build Evidence (CBE).

An assessment that finds no useful correction is still a valid scientific
result. Integration then emits an identity-preserving collection and CBE proves
why no tonal operation was applied. Review classifications are annotations,
not pipeline failures.

## Cache and provenance hierarchy

The photometric input release is addressed by its SHA-256. Resolution checks a
verified runner-local cache first (`/tmp/.ar/.hth-release-cache` on self-hosted
runners, or `HTH_RELEASE_LOCAL_CACHE_ROOT` when configured). GitHub-hosted jobs
also use the portable Actions cache. A miss downloads the authoritative Results
repository release, verifies it before cache admission, and verifies it again
before use. The Results release remains authoritative; caches are performance
layers and cannot change identity or trust.

CBE fingerprints the upstream release asset, all compact assessment evidence,
the three configuration files, implementation, and runtime contract. An exact
rerun audits and reuses the immutable tonal result; `force-verify` reconstructs
it and proves equivalence; `rebuild` is reserved for changed effective inputs.

The stage applies no sharpening, denoising, binarization, geometric transform,
or HTR-specific enhancement.
