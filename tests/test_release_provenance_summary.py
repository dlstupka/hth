import tempfile
import unittest
from pathlib import Path

from hth.release_provenance_summary import input_release_lines, main, summary_lines


class ReleaseProvenanceSummaryTests(unittest.TestCase):
    def test_release_identity_and_all_provenance_are_linked(self) -> None:
        rendered = "\n".join(
            summary_lines(
                title="Durable chromatic result",
                result_label="Chromatic result",
                result_identity="result-id",
                repository="owner/results",
                release_tag="HTH-CHROMATIC-result-id",
                release_activity="CREATED",
                results_commit="abc123",
            )
        )
        release_url = "https://github.com/owner/results/releases/tag/HTH-CHROMATIC-result-id"
        self.assertIn(f"- Chromatic result: [`result-id`]({release_url})", rendered)
        self.assertIn(f"- Release: [`HTH-CHROMATIC-result-id`]({release_url})", rendered)
        self.assertIn("- Release activity: `CREATED`", rendered)
        self.assertIn("- Results commit: [`abc123`](https://github.com/owner/results/commit/abc123)", rendered)

    def test_optional_upstream_identity_is_linked(self) -> None:
        rendered = "\n".join(
            summary_lines(
                title="Durable photometric result",
                result_label="Photometric result",
                result_identity="photo-id",
                repository="owner/results",
                release_tag="HTH-PHOTOMETRIC-photo-id",
                release_activity="REUSED",
                upstream_label="Base normalization result",
                upstream_identity="base-id",
                upstream_url="https://example.invalid/manifest.json",
            )
        )
        self.assertIn(
            "- Base normalization result: [`base-id`](https://example.invalid/manifest.json)",
            rendered,
        )

    def test_partial_upstream_provenance_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires label, identity, and URL"):
            summary_lines(
                title="Durable result",
                result_label="Result",
                result_identity="result-id",
                repository="owner/results",
                release_tag="release-id",
                release_activity="CREATED",
                upstream_label="Input",
            )

    def test_cli_appends_to_github_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "summary.md"
            result = main(
                [
                    "--title", "Reusable tonal result",
                    "--result-label", "Tonal result",
                    "--result-identity", "tonal-id",
                    "--repository", "owner/results",
                    "--release-tag", "HTH-TONAL-tonal-id",
                    "--release-activity", "AUDITED",
                    "--github-summary", str(output),
                ]
            )
            self.assertEqual(result, 0)
            text = output.read_text(encoding="utf-8")
            self.assertIn("## Reusable tonal result", text)
            self.assertIn("- Release: [`HTH-TONAL-tonal-id`]", text)

    def test_cached_input_identifies_authoritative_release_and_resolution(self) -> None:
        rendered = "\n".join(
            input_release_lines(
                title="Immutable release input",
                repository="owner/results",
                release_tag="HTH-TONAL-result-id",
                asset="hth-tonal-result-id.zip",
                asset_sha256="a" * 64,
                cache_source="runner-local-cache",
            )
        )
        release_url = "https://github.com/owner/results/releases/tag/HTH-TONAL-result-id"
        self.assertIn(f"- Release: [`HTH-TONAL-result-id`]({release_url})", rendered)
        self.assertIn("- Repository: [`owner/results`](https://github.com/owner/results)", rendered)
        self.assertIn("- Asset: `hth-tonal-result-id.zip`", rendered)
        self.assertIn(f"- Asset SHA-256: `{'a' * 64}`", rendered)
        self.assertIn("- Resolution: `runner-local-cache`", rendered)


if __name__ == "__main__":
    unittest.main()
