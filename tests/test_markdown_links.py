import unittest

from hth.markdown_links import (
    code_link,
    github_blob_url,
    github_commit_url,
    github_release_url,
    github_repository_url,
    github_tree_url,
)


class MarkdownLinkTests(unittest.TestCase):
    def test_preserves_code_styling_inside_link(self):
        self.assertEqual(code_link("abc", "https://example.invalid"), "[`abc`](https://example.invalid)")

    def test_repository_url_accepts_slug_https_and_ssh_forms(self):
        expected = "https://github.com/dlstupka/hth"
        self.assertEqual(github_repository_url("dlstupka/hth"), expected)
        self.assertEqual(github_repository_url("https://github.com/dlstupka/hth.git"), expected)
        self.assertEqual(github_repository_url("git@github.com:dlstupka/hth.git"), expected)

    def test_provenance_urls_encode_refs_and_paths(self):
        self.assertEqual(
            github_release_url("owner/repo", "release one"),
            "https://github.com/owner/repo/releases/tag/release%20one",
        )
        self.assertEqual(
            github_commit_url("owner/repo", "abc123"),
            "https://github.com/owner/repo/commit/abc123",
        )
        self.assertEqual(
            github_blob_url("owner/repo", "release one", "path/file.json"),
            "https://github.com/owner/repo/blob/release%20one/path/file.json",
        )
        self.assertEqual(
            github_tree_url("owner/repo", "abc123", "records/run 1"),
            "https://github.com/owner/repo/tree/abc123/records/run%201",
        )


if __name__ == "__main__":
    unittest.main()
