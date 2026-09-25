"""Offline checks for the director's public-release trust boundary."""

import importlib.util
import hashlib
import io
import json
import subprocess
import tempfile
import threading
import unittest
import urllib.parse
import urllib.request
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reference_collection_director", ROOT / "tools/reference-collection-director.py"
)
DIRECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIRECTOR)
REPOSITORY = "dlstupka/hth-baptisms-san-antonio-1788-1824--1858-1898"
TAG = "HTH-GOLDEN-0002"


class ReferenceCollectionDirectorTests(unittest.TestCase):
    def setUp(self):
        self.cache = tempfile.TemporaryDirectory()
        self.addCleanup(self.cache.cleanup)
        cache_patch = patch.object(DIRECTOR, "CACHE_ROOT", Path(self.cache.name))
        cache_patch.start()
        self.addCleanup(cache_patch.stop)
        self.freeze = json.loads((ROOT / f"config/golden_sets/{TAG}.freeze.json").read_text(encoding="utf-8"))
        self.golden_bytes = (ROOT / f"config/golden_sets/{TAG}.golden-set.json").read_bytes()
        self.release_url = f"https://api.github.com/repos/{REPOSITORY}/releases/tags/{TAG}"
        self.assets = [
            {"name": f"{TAG}.{name}", "size": size, "digest": digest,
             "browser_download_url": f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{TAG}.{name}"}
            for name, size, digest in (
                ("freeze.json", 0, ""),
                ("golden-set.json", 0, ""),
                ("images.zip", self.freeze["image_bundle"]["size"],
                 "sha256:" + self.freeze["image_bundle"]["sha256"]),
            )
        ]

    def resolve(self):
        values = {
            self.release_url: json.dumps({"tag_name": TAG, "assets": self.assets}).encode(),
            self.assets[0]["browser_download_url"]: json.dumps(self.freeze).encode(),
            self.assets[1]["browser_download_url"]: self.golden_bytes,
        }
        with patch.object(DIRECTOR, "_read_limited", side_effect=values.__getitem__):
            return DIRECTOR.resolve_release(REPOSITORY, TAG)

    def test_rejects_non_github_source_url(self):
        for url in ("https://example.com/owner/repo", "https://github.com/owner/repo/tree/main", "file:///tmp/repo"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                DIRECTOR.source_repository(url)
        self.assertEqual(DIRECTOR.source_repository("https://github.com/owner/repo"), "owner/repo")

    def test_resolves_matching_frozen_release(self):
        result = self.resolve()
        self.assertEqual(result["tag"], TAG)
        self.assertEqual(result["golden_set_sha256"], self.freeze["golden_set_sha256"])
        self.assertEqual(result["bundle"]["size"], 37619784)

    def test_reuses_digest_checked_golden_set_json(self):
        values = {
            self.release_url: json.dumps({"tag_name": TAG, "assets": self.assets}).encode(),
            self.assets[0]["browser_download_url"]: json.dumps(self.freeze).encode(),
            self.assets[1]["browser_download_url"]: self.golden_bytes,
        }
        with patch.object(DIRECTOR, "_read_limited", side_effect=values.__getitem__) as read:
            DIRECTOR.resolve_release(REPOSITORY, TAG)
            DIRECTOR.resolve_release(REPOSITORY, TAG)
        urls = [call.args[0] for call in read.call_args_list]
        self.assertEqual(urls.count(self.assets[1]["browser_download_url"]), 1)
        self.assertEqual(urls.count(self.assets[0]["browser_download_url"]), 2)

    def test_bundle_cache_hit_and_corruption_refetch(self):
        payload = b"verified Golden Set ZIP fixture"
        sha256 = hashlib.sha256(payload).hexdigest()
        bundle_url = f"https://github.com/{REPOSITORY}/releases/download/{TAG}/images.zip"
        release = {"bundle": {"size": len(payload), "sha256": sha256}, "bundle_url": bundle_url}
        server = DIRECTOR.DirectorServer(("127.0.0.1", 0))
        server.releases[(REPOSITORY, TAG)] = release
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def stop_server():
            server.shutdown()
            thread.join(5)
            server.server_close()
        self.addCleanup(stop_server)

        def upstream(url, accept="application/octet-stream"):
            self.assertEqual(url, bundle_url)
            response = io.BytesIO(payload)
            response.headers = {"Content-Length": str(len(payload))}
            return response

        query = urllib.parse.urlencode({"source_repo": f"https://github.com/{REPOSITORY}", "golden_set_id": TAG})
        url = f"http://127.0.0.1:{server.server_port}/api/image-bundle?{query}"
        with patch.object(DIRECTOR, "_open", side_effect=upstream) as opened:
            with urllib.request.urlopen(url) as response:
                self.assertEqual(response.read(), payload)
                self.assertEqual(response.headers["X-HTH-Cache"], "miss")
            with urllib.request.urlopen(url) as response:
                self.assertEqual(response.read(), payload)
                self.assertEqual(response.headers["X-HTH-Cache"], "hit")
            self.assertEqual(opened.call_count, 1)
            DIRECTOR._cache_path(sha256).write_bytes(b"corrupt")
            with urllib.request.urlopen(url) as response:
                self.assertEqual(response.read(), payload)
                self.assertEqual(response.headers["X-HTH-Cache"], "miss")
            self.assertEqual(opened.call_count, 2)

    def test_rejects_changed_bundle_digest(self):
        self.assets[2]["digest"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "Image bundle metadata"):
            self.resolve()

    def test_rejects_mismatched_page_membership(self):
        self.freeze["membership"]["global_ordinals"][0] = 4
        with self.assertRaisesRegex(ValueError, "page membership"):
            self.resolve()

    def test_results_pull_uses_only_matching_clean_sibling_checkout(self):
        def completed(output=""):
            return subprocess.CompletedProcess([], 0, output, "")

        with tempfile.TemporaryDirectory() as directory:
            responses = [
                completed(f"git@github.com:{REPOSITORY}-results.git\n"),
                completed(), completed("a" * 40 + "\n"),
                completed("Already up to date.\n"), completed("a" * 40 + "\n"),
            ]
            with patch.object(DIRECTOR, "results_checkout", return_value=Path(directory)):
                with patch.object(DIRECTOR.subprocess, "run", side_effect=responses) as run:
                    result = DIRECTOR.update_results_checkout(REPOSITORY)
            self.assertEqual(result["status"], "already current")
            self.assertEqual(run.call_args_list[3].args[0][-2:], ["pull", "--ff-only"])

    def test_results_pull_refuses_wrong_remote(self):
        with tempfile.TemporaryDirectory() as directory:
            wrong = subprocess.CompletedProcess([], 0, "https://github.com/other/repo.git\n", "")
            with patch.object(DIRECTOR, "results_checkout", return_value=Path(directory)):
                with patch.object(DIRECTOR.subprocess, "run", return_value=wrong):
                    with self.assertRaisesRegex(ValueError, "origin does not match"):
                        DIRECTOR.update_results_checkout(REPOSITORY)


if __name__ == "__main__":
    unittest.main()
