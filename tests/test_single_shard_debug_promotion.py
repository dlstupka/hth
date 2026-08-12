from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run-detector-regressions.sh"


def test_single_shard_debug_is_promoted_out_of_hidden_shard_workspace():
    text = SCRIPT.read_text(encoding="utf-8")
    single = text.index('if (( ${#detector_shard_dirs[@]} == 1 )); then')
    merged = text.index('  else', single)
    block = text[single:merged]
    assert 'single_shard_root="$(dirname "$(dirname "$source_dir")")"' in block
    assert '[[ -d "$single_shard_root/debug" ]]' in block
    assert 'mkdir -p "$OUTPUT_DIR/debug"' in block
    assert 'cp -a "$single_shard_root/debug/." "$OUTPUT_DIR/debug/"' in block
