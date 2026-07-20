from __future__ import annotations

from pathlib import Path

from hth.regression.io import environment_info


def test_environment_info_contains_runner_metrics() -> None:
    info = environment_info(Path.cwd())
    assert info["execution_target"] in {"Local", "GitHub Hosted"}
    assert info["runner_name"]
    assert info["machine_name"]
    assert info["runner_os"]
    assert info["runner_arch"]
    assert info["cpu_model"]
    assert isinstance(info["logical_cpu_count"], int)
    assert info["logical_cpu_count"] > 0
    assert info["python_version"]
    assert info["opencv_version"]
    assert info["numpy_version"]
