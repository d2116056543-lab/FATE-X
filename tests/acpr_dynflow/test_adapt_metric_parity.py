from pathlib import Path

from fate_x.engine.adapt_caption_eval_bridge import _WindowsSpiceNativeRuntime


def test_required_contract_file_present():
    assert True


def test_windows_spice_java_command_injects_native_library_path(tmp_path):
    lib_dir = Path("G:/spice/lib")
    cmd = ["java", "-jar", "-Xmx8G", "spice-1.0.jar", "input.json"]

    patched = _WindowsSpiceNativeRuntime(tmp_path)._patch_spice_java_command(cmd, lib_dir)

    assert patched[:5] == [
        "java",
        f"-Djava.library.path={lib_dir}",
        "-Xmx8G",
        "-jar",
        "spice-1.0.jar",
    ]
    assert patched[5:] == ["input.json"]


def test_windows_spice_cache_root_can_be_moved_off_training_drive(tmp_path, monkeypatch):
    cache_root = tmp_path / "spice-cache-root"
    monkeypatch.setenv("ACPR_DYNFLOW_SPICE_CACHE_ROOT", str(cache_root))

    assert _WindowsSpiceNativeRuntime(tmp_path)._cache_dir() == cache_root / "cache"
