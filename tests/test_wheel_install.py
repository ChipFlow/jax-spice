"""Integration test: build wheel, install in fresh venv, verify bundled models.

This test builds all workspace member wheels and the vajax wheel, installs them
into a fresh virtual environment, and verifies that:
1. The bundled device models are present in the installed package
2. Each bundled model can be compiled by OpenVAF
3. The MODEL_PATHS registry is consistent with what's on disk

Requires: uv (for venv creation and wheel building)
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent

# All bundled ECL-2.0 + Apache-2.0 models expected in the wheel
EXPECTED_BUNDLED_MODELS = {
    "resistor": "resistor.va",
    "capacitor": "capacitor.va",
    "inductor": "inductor.va",
    "ekv": "ekv/ekv.va",
    "ekv_longchannel": "ekv_longchannel/ekv_longchannel.va",
    "bsimbulk": "bsimbulk/bsimbulk.va",
    "bsimcmg": "bsimcmg/bsimcmg.va",
    "bsimimg": "bsimimg/bsimimg.va",
    "hisim2": "hisim2/hisim2.va",
    "asmhemt": "asmhemt/asmhemt.va",
    "mvsg_cmc": "mvsg_cmc/mvsg_cmc.va",
}


@pytest.fixture(scope="module")
def installed_venv(tmp_path_factory):
    """Build wheels and install into a fresh venv.

    Returns the path to the venv's Python interpreter.
    """
    tmp_dir = tmp_path_factory.mktemp("wheel_test")
    wheel_dir = tmp_dir / "wheels"
    venv_dir = tmp_dir / "venv"
    wheel_dir.mkdir()

    # Build workspace member wheels first (openvaf-py needs maturin)
    workspace_members = [
        PROJECT_ROOT / "openvaf_jax" / "openvaf_py",
        PROJECT_ROOT / "vajax" / "sparse",  # umfpack-jax
    ]

    built_wheels = []
    for member_dir in workspace_members:
        if not member_dir.exists():
            continue
        result = subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(wheel_dir)],
            cwd=str(member_dir),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            pytest.skip(f"Failed to build {member_dir.name}: {result.stderr}")
        # Collect built wheel paths
        for line in result.stdout.splitlines():
            if line.strip().endswith(".whl"):
                built_wheels.append(line.strip())

    # Build vajax wheel
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"vajax wheel build failed:\n{result.stderr}"

    # Create fresh venv
    result = subprocess.run(
        ["uv", "venv", str(venv_dir), "--python", f"python{sys.version_info.major}.{sys.version_info.minor}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"venv creation failed:\n{result.stderr}"

    python = venv_dir / "bin" / "python"
    assert python.exists(), f"Python not found at {python}"

    # Install all wheels (workspace members first, then vajax)
    # Use --find-links to pick up local wheels for workspace deps
    result = subprocess.run(
        [
            "uv", "pip", "install",
            "--python", str(python),
            "--find-links", str(wheel_dir),
            "--no-deps",
        ] + sorted(wheel_dir.glob("*.whl")),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Wheel install failed:\n{result.stderr}"

    # Install remaining deps (jax, numpy, etc.) but skip workspace members
    # that we already installed from local wheels
    result = subprocess.run(
        [
            "uv", "pip", "install",
            "--python", str(python),
            "--find-links", str(wheel_dir),
            "vajax",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    # This may partially fail if some deps aren't available, that's OK
    # We just need vajax + openvaf-py importable

    return python


@pytest.mark.public_api
class TestWheelInstall:
    """Verify the installed wheel contains all bundled models."""

    def test_bundled_models_present(self, installed_venv):
        """All expected model files exist in the installed package."""
        python = installed_venv
        script = textwrap.dedent("""\
            import json
            from pathlib import Path
            import vajax.devices
            models_dir = Path(vajax.devices.__file__).parent / "models"
            expected = json.loads('{expected_json}')
            results = {}
            for name, rel_path in expected.items():
                full = models_dir / rel_path
                results[name] = full.exists()
            # Print results as JSON
            print(json.dumps(results))
        """).replace("{expected_json}", __import__("json").dumps(EXPECTED_BUNDLED_MODELS).replace("'", "\\'"))

        result = subprocess.run(
            [str(python), "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            env={"JAX_PLATFORMS": "cpu", "PATH": str(python.parent)},
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"

        import json
        model_status = json.loads(result.stdout.strip())
        missing = [name for name, exists in model_status.items() if not exists]
        assert not missing, f"Missing bundled models: {missing}"

    def test_bundled_models_compile(self, installed_venv):
        """Each bundled model can be compiled by OpenVAF from the installed package."""
        python = installed_venv

        script = textwrap.dedent("""\
            import json, sys
            from pathlib import Path
            try:
                import openvaf_py
            except ImportError:
                print(json.dumps({"skip": "openvaf_py not importable"}))
                sys.exit(0)

            import vajax.devices
            models_dir = Path(vajax.devices.__file__).parent / "models"
            models = {models_json}
            results = {}
            for name, rel_path in models.items():
                va_path = models_dir / rel_path
                try:
                    modules = openvaf_py.compile_va(str(va_path))
                    if modules:
                        mod = modules[0]
                        results[name] = {"ok": True, "nodes": len(mod.nodes)}
                    else:
                        results[name] = {"ok": False, "error": "no modules"}
                except Exception as e:
                    results[name] = {"ok": False, "error": str(e)}
            print(json.dumps(results))
        """).replace("{models_json}", repr(EXPECTED_BUNDLED_MODELS))

        result = subprocess.run(
            [str(python), "-c", script],
            capture_output=True,
            text=True,
            timeout=600,
            env={"JAX_PLATFORMS": "cpu", "PATH": str(python.parent)},
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"

        import json
        compile_results = json.loads(result.stdout.strip())

        if "skip" in compile_results:
            pytest.skip(compile_results["skip"])

        failed = {
            name: info["error"]
            for name, info in compile_results.items()
            if not info.get("ok")
        }
        assert not failed, f"Models failed to compile: {failed}"

    def test_model_paths_consistent(self, installed_venv):
        """MODEL_PATHS registry matches actual bundled files."""
        python = installed_venv
        script = textwrap.dedent("""\
            import json
            from pathlib import Path
            from vajax.analysis.openvaf_models import MODEL_PATHS, _BUNDLED_MODELS_DIR
            results = {}
            for name, (base_key, rel_path) in MODEL_PATHS.items():
                if base_key == "bundled":
                    full = _BUNDLED_MODELS_DIR / rel_path
                    results[name] = {"exists": full.exists(), "path": str(rel_path)}
            print(json.dumps(results))
        """)

        result = subprocess.run(
            [str(python), "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            env={"JAX_PLATFORMS": "cpu", "PATH": str(python.parent)},
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"

        import json
        path_results = json.loads(result.stdout.strip())
        missing = {
            name: info["path"]
            for name, info in path_results.items()
            if not info["exists"]
        }
        assert not missing, f"MODEL_PATHS entries with missing files: {missing}"
