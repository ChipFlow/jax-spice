"""Tests for openvaf-py compilation and metadata extraction."""

import pytest
from pathlib import Path

import openvaf_py

# Base paths
REPO_ROOT = Path(__file__).parent.parent.parent
VACASK_DEVICES = REPO_ROOT / "vendor" / "VACASK" / "devices"
OPENVAF_TESTS = REPO_ROOT / "vendor" / "OpenVAF" / "integration_tests"


class TestBasicCompilation:
    """Test basic compilation of simple models."""

    def test_compile_capacitor(self):
        """Test compiling the capacitor model."""
        modules = openvaf_py.compile_va(str(VACASK_DEVICES / "capacitor.va"))
        assert len(modules) == 1
        cap = modules[0]
        assert cap.name == "capacitor"
        assert cap.num_residuals == 2
        assert cap.num_cached_values == 2

    def test_compile_resistor(self):
        """Test compiling the resistor model."""
        modules = openvaf_py.compile_va(str(VACASK_DEVICES / "resistor.va"))
        assert len(modules) == 1
        res = modules[0]
        assert res.name == "resistor"
        assert res.num_residuals == 2

    def test_compile_diode(self):
        """Test compiling the diode model."""
        modules = openvaf_py.compile_va(str(VACASK_DEVICES / "diode.va"))
        assert len(modules) == 1
        diode = modules[0]
        assert diode.name == "diode"
        assert diode.num_residuals == 3  # A, K, internal node


class TestComplexModels:
    """Test compilation of complex models."""

    @pytest.mark.skipif(
        not (OPENVAF_TESTS / "PSP103" / "psp103.va").exists(),
        reason="PSP103 model not available"
    )
    def test_compile_psp103(self):
        """Test compiling PSP103 - complex MOSFET model."""
        modules = openvaf_py.compile_va(str(OPENVAF_TESTS / "PSP103" / "psp103.va"))
        assert len(modules) == 1
        psp = modules[0]
        assert psp.name == "PSP103VA"
        assert psp.num_terminals == 4  # D, G, S, B
        assert psp.num_cached_values == 462
        assert psp.num_collapsible == 7

    @pytest.mark.skipif(
        not (OPENVAF_TESTS / "BSIM4" / "bsim4.va").exists(),
        reason="BSIM4 model not available"
    )
    def test_compile_bsim4(self):
        """Test compiling BSIM4 model."""
        modules = openvaf_py.compile_va(str(OPENVAF_TESTS / "BSIM4" / "bsim4.va"))
        assert len(modules) == 1
        bsim = modules[0]
        assert "bsim4" in bsim.name.lower()
        assert bsim.num_cached_values > 100  # BSIM4 has many cache values

    @pytest.mark.skipif(
        not (OPENVAF_TESTS / "HICUML2" / "hicumL2.va").exists(),
        reason="HICUM model not available"
    )
    def test_compile_hicum(self):
        """Test compiling HICUM L2 model."""
        modules = openvaf_py.compile_va(str(OPENVAF_TESTS / "HICUML2" / "hicumL2.va"))
        assert len(modules) == 1
        hicum = modules[0]
        assert "hicum" in hicum.name.lower()


class TestMetadataExtraction:
    """Test metadata extraction functions."""

    @pytest.fixture
    def capacitor(self):
        """Load capacitor module."""
        modules = openvaf_py.compile_va(str(VACASK_DEVICES / "capacitor.va"))
        return modules[0]

    def test_get_codegen_metadata(self, capacitor):
        """Test get_codegen_metadata returns expected structure."""
        metadata = capacitor.get_codegen_metadata()

        assert "eval_param_mapping" in metadata
        assert "init_param_mapping" in metadata
        assert "cache_info" in metadata
        assert "residuals" in metadata
        assert "jacobian" in metadata
        assert "model_name" in metadata
        assert metadata["model_name"] == "capacitor"

    def test_given_suffix_fix(self, capacitor):
        """Test that param_given params have _given suffix to avoid collisions."""
        metadata = capacitor.get_codegen_metadata()
        init_map = metadata["init_param_mapping"]

        # Should have separate keys for c and c_given
        assert "c" in init_map
        assert "c_given" in init_map
        assert init_map["c"] != init_map["c_given"]

        # No duplicate values for different semantic meanings
        keys = list(init_map.keys())
        assert len(keys) == len(set(keys)), "Duplicate keys in init_param_mapping"

    def test_get_mir_instructions(self, capacitor):
        """Test MIR instruction extraction."""
        mir = capacitor.get_mir_instructions()

        assert "constants" in mir
        assert "params" in mir
        assert "instructions" in mir
        assert "blocks" in mir
        assert len(mir["instructions"]) > 0

    def test_get_init_mir_instructions(self, capacitor):
        """Test init MIR instruction extraction."""
        init_mir = capacitor.get_init_mir_instructions()

        assert "constants" in init_mir
        assert "cache_mapping" in init_mir
        assert len(init_mir["cache_mapping"]) == capacitor.num_cached_values

    def test_get_dae_system(self, capacitor):
        """Test DAE system extraction."""
        dae = capacitor.get_dae_system()

        assert "nodes" in dae
        assert "residuals" in dae
        assert "jacobian" in dae
        assert "terminals" in dae
        assert len(dae["residuals"]) == capacitor.num_residuals
        assert len(dae["jacobian"]) == capacitor.num_jacobian

    def test_get_osdi_descriptor(self, capacitor):
        """Test OSDI descriptor extraction."""
        osdi = capacitor.get_osdi_descriptor()

        assert "params" in osdi
        assert "nodes" in osdi
        assert "jacobian" in osdi
        assert osdi["num_terminals"] == 2

    def test_get_cache_mapping(self, capacitor):
        """Test cache mapping extraction."""
        cache_map = capacitor.get_cache_mapping()

        assert len(cache_map) == capacitor.num_cached_values
        for init_val, eval_param in cache_map:
            assert isinstance(init_val, int)
            assert isinstance(eval_param, int)


class TestPSP103Metadata:
    """Detailed tests for PSP103 metadata (complex model)."""

    @pytest.fixture
    def psp103(self):
        """Load PSP103 module."""
        path = OPENVAF_TESTS / "PSP103" / "psp103.va"
        if not path.exists():
            pytest.skip("PSP103 model not available")
        modules = openvaf_py.compile_va(str(path))
        return modules[0]

    def test_psp103_given_suffix(self, psp103):
        """Test _given suffix for PSP103's many parameters."""
        metadata = psp103.get_codegen_metadata()
        init_map = metadata["init_param_mapping"]

        # Count _given suffixed params
        given_params = [k for k in init_map.keys() if k.endswith("_given")]

        # PSP103 should have many param_given entries
        assert len(given_params) > 50, f"Expected many _given params, got {len(given_params)}"

        # No duplicates
        keys = list(init_map.keys())
        assert len(keys) == len(set(keys)), "Duplicate keys in init_param_mapping"

    def test_psp103_cache_values(self, psp103):
        """Test PSP103 has expected number of cache values."""
        assert psp103.num_cached_values == 462

    def test_psp103_collapsible_pairs(self, psp103):
        """Test PSP103 has collapsible node pairs."""
        assert psp103.num_collapsible == 7
        assert len(psp103.collapsible_pairs) == 7

    def test_psp103_dae_structure(self, psp103):
        """Test PSP103 DAE system structure."""
        dae = psp103.get_dae_system()

        assert dae["num_terminals"] == 4  # D, G, S, B
        assert dae["num_internal"] == 9  # Internal nodes
        assert len(dae["residuals"]) == 13  # Total nodes
