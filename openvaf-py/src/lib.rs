//! OpenVAF Python bindings for MIR metadata extraction
//!
//! This module provides Python bindings to compile Verilog-A files using OpenVAF
//! and extract metadata needed for code generation. It does NOT include interpreter
//! functionality - use osdi-py for reference evaluation against OSDI libraries.

use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use basedb::diagnostics::ConsoleSink;
use hir::{CompilationDB, CompilationOpts, Expr, Literal, Type};
use hir_lower::{CurrentKind, ParamKind, PlaceKind};
use syntax::ast::UnaryOp;
use lasso::Rodeo;
use mir::{FuncRef, Function, Param, Value, F_ZERO, ValueDef};
use paths::AbsPathBuf;
use sim_back::{collect_modules, CompiledModule};

// OSDI flag constants (matching osdi_0_4.rs)
const PARA_TY_REAL: u32 = 0;
const PARA_TY_INT: u32 = 1;
const PARA_TY_STR: u32 = 2;
const PARA_KIND_MODEL: u32 = 0 << 30;
const PARA_KIND_INST: u32 = 1 << 30;
const JACOBIAN_ENTRY_RESIST_CONST: u32 = 1;
const JACOBIAN_ENTRY_REACT_CONST: u32 = 2;
const JACOBIAN_ENTRY_RESIST: u32 = 4;
const JACOBIAN_ENTRY_REACT: u32 = 8;

/// Parameter metadata matching OSDI descriptor
#[derive(Clone)]
struct OsdiParamInfo {
    name: String,
    aliases: Vec<String>,
    units: String,
    description: String,
    flags: u32,
    is_instance: bool,
}

/// Node metadata matching OSDI descriptor
#[derive(Clone)]
struct OsdiNodeInfo {
    name: String,
    kind: String,  // "KirchoffLaw", "BranchCurrent", or "Implicit"
    units: String,
    residual_units: String,
    is_internal: bool,
}

/// Jacobian entry metadata matching OSDI descriptor
#[derive(Clone)]
struct OsdiJacobianInfo {
    row: u32,
    col: u32,
    flags: u32,
}

/// Noise source metadata
#[derive(Clone)]
struct OsdiNoiseInfo {
    name: String,
    node1: u32,
    node2: u32,  // u32::MAX for ground
}

/// Python wrapper for a compiled Verilog-A module
///
/// This struct contains all metadata extracted from OpenVAF compilation,
/// suitable for code generation. It does NOT contain interpreter functionality.
#[pyclass]
struct VaModule {
    /// Module name
    #[pyo3(get)]
    name: String,

    // === Eval function metadata ===
    /// Parameter names in order (for eval function)
    #[pyo3(get)]
    param_names: Vec<String>,
    /// Parameter types/kinds
    #[pyo3(get)]
    param_kinds: Vec<String>,
    /// Parameter Value indices
    #[pyo3(get)]
    param_value_indices: Vec<u32>,
    /// Node names
    #[pyo3(get)]
    nodes: Vec<String>,
    /// Number of residual equations
    #[pyo3(get)]
    num_residuals: usize,
    /// Number of Jacobian entries
    #[pyo3(get)]
    num_jacobian: usize,
    /// Number of eval function parameters
    #[pyo3(get)]
    func_num_params: usize,
    /// Callback descriptions
    #[pyo3(get)]
    callback_names: Vec<String>,

    // MIR indices for residuals and Jacobian (internal)
    residual_resist_indices: Vec<u32>,
    residual_react_indices: Vec<u32>,
    residual_resist_lim_rhs_indices: Vec<u32>,
    residual_react_lim_rhs_indices: Vec<u32>,
    jacobian_resist_indices: Vec<u32>,
    jacobian_react_indices: Vec<u32>,
    jacobian_rows: Vec<u32>,
    jacobian_cols: Vec<u32>,

    /// The compiled MIR function for evaluation
    eval_func: Function,

    // === Init function metadata ===
    /// Init function parameter names
    #[pyo3(get)]
    init_param_names: Vec<String>,
    /// Init function parameter kinds
    #[pyo3(get)]
    init_param_kinds: Vec<String>,
    /// Init function parameter Value indices
    #[pyo3(get)]
    init_param_value_indices: Vec<u32>,
    /// Number of init function parameters
    #[pyo3(get)]
    init_num_params: usize,
    /// Number of cached values from init
    #[pyo3(get)]
    num_cached_values: usize,

    /// The init MIR function
    init_func: Function,
    /// Cache slot mapping: (init_value_index, eval_param_index)
    cache_mapping: Vec<(u32, u32)>,

    // === Node collapse metadata ===
    /// Collapsible node pairs: (node1_idx, node2_idx_or_ground)
    #[pyo3(get)]
    collapsible_pairs: Vec<(u32, u32)>,
    /// Number of collapsible pairs
    #[pyo3(get)]
    num_collapsible: usize,
    /// Collapse decision outputs: (eq_index, value_name)
    #[pyo3(get)]
    collapse_decision_outputs: Vec<(u32, String)>,

    // === Parameter defaults ===
    param_defaults: HashMap<String, f64>,
    /// String constant values (resolved from Spur keys)
    str_constant_values: HashMap<String, String>,

    // === OSDI descriptor metadata ===
    /// Number of terminal nodes (ports)
    #[pyo3(get)]
    num_terminals: usize,
    /// Number of limiting states
    #[pyo3(get)]
    num_states: usize,
    /// Whether module has bound_step
    #[pyo3(get)]
    has_bound_step: bool,

    osdi_params: Vec<OsdiParamInfo>,
    osdi_nodes: Vec<OsdiNodeInfo>,
    osdi_jacobian: Vec<OsdiJacobianInfo>,
    osdi_noise_sources: Vec<OsdiNoiseInfo>,
}

#[pymethods]
impl VaModule {
    fn __repr__(&self) -> String {
        format!(
            "VaModule(name='{}', params={}, nodes={}, residuals={}, jacobian={})",
            self.name,
            self.param_names.len(),
            self.nodes.len(),
            self.num_residuals,
            self.num_jacobian
        )
    }

    /// Get all Param-defined values in the eval function
    fn get_all_func_params(&self) -> Vec<(u32, u32)> {
        let mut result = Vec::new();
        for val in self.eval_func.dfg.values.iter() {
            if let mir::ValueDef::Param(p) = self.eval_func.dfg.value_def(val) {
                result.push((u32::from(p), u32::from(val)));
            }
        }
        result.sort_by_key(|(param_idx, _)| *param_idx);
        result
    }

    /// Get MIR function as string for debugging
    fn get_mir(&self, literals: Vec<String>) -> String {
        let mut rodeo = lasso::Rodeo::new();
        for lit in literals {
            rodeo.get_or_intern(&lit);
        }
        self.eval_func.print(&rodeo).to_string()
    }

    /// Get number of function calls in the MIR
    fn get_num_func_calls(&self) -> usize {
        self.eval_func.dfg.signatures.len()
    }

    /// Get cache mapping as list of (init_value_idx, eval_param_idx)
    fn get_cache_mapping(&self) -> Vec<(u32, u32)> {
        self.cache_mapping.clone()
    }

    /// Get parameter defaults from Verilog-A source
    fn get_param_defaults(&self) -> HashMap<String, f64> {
        self.param_defaults.clone()
    }

    /// Get resolved string constant values
    fn get_str_constants(&self) -> HashMap<String, String> {
        self.str_constant_values.clone()
    }

    /// Get OSDI-compatible descriptor metadata
    fn get_osdi_descriptor(&self) -> HashMap<String, PyObject> {
        use pyo3::types::{PyDict, PyList};

        Python::with_gil(|py| {
            let mut result = HashMap::new();

            // Parameters
            let params = PyList::empty(py);
            for param in &self.osdi_params {
                let d = PyDict::new(py);
                d.set_item("name", &param.name).unwrap();
                d.set_item("aliases", &param.aliases).unwrap();
                d.set_item("units", &param.units).unwrap();
                d.set_item("description", &param.description).unwrap();
                d.set_item("flags", param.flags).unwrap();
                d.set_item("is_instance", param.is_instance).unwrap();
                d.set_item("is_model_param", (param.flags & PARA_KIND_INST) == 0).unwrap();
                params.append(d).unwrap();
            }
            result.insert("params".to_string(), params.into());

            // Nodes
            let nodes = PyList::empty(py);
            for node in &self.osdi_nodes {
                let d = PyDict::new(py);
                d.set_item("name", &node.name).unwrap();
                d.set_item("units", &node.units).unwrap();
                d.set_item("residual_units", &node.residual_units).unwrap();
                d.set_item("is_internal", node.is_internal).unwrap();
                nodes.append(d).unwrap();
            }
            result.insert("nodes".to_string(), nodes.into());

            // Jacobian entries with flags
            let jacobian = PyList::empty(py);
            for entry in &self.osdi_jacobian {
                let d = PyDict::new(py);
                d.set_item("row", entry.row).unwrap();
                d.set_item("col", entry.col).unwrap();
                d.set_item("flags", entry.flags).unwrap();
                d.set_item("has_resist", (entry.flags & JACOBIAN_ENTRY_RESIST) != 0).unwrap();
                d.set_item("has_react", (entry.flags & JACOBIAN_ENTRY_REACT) != 0).unwrap();
                d.set_item("resist_const", (entry.flags & JACOBIAN_ENTRY_RESIST_CONST) != 0).unwrap();
                d.set_item("react_const", (entry.flags & JACOBIAN_ENTRY_REACT_CONST) != 0).unwrap();
                jacobian.append(d).unwrap();
            }
            result.insert("jacobian".to_string(), jacobian.into());

            // Collapsible pairs
            let collapsible = PyList::empty(py);
            for (n1, n2) in &self.collapsible_pairs {
                let pair = PyList::empty(py);
                pair.append(*n1).unwrap();
                if *n2 == u32::MAX {
                    pair.append("gnd").unwrap();
                } else {
                    pair.append(*n2).unwrap();
                }
                collapsible.append(pair).unwrap();
            }
            result.insert("collapsible".to_string(), collapsible.into());

            // Noise sources
            let noise = PyList::empty(py);
            for src in &self.osdi_noise_sources {
                let d = PyDict::new(py);
                d.set_item("name", &src.name).unwrap();
                d.set_item("node1", src.node1).unwrap();
                if src.node2 == u32::MAX {
                    d.set_item("node2", "gnd").unwrap();
                } else {
                    d.set_item("node2", src.node2).unwrap();
                }
                noise.append(d).unwrap();
            }
            result.insert("noise_sources".to_string(), noise.into());

            // Scalar values
            result.insert("num_terminals".to_string(), self.num_terminals.into_py(py));
            result.insert("num_states".to_string(), self.num_states.into_py(py));
            result.insert("has_bound_step".to_string(), self.has_bound_step.into_py(py));
            result.insert("num_nodes".to_string(), self.osdi_nodes.len().into_py(py));
            result.insert("num_params".to_string(), self.osdi_params.len().into_py(py));
            result.insert("num_jacobian_entries".to_string(), self.osdi_jacobian.len().into_py(py));
            result.insert("num_collapsible".to_string(), self.collapsible_pairs.len().into_py(py));
            result.insert("num_noise_sources".to_string(), self.osdi_noise_sources.len().into_py(py));

            result
        })
    }

    /// Export MIR instructions for JAX translation (eval function)
    fn get_mir_instructions(&self) -> HashMap<String, PyObject> {
        export_mir_instructions(&self.eval_func, &self.callback_names)
    }

    /// Export init function MIR instructions for JAX translation
    fn get_init_mir_instructions(&self) -> HashMap<String, PyObject> {
        let mut result = export_mir_instructions(&self.init_func, &[]);

        // Add cache mapping
        Python::with_gil(|py| {
            use pyo3::types::{PyDict, PyList};
            let cache_map = PyList::empty(py);
            for (init_val, eval_param) in &self.cache_mapping {
                let entry = PyDict::new(py);
                entry.set_item("init_value", format!("v{}", init_val)).unwrap();
                entry.set_item("eval_param", *eval_param).unwrap();
                cache_map.append(entry).unwrap();
            }
            result.insert("cache_mapping".to_string(), cache_map.into());
        });

        result
    }

    /// Export DAE system (residuals and Jacobian) with clear naming
    fn get_dae_system(&self) -> HashMap<String, PyObject> {
        use pyo3::types::{PyDict, PyList};

        Python::with_gil(|py| {
            let mut result = HashMap::new();

            // Node information
            let nodes_list = PyList::empty(py);
            let mut terminal_names: Vec<String> = Vec::new();
            let mut internal_names: Vec<String> = Vec::new();

            for (i, node_info) in self.osdi_nodes.iter().enumerate() {
                let node_dict = PyDict::new(py);
                node_dict.set_item("idx", i).unwrap();
                node_dict.set_item("name", &node_info.name).unwrap();
                node_dict.set_item("kind", &node_info.kind).unwrap();
                node_dict.set_item("is_internal", node_info.is_internal).unwrap();
                nodes_list.append(node_dict).unwrap();

                if node_info.is_internal {
                    internal_names.push(node_info.name.clone());
                } else {
                    terminal_names.push(node_info.name.clone());
                }
            }
            result.insert("nodes".to_string(), nodes_list.into());

            // Residuals
            let residuals_list = PyList::empty(py);
            for i in 0..self.num_residuals {
                let res_dict = PyDict::new(py);
                res_dict.set_item("equation_idx", i).unwrap();
                res_dict.set_item("node_idx", i).unwrap();
                let node_name = if i < self.osdi_nodes.len() {
                    &self.osdi_nodes[i].name
                } else {
                    "unknown"
                };
                res_dict.set_item("node_name", node_name).unwrap();
                res_dict.set_item("resist_var", format!("mir_{}", self.residual_resist_indices[i])).unwrap();
                res_dict.set_item("react_var", format!("mir_{}", self.residual_react_indices[i])).unwrap();
                res_dict.set_item("resist_lim_rhs_var", format!("mir_{}", self.residual_resist_lim_rhs_indices[i])).unwrap();
                res_dict.set_item("react_lim_rhs_var", format!("mir_{}", self.residual_react_lim_rhs_indices[i])).unwrap();
                residuals_list.append(res_dict).unwrap();
            }
            result.insert("residuals".to_string(), residuals_list.into());

            // Jacobian
            let jacobian_list = PyList::empty(py);
            for i in 0..self.num_jacobian {
                let jac_dict = PyDict::new(py);
                jac_dict.set_item("entry_idx", i).unwrap();

                let row_idx = self.jacobian_rows[i] as usize;
                let col_idx = self.jacobian_cols[i] as usize;

                jac_dict.set_item("row_node_idx", row_idx).unwrap();
                jac_dict.set_item("col_node_idx", col_idx).unwrap();

                let row_name = if row_idx < self.osdi_nodes.len() {
                    &self.osdi_nodes[row_idx].name
                } else {
                    "unknown"
                };
                let col_name = if col_idx < self.osdi_nodes.len() {
                    &self.osdi_nodes[col_idx].name
                } else {
                    "unknown"
                };
                jac_dict.set_item("row_node_name", row_name).unwrap();
                jac_dict.set_item("col_node_name", col_name).unwrap();

                jac_dict.set_item("resist_var", format!("mir_{}", self.jacobian_resist_indices[i])).unwrap();
                jac_dict.set_item("react_var", format!("mir_{}", self.jacobian_react_indices[i])).unwrap();

                if i < self.osdi_jacobian.len() {
                    let flags = self.osdi_jacobian[i].flags;
                    jac_dict.set_item("has_resist", (flags & JACOBIAN_ENTRY_RESIST) != 0).unwrap();
                    jac_dict.set_item("has_react", (flags & JACOBIAN_ENTRY_REACT) != 0).unwrap();
                } else {
                    jac_dict.set_item("has_resist", true).unwrap();
                    jac_dict.set_item("has_react", true).unwrap();
                }

                jacobian_list.append(jac_dict).unwrap();
            }
            result.insert("jacobian".to_string(), jacobian_list.into());

            // Terminal and internal node lists
            result.insert("terminals".to_string(), terminal_names.clone().into_py(py));
            result.insert("internal_nodes".to_string(), internal_names.clone().into_py(py));
            result.insert("num_terminals".to_string(), terminal_names.len().into_py(py));
            result.insert("num_internal".to_string(), internal_names.len().into_py(py));

            // Collapsible pairs
            let collapsible_list = PyList::empty(py);
            for (i, (n1, n2)) in self.collapsible_pairs.iter().enumerate() {
                let pair_dict = PyDict::new(py);
                pair_dict.set_item("pair_idx", i).unwrap();
                pair_dict.set_item("node1_idx", *n1).unwrap();
                pair_dict.set_item("node2_idx", *n2).unwrap();

                let n1_name = if (*n1 as usize) < self.osdi_nodes.len() {
                    self.osdi_nodes[*n1 as usize].name.clone()
                } else {
                    format!("unknown_{}", n1)
                };
                let n2_name = if *n2 == u32::MAX {
                    "ground".to_string()
                } else if (*n2 as usize) < self.osdi_nodes.len() {
                    self.osdi_nodes[*n2 as usize].name.clone()
                } else {
                    format!("unknown_{}", n2)
                };
                pair_dict.set_item("node1_name", n1_name).unwrap();
                pair_dict.set_item("node2_name", n2_name).unwrap();

                if let Some((_, decision_var)) = self.collapse_decision_outputs.iter()
                    .find(|(idx, _)| *idx == i as u32) {
                    pair_dict.set_item("decision_var", decision_var).unwrap();
                }

                collapsible_list.append(pair_dict).unwrap();
            }
            result.insert("collapsible_pairs".to_string(), collapsible_list.into());
            result.insert("num_collapsible".to_string(), self.collapsible_pairs.len().into_py(py));

            result
        })
    }

    /// Get comprehensive metadata for code generation and validation
    ///
    /// IMPORTANT: This function includes the "_given" suffix fix for param_given parameters.
    /// This prevents duplicate keys in init_param_mapping (e.g., both 'c' param and 'c' param_given
    /// would collide without the suffix).
    fn get_codegen_metadata(&self) -> PyResult<HashMap<String, PyObject>> {
        use pyo3::types::{PyDict, PyList};

        Python::with_gil(|py| {
            let mut metadata = HashMap::new();

            // 1. Eval parameter mapping: semantic name → MIR variable
            //    Filters out hidden_state parameters (inlined by optimizer)
            let eval_param_map = PyDict::new(py);
            for ((name, value_idx), kind) in self.param_names.iter()
                .zip(&self.param_value_indices)
                .zip(&self.param_kinds)
            {
                if !kind.contains("hidden_state") {
                    eval_param_map.set_item(name, format!("v{}", value_idx))?;
                }
            }
            metadata.insert("eval_param_mapping".to_string(), eval_param_map.into());

            // 2. Init parameter mapping: semantic name → MIR variable
            //    CRITICAL FIX: Append "_given" suffix for param_given kinds to avoid
            //    duplicate keys (e.g., 'c' and 'c_given' instead of both being 'c')
            let init_param_map = PyDict::new(py);
            for ((name, value_idx), kind) in self.init_param_names.iter()
                .zip(&self.init_param_value_indices)
                .zip(&self.init_param_kinds)
            {
                if !kind.contains("hidden_state") {
                    let map_name = if kind.contains("param_given") {
                        format!("{}_given", name)
                    } else {
                        name.to_string()
                    };
                    init_param_map.set_item(map_name, format!("v{}", value_idx))?;
                }
            }
            metadata.insert("init_param_mapping".to_string(), init_param_map.into());

            // 3. Cache slot information
            let cache_info = PyList::empty(py);
            for (cache_idx, (init_val, eval_param)) in self.cache_mapping.iter().enumerate() {
                let entry = PyDict::new(py);
                entry.set_item("cache_idx", cache_idx)?;
                entry.set_item("init_value", format!("v{}", init_val))?;
                entry.set_item("eval_param", format!("v{}", eval_param))?;
                cache_info.append(entry)?;
            }
            metadata.insert("cache_info".to_string(), cache_info.into());

            // 4. Residual information
            let residuals = PyList::empty(py);
            for i in 0..self.num_residuals {
                let entry = PyDict::new(py);
                entry.set_item("residual_idx", i)?;
                entry.set_item("resist_var", format!("v{}", self.residual_resist_indices[i]))?;
                entry.set_item("react_var", format!("v{}", self.residual_react_indices[i]))?;
                if i < self.residual_resist_lim_rhs_indices.len() {
                    entry.set_item("resist_lim_rhs_var", format!("v{}", self.residual_resist_lim_rhs_indices[i]))?;
                }
                if i < self.residual_react_lim_rhs_indices.len() {
                    entry.set_item("react_lim_rhs_var", format!("v{}", self.residual_react_lim_rhs_indices[i]))?;
                }
                residuals.append(entry)?;
            }
            metadata.insert("residuals".to_string(), residuals.into());

            // 5. Jacobian information
            let jacobian = PyList::empty(py);
            for i in 0..self.num_jacobian {
                let entry = PyDict::new(py);
                entry.set_item("jacobian_idx", i)?;
                entry.set_item("row", self.jacobian_rows[i])?;
                entry.set_item("col", self.jacobian_cols[i])?;
                entry.set_item("resist_var", format!("v{}", self.jacobian_resist_indices[i]))?;
                entry.set_item("react_var", format!("v{}", self.jacobian_react_indices[i]))?;
                jacobian.append(entry)?;
            }
            metadata.insert("jacobian".to_string(), jacobian.into());

            // 6. Basic model info
            metadata.insert("model_name".to_string(), self.name.clone().into_py(py));
            metadata.insert("num_terminals".to_string(), self.nodes.len().into_py(py));
            metadata.insert("num_residuals".to_string(), self.num_residuals.into_py(py));
            metadata.insert("num_jacobian".to_string(), self.num_jacobian.into_py(py));
            metadata.insert("num_cache_slots".to_string(), self.num_cached_values.into_py(py));

            Ok(metadata)
        })
    }
}

/// Helper function to export MIR instructions from a Function
fn export_mir_instructions(func: &Function, callback_names: &[String]) -> HashMap<String, PyObject> {
    use pyo3::types::{PyDict, PyList};

    Python::with_gil(|py| {
        let mut result = HashMap::new();

        // Extract constants
        let constants = PyDict::new(py);
        let bool_constants = PyDict::new(py);
        let int_constants = PyDict::new(py);
        let str_constants = PyDict::new(py);

        for val in func.dfg.values.iter() {
            if let mir::ValueDef::Const(data) = func.dfg.value_def(val) {
                match data {
                    mir::Const::Float(ieee64) => {
                        let float_val: f64 = ieee64.into();
                        constants.set_item(format!("v{}", u32::from(val)), float_val).unwrap();
                    }
                    mir::Const::Bool(b) => {
                        bool_constants.set_item(format!("v{}", u32::from(val)), b).unwrap();
                    }
                    mir::Const::Int(i) => {
                        int_constants.set_item(format!("v{}", u32::from(val)), i).unwrap();
                    }
                    mir::Const::Str(spur) => {
                        str_constants.set_item(format!("v{}", u32::from(val)), spur.into_inner()).unwrap();
                    }
                }
            }
        }
        result.insert("constants".to_string(), constants.into());
        result.insert("bool_constants".to_string(), bool_constants.into());
        result.insert("int_constants".to_string(), int_constants.into());
        result.insert("str_constants".to_string(), str_constants.into());

        // Extract parameters
        let params = PyList::empty(py);
        let mut all_params: Vec<(u32, mir::Value)> = Vec::new();
        for val in func.dfg.values.iter() {
            if let mir::ValueDef::Param(p) = func.dfg.value_def(val) {
                all_params.push((u32::from(p), val));
            }
        }
        all_params.sort_by_key(|(param_idx, _)| *param_idx);
        for (_, val) in all_params.iter() {
            params.append(format!("v{}", u32::from(*val))).unwrap();
        }
        result.insert("params".to_string(), params.into());

        // Extract instructions
        let instructions = PyList::empty(py);
        for block in func.layout.blocks() {
            let block_name = format!("block{}", u32::from(block));

            for inst in func.layout.block_insts(block) {
                let inst_data = func.dfg.insts[inst].clone();
                let results = func.dfg.inst_results(inst);

                let inst_dict = PyDict::new(py);
                inst_dict.set_item("block", &block_name).unwrap();

                if !results.is_empty() {
                    inst_dict.set_item("result", format!("v{}", u32::from(results[0]))).unwrap();
                }

                let opcode = inst_data.opcode();
                inst_dict.set_item("opcode", opcode.name()).unwrap();

                match &inst_data {
                    mir::InstructionData::Unary { arg, .. } => {
                        inst_dict.set_item("operands", vec![format!("v{}", u32::from(*arg))]).unwrap();
                    }
                    mir::InstructionData::Binary { args, .. } => {
                        inst_dict.set_item("operands", vec![
                            format!("v{}", u32::from(args[0])),
                            format!("v{}", u32::from(args[1]))
                        ]).unwrap();
                    }
                    mir::InstructionData::Call { func_ref, args } => {
                        let args_slice = args.as_slice(&func.dfg.insts.value_lists);
                        let args_vec: Vec<String> = args_slice.iter()
                            .map(|v| format!("v{}", u32::from(*v)))
                            .collect();
                        inst_dict.set_item("operands", args_vec).unwrap();
                        inst_dict.set_item("func_ref", format!("inst{}", u32::from(*func_ref))).unwrap();
                    }
                    mir::InstructionData::PhiNode(phi) => {
                        let phi_ops = PyList::empty(py);
                        for (blk, val) in phi.edges(&func.dfg.insts.value_lists, &func.dfg.phi_forest) {
                            let edge = PyDict::new(py);
                            edge.set_item("value", format!("v{}", u32::from(val))).unwrap();
                            edge.set_item("block", format!("block{}", u32::from(blk))).unwrap();
                            phi_ops.append(edge).unwrap();
                        }
                        inst_dict.set_item("phi_operands", phi_ops).unwrap();
                    }
                    mir::InstructionData::Branch { cond, then_dst, else_dst, .. } => {
                        inst_dict.set_item("condition", format!("v{}", u32::from(*cond))).unwrap();
                        inst_dict.set_item("true_block", format!("block{}", u32::from(*then_dst))).unwrap();
                        inst_dict.set_item("false_block", format!("block{}", u32::from(*else_dst))).unwrap();
                    }
                    mir::InstructionData::Jump { destination } => {
                        inst_dict.set_item("destination", format!("block{}", u32::from(*destination))).unwrap();
                    }
                    mir::InstructionData::Exit => {}
                }

                instructions.append(inst_dict).unwrap();
            }
        }
        result.insert("instructions".to_string(), instructions.into());

        // Extract blocks
        let blocks = PyDict::new(py);
        for block in func.layout.blocks() {
            let block_name = format!("block{}", u32::from(block));
            let block_dict = PyDict::new(py);

            let mut predecessors = Vec::new();
            let mut successors = Vec::new();

            for other_block in func.layout.blocks() {
                if let Some(inst) = func.layout.block_insts(other_block).last() {
                    let inst_data = &func.dfg.insts[inst];
                    match inst_data {
                        mir::InstructionData::Branch { then_dst, else_dst, .. } => {
                            if *then_dst == block || *else_dst == block {
                                predecessors.push(format!("block{}", u32::from(other_block)));
                            }
                            if other_block == block {
                                successors.push(format!("block{}", u32::from(*then_dst)));
                                successors.push(format!("block{}", u32::from(*else_dst)));
                            }
                        }
                        mir::InstructionData::Jump { destination } => {
                            if *destination == block {
                                predecessors.push(format!("block{}", u32::from(other_block)));
                            }
                            if other_block == block {
                                successors.push(format!("block{}", u32::from(*destination)));
                            }
                        }
                        _ => {}
                    }
                }
            }

            block_dict.set_item("predecessors", predecessors).unwrap();
            block_dict.set_item("successors", successors).unwrap();
            blocks.set_item(&block_name, block_dict).unwrap();
        }
        result.insert("blocks".to_string(), blocks.into());

        // Extract function declarations
        let func_decls = PyDict::new(py);
        for (i, name) in callback_names.iter().enumerate() {
            let decl = PyDict::new(py);
            decl.set_item("name", name.clone()).unwrap();
            if i < func.dfg.signatures.len() {
                let sig = &func.dfg.signatures[mir::FuncRef::from(i as u32)];
                decl.set_item("num_args", sig.params).unwrap();
                decl.set_item("num_returns", sig.returns).unwrap();
            }
            func_decls.set_item(format!("inst{}", i), decl).unwrap();
        }
        result.insert("function_decls".to_string(), func_decls.into());

        result
    })
}

/// Compile a Verilog-A file and return module information
///
/// Args:
///     path: Path to the .va file
///     allow_analog_in_cond: Allow analog operators in conditionals (default: false)
///     allow_builtin_primitives: Allow built-in primitives (default: false)
#[pyfunction]
#[pyo3(signature = (path, allow_analog_in_cond=false, allow_builtin_primitives=false))]
fn compile_va(path: &str, allow_analog_in_cond: bool, allow_builtin_primitives: bool) -> PyResult<Vec<VaModule>> {
    let input = std::path::Path::new(path)
        .canonicalize()
        .map_err(|e| PyValueError::new_err(format!("Failed to resolve path: {}", e)))?;
    let input = AbsPathBuf::assert(input);

    let opts = CompilationOpts {
        allow_analog_in_cond,
        allow_builtin_primitives,
    };

    let db = CompilationDB::new_fs(input, &[], &[], &[], &opts)
        .map_err(|e| PyValueError::new_err(format!("Failed to create compilation DB: {}", e)))?;

    let modules = collect_modules(&db, false, &mut ConsoleSink::new(&db))
        .ok_or_else(|| PyValueError::new_err("Compilation failed with errors"))?;

    let mut literals = Rodeo::new();
    let mut result = Vec::new();

    for module_info in &modules {
        let compiled = CompiledModule::new(&db, module_info, &mut literals, false, false);

        // Extract eval parameter metadata
        let mut param_names = Vec::new();
        let mut param_kinds = Vec::new();
        let mut param_value_indices = Vec::new();

        for (kind, val) in compiled.intern.params.iter() {
            param_value_indices.push(u32::from(*val));
            let (kind_str, name) = extract_param_info(&db, kind);
            param_kinds.push(kind_str);
            param_names.push(name);
        }

        // Extract node names
        let mut nodes = Vec::new();
        for kind in compiled.dae_system.unknowns.iter() {
            nodes.push(format!("{:?}", kind));
        }

        // Extract residual indices
        let mut residual_resist_indices = Vec::new();
        let mut residual_react_indices = Vec::new();
        let mut residual_resist_lim_rhs_indices = Vec::new();
        let mut residual_react_lim_rhs_indices = Vec::new();
        for residual in compiled.dae_system.residual.iter() {
            residual_resist_indices.push(u32::from(residual.resist));
            residual_react_indices.push(u32::from(residual.react));
            residual_resist_lim_rhs_indices.push(u32::from(residual.resist_lim_rhs));
            residual_react_lim_rhs_indices.push(u32::from(residual.react_lim_rhs));
        }

        // Extract Jacobian structure
        let mut jacobian_resist_indices = Vec::new();
        let mut jacobian_react_indices = Vec::new();
        let mut jacobian_rows = Vec::new();
        let mut jacobian_cols = Vec::new();
        for entry in compiled.dae_system.jacobian.iter() {
            jacobian_rows.push(u32::from(entry.row));
            jacobian_cols.push(u32::from(entry.col));
            jacobian_resist_indices.push(u32::from(entry.resist));
            jacobian_react_indices.push(u32::from(entry.react));
        }

        // Count function parameters
        let mut max_param_idx: i32 = -1;
        for val in compiled.eval.dfg.values.iter() {
            if let mir::ValueDef::Param(p) = compiled.eval.dfg.value_def(val) {
                let p_idx: u32 = p.into();
                if p_idx as i32 > max_param_idx {
                    max_param_idx = p_idx as i32;
                }
            }
        }
        let func_num_params = if max_param_idx >= 0 { (max_param_idx + 1) as usize } else { 0 };

        // Collect callback names
        let callback_names: Vec<String> = compiled.intern.callbacks
            .iter()
            .map(|cb| format!("{:?}", cb))
            .collect();

        // === Init function metadata ===
        let mut init_param_names = Vec::new();
        let mut init_param_kinds = Vec::new();
        let mut init_param_value_indices = Vec::new();

        for (kind, val) in compiled.init.intern.params.iter() {
            init_param_value_indices.push(u32::from(*val));
            let (kind_str, name) = extract_param_info(&db, kind);
            init_param_kinds.push(kind_str);
            init_param_names.push(name);
        }

        let mut init_max_param_idx: i32 = -1;
        for val in compiled.init.func.dfg.values.iter() {
            if let mir::ValueDef::Param(p) = compiled.init.func.dfg.value_def(val) {
                let p_idx: u32 = p.into();
                if p_idx as i32 > init_max_param_idx {
                    init_max_param_idx = p_idx as i32;
                }
            }
        }
        let init_num_params = if init_max_param_idx >= 0 { (init_max_param_idx + 1) as usize } else { 0 };

        // Cache mapping
        let num_eval_named_params = compiled.intern.params.len();
        let cache_mapping: Vec<(u32, u32)> = compiled.init.cached_vals
            .iter()
            .map(|(&init_val, &cache_slot)| {
                let eval_param_idx = cache_slot.0 as usize + num_eval_named_params;
                (u32::from(init_val), eval_param_idx as u32)
            })
            .collect();

        // Collapsible pairs
        let collapsible_pairs: Vec<(u32, u32)> = compiled.node_collapse
            .pairs()
            .map(|(_, node1, node2_opt)| {
                let n1: u32 = node1.into();
                let n2: u32 = node2_opt.map(|n| n.into()).unwrap_or(u32::MAX);
                (n1, n2)
            })
            .collect();
        let num_collapsible = collapsible_pairs.len();

        // Collapse decision outputs
        let collapse_decision_outputs = extract_collapse_decisions(&compiled, &collapsible_pairs);

        // Parameter defaults
        let param_defaults = extract_param_defaults(&db, &compiled);

        // String constant values
        let mut str_constant_values: HashMap<String, String> = HashMap::new();
        for val in compiled.eval.dfg.values.iter() {
            if let mir::ValueDef::Const(mir::Const::Str(spur)) = compiled.eval.dfg.value_def(val) {
                let operand_name = format!("v{}", u32::from(val));
                let resolved_str = literals.resolve(&spur).to_owned();
                str_constant_values.insert(operand_name, resolved_str);
            }
        }

        // === OSDI metadata ===
        let num_terminals = module_info.module.ports(&db).len();

        let osdi_params = extract_osdi_params(&db, module_info);
        let osdi_nodes = extract_osdi_nodes(&db, &compiled, num_terminals);
        let osdi_jacobian = extract_osdi_jacobian(&compiled);
        let osdi_noise_sources = extract_osdi_noise(&compiled, &literals);

        let has_bound_step = compiled.intern.outputs.contains_key(&PlaceKind::BoundStep);
        let num_states = compiled.intern.lim_state.len();

        result.push(VaModule {
            name: module_info.module.name(&db).to_string(),
            param_names,
            param_kinds,
            param_value_indices,
            nodes,
            num_residuals: compiled.dae_system.residual.len(),
            num_jacobian: compiled.dae_system.jacobian.len(),
            residual_resist_indices,
            residual_react_indices,
            residual_resist_lim_rhs_indices,
            residual_react_lim_rhs_indices,
            jacobian_resist_indices,
            jacobian_react_indices,
            jacobian_rows,
            jacobian_cols,
            eval_func: compiled.eval.clone(),
            func_num_params,
            callback_names,
            init_func: compiled.init.func.clone(),
            init_param_names,
            init_param_kinds,
            init_param_value_indices,
            init_num_params,
            cache_mapping: cache_mapping.clone(),
            num_cached_values: cache_mapping.len(),
            collapsible_pairs,
            num_collapsible,
            collapse_decision_outputs,
            param_defaults,
            str_constant_values,
            num_terminals,
            osdi_params,
            osdi_nodes,
            osdi_jacobian,
            osdi_noise_sources,
            num_states,
            has_bound_step,
        });
    }

    Ok(result)
}

/// Extract parameter info (kind string and name) from ParamKind
fn extract_param_info(db: &CompilationDB, kind: &ParamKind) -> (String, String) {
    match kind {
        ParamKind::Param(param) => ("param".to_string(), param.name(db).to_string()),
        ParamKind::ParamGiven { param } => ("param_given".to_string(), param.name(db).to_string()),
        ParamKind::Voltage { hi, lo } => {
            let name = if let Some(lo) = lo {
                format!("V({},{})", hi.name(db), lo.name(db))
            } else {
                format!("V({})", hi.name(db))
            };
            ("voltage".to_string(), name)
        }
        ParamKind::Current(ck) => {
            let name = match ck {
                CurrentKind::Branch(br) => format!("I({})", br.name(db)),
                CurrentKind::Unnamed { hi, lo } => {
                    if let Some(lo) = lo {
                        format!("I({},{})", hi.name(db), lo.name(db))
                    } else {
                        format!("I({})", hi.name(db))
                    }
                }
                CurrentKind::Port(n) => format!("I({})", n.name(db)),
            };
            ("current".to_string(), name)
        }
        ParamKind::Temperature => ("temperature".to_string(), "$temperature".to_string()),
        ParamKind::Abstime => ("abstime".to_string(), "$abstime".to_string()),
        ParamKind::HiddenState(var) => ("hidden_state".to_string(), var.name(db).to_string()),
        ParamKind::PortConnected { port } => ("port_connected".to_string(), port.name(db).to_string()),
        ParamKind::ParamSysFun(param) => ("sysfun".to_string(), format!("{:?}", param)),
        _ => ("unknown".to_string(), "unknown".to_string()),
    }
}

/// Extract collapse decision outputs from init function
fn extract_collapse_decisions(compiled: &CompiledModule, collapsible_pairs: &[(u32, u32)]) -> Vec<(u32, String)> {
    use hir_lower::CallBackKind;

    let mut collapse_decision_outputs: Vec<(u32, String)> = Vec::new();

    // Build mapping from FuncRef to CollapseHint callback info
    let mut collapse_hint_funcs: HashMap<mir::FuncRef, usize> = HashMap::new();
    let mut collapse_hint_index = 0usize;
    for (idx, kind) in compiled.init.intern.callbacks.iter().enumerate() {
        if matches!(kind, CallBackKind::CollapseHint(_, _)) {
            let func_ref = mir::FuncRef::from(idx as u32);
            collapse_hint_funcs.insert(func_ref, collapse_hint_index);
            collapse_hint_index += 1;
        }
    }

    let init_func = &compiled.init.func;

    // Build map of block -> instructions containing calls to CollapseHint
    let mut callback_blocks: HashMap<mir::Block, Vec<(mir::FuncRef, usize)>> = HashMap::new();
    for block in init_func.layout.blocks() {
        for inst in init_func.layout.block_insts(block) {
            if let mir::InstructionData::Call { func_ref, .. } = init_func.dfg.insts[inst] {
                if let Some(&hint_idx) = collapse_hint_funcs.get(&func_ref) {
                    callback_blocks.entry(block).or_default().push((func_ref, hint_idx));
                }
            }
        }
    }

    // Find branches that target blocks with CollapseHint calls
    for block in init_func.layout.blocks() {
        for inst in init_func.layout.block_insts(block) {
            if let mir::InstructionData::Branch { cond, then_dst, else_dst, .. } = init_func.dfg.insts[inst] {
                for (target_block, is_true_branch) in [(then_dst, true), (else_dst, false)] {
                    if let Some(callbacks) = callback_blocks.get(&target_block) {
                        for &(_func_ref, hint_idx) in callbacks {
                            if hint_idx < collapsible_pairs.len() {
                                let pair_idx = hint_idx as u32;
                                let cond_idx: u32 = cond.into();
                                let prefix = if is_true_branch { "" } else { "!" };
                                collapse_decision_outputs.push((pair_idx, format!("{}v{}", prefix, cond_idx)));
                            }
                        }
                    }
                }
            }
        }
    }

    collapse_decision_outputs.sort_by_key(|(idx, _)| *idx);
    collapse_decision_outputs
}

/// Extract parameter defaults from HIR
fn extract_param_defaults(db: &CompilationDB, compiled: &CompiledModule) -> HashMap<String, f64> {
    let mut param_defaults = HashMap::new();

    for (kind, _) in compiled.intern.params.iter() {
        if let ParamKind::Param(param) = kind {
            let param_name = param.name(db).to_lowercase();
            let body = param.init(db);
            let body_ref = body.borrow();

            if !body_ref.entry().is_empty() {
                let expr_id = body_ref.get_entry_expr(0);

                if let Some(lit) = body_ref.as_literal(expr_id) {
                    match lit {
                        Literal::Float(ieee64) => {
                            param_defaults.insert(param_name.clone(), f64::from(*ieee64));
                        }
                        Literal::Int(i) => {
                            param_defaults.insert(param_name.clone(), *i as f64);
                        }
                        Literal::Inf => {
                            param_defaults.insert(param_name.clone(), f64::INFINITY);
                        }
                        _ => {}
                    }
                } else {
                    let expr = body_ref.get_expr(expr_id);
                    if let Expr::UnaryOp { expr: inner_expr, op: UnaryOp::Neg } = expr {
                        if let Some(lit) = body_ref.as_literal(inner_expr) {
                            match lit {
                                Literal::Float(ieee64) => {
                                    param_defaults.insert(param_name.clone(), -f64::from(*ieee64));
                                }
                                Literal::Int(i) => {
                                    param_defaults.insert(param_name.clone(), -(*i as f64));
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
    }

    param_defaults
}

/// Extract OSDI parameter metadata
fn extract_osdi_params(db: &CompilationDB, module_info: &sim_back::ModuleInfo) -> Vec<OsdiParamInfo> {
    let mut osdi_params = Vec::new();

    for (param, param_info) in module_info.params.iter() {
        let ty = param.ty(db);
        let type_flag = match ty.base_type() {
            Type::Real => PARA_TY_REAL,
            Type::Integer => PARA_TY_INT,
            Type::String => PARA_TY_STR,
            _ => PARA_TY_REAL,
        };
        let kind_flag = if param_info.is_instance { PARA_KIND_INST } else { PARA_KIND_MODEL };

        osdi_params.push(OsdiParamInfo {
            name: param_info.name.to_string(),
            aliases: param_info.alias.iter().map(|s| s.to_string()).collect(),
            units: param_info.unit.clone(),
            description: param_info.description.clone(),
            flags: type_flag | kind_flag,
            is_instance: param_info.is_instance,
        });
    }

    osdi_params
}

/// Extract OSDI node metadata
fn extract_osdi_nodes(db: &CompilationDB, compiled: &CompiledModule, num_terminals: usize) -> Vec<OsdiNodeInfo> {
    let mut osdi_nodes = Vec::new();

    for (idx, unknown_kind) in compiled.dae_system.unknowns.iter_enumerated() {
        let is_internal = u32::from(idx) >= num_terminals as u32;
        let (clean_name, kind_str) = match unknown_kind {
            sim_back::SimUnknownKind::KirchoffLaw(node) => {
                (node.name(db).to_string(), "KirchoffLaw")
            }
            sim_back::SimUnknownKind::Current(ck) => {
                let name = match ck {
                    CurrentKind::Branch(br) => format!("flow({})", br.name(db)),
                    CurrentKind::Unnamed { hi, lo } => {
                        if let Some(lo) = lo {
                            format!("flow({},{})", hi.name(db), lo.name(db))
                        } else {
                            format!("flow({})", hi.name(db))
                        }
                    }
                    CurrentKind::Port(node) => format!("flow(<{}>)", node.name(db)),
                };
                (name, "BranchCurrent")
            }
            sim_back::SimUnknownKind::Implicit(eq) => {
                (format!("implicit_equation_{}", u32::from(*eq)), "Implicit")
            }
        };

        osdi_nodes.push(OsdiNodeInfo {
            name: clean_name,
            kind: kind_str.to_string(),
            units: "V".to_string(),
            residual_units: "A".to_string(),
            is_internal,
        });
    }

    osdi_nodes
}

/// Extract OSDI Jacobian entry metadata
fn extract_osdi_jacobian(compiled: &CompiledModule) -> Vec<OsdiJacobianInfo> {
    let is_entry_const = |entry_val: mir::Value, func: &mir::Function| -> bool {
        match func.dfg.value_def(entry_val) {
            ValueDef::Const(_) => true,
            ValueDef::Param(param) => {
                compiled.intern.params.get_index(param)
                    .map_or(true, |(kind, _)| !kind.op_dependent())
            }
            _ => false,
        }
    };

    let mut osdi_jacobian = Vec::new();

    for entry in compiled.dae_system.jacobian.iter() {
        let mut flags: u32 = 0;

        if entry.resist != F_ZERO {
            flags |= JACOBIAN_ENTRY_RESIST;
        }
        if is_entry_const(entry.resist, &compiled.eval) {
            flags |= JACOBIAN_ENTRY_RESIST_CONST;
        }
        if entry.react != F_ZERO {
            flags |= JACOBIAN_ENTRY_REACT;
        }
        if is_entry_const(entry.react, &compiled.eval) {
            flags |= JACOBIAN_ENTRY_REACT_CONST;
        }

        osdi_jacobian.push(OsdiJacobianInfo {
            row: u32::from(entry.row),
            col: u32::from(entry.col),
            flags,
        });
    }

    osdi_jacobian
}

/// Extract OSDI noise source metadata
fn extract_osdi_noise(compiled: &CompiledModule, literals: &Rodeo) -> Vec<OsdiNoiseInfo> {
    compiled.dae_system.noise_sources
        .iter()
        .map(|src| {
            let name = literals.resolve(&src.name).to_owned();
            OsdiNoiseInfo {
                name,
                node1: u32::from(src.hi),
                node2: src.lo.map_or(u32::MAX, |lo| u32::from(lo)),
            }
        })
        .collect()
}

/// Python module definition
#[pymodule]
fn openvaf_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile_va, m)?)?;
    m.add_class::<VaModule>()?;
    Ok(())
}
