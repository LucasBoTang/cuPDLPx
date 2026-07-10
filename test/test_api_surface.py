# Copyright 2025 Haihao Lu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import scipy.sparse as sp
import pytest

from cupdlpx import Model, PDLP, read


def _model(base_lp_data):
    c, A, l, u, lb, ub = base_lp_data
    model = Model(c, A, l, u, lb, ub)
    model.setParams(OutputFlag=False, Presolve=False)
    return model


def test_data_property_reads(base_lp_data):
    """Read-side of the model-data properties."""
    model = _model(base_lp_data)
    assert model.c.shape == (2,)
    assert model.c0 == 0.0
    assert model.A.shape == (3, 2)
    assert model.constr_lb.shape == (3,)
    assert model.constr_ub.shape == (3,)
    assert model.lb is None
    assert model.ub is None
    assert model.ModelSense == PDLP.MINIMIZE


def test_data_property_writes(base_lp_data):
    """Direct assignment routes through the validating setters."""
    model = _model(base_lp_data)
    model.c = [2.0, 3.0]
    assert np.allclose(model.c, [2.0, 3.0])
    model.c0 = 4.0
    assert model.c0 == 4.0
    model.lb = [0.0, 0.0]
    assert np.allclose(model.lb, [0.0, 0.0])
    model.ub = [10.0, 10.0]
    assert np.allclose(model.ub, [10.0, 10.0])
    model.constr_lb = [1.0, 2.0, 3.0]
    assert np.allclose(model.constr_lb, [1.0, 2.0, 3.0])
    model.constr_ub = [4.0, 5.0, 6.0]
    assert np.allclose(model.constr_ub, [4.0, 5.0, 6.0])
    model.A = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    assert model.A.shape == (3, 2)
    model.ModelSense = PDLP.MAXIMIZE
    assert model.ModelSense == PDLP.MAXIMIZE


def test_bounds_none_clears(base_lp_data):
    """Passing None clears optional bounds."""
    model = _model(base_lp_data)
    model.lb = [0.0, 0.0]
    model.lb = None
    assert model.lb is None
    model.ub = None
    assert model.ub is None
    model.constr_lb = None
    assert model.constr_lb is None
    model.constr_ub = None
    assert model.constr_ub is None


def test_property_validation_errors(base_lp_data):
    """Invalid assignments raise immediately instead of failing in the solver."""
    model = _model(base_lp_data)
    with pytest.raises(ValueError):
        model.c = [1.0, 2.0, 3.0]          # wrong length
    with pytest.raises(ValueError):
        model.c = [[1.0, 2.0]]             # not 1D
    with pytest.raises(TypeError):
        model.A = "not a matrix"
    with pytest.raises(ValueError):
        model.A = np.ones(3)               # not 2D
    with pytest.raises(ValueError):
        model.A = np.ones((3, 5))          # wrong number of variables
    with pytest.raises(ValueError):
        model.lb = [0.0]                   # wrong length
    with pytest.raises(ValueError):
        model.ub = [0.0]
    with pytest.raises(ValueError):
        model.constr_lb = [0.0, 0.0]       # need 3
    with pytest.raises(ValueError):
        model.constr_ub = [0.0, 0.0]
    with pytest.raises(ValueError):
        model.ModelSense = 99


def test_invalid_objective_assignment_preserves_previous_value(base_lp_data):
    model = _model(base_lp_data)
    original = model.c.copy()
    with pytest.raises(ValueError):
        model.c = [1.0, 2.0, 3.0]
    assert np.allclose(model.c, original)


def test_dense_inputs_are_copied(base_lp_data):
    c, A, l, u, lb, ub = base_lp_data
    c = c.copy()
    A = A.copy()
    l = l.copy()
    u = u.copy()
    model = Model(c, A, l, u, lb, ub)
    c[0] = 99.0
    A[0, 0] = 99.0
    l[0] = 99.0
    u[0] = 99.0
    assert np.allclose(model.c, [1.0, 1.0])
    assert model.A[0, 0] == 1.0
    assert model.constr_lb[0] == 5.0
    assert model.constr_ub[0] == 5.0


def test_bound_order_validation(base_lp_data):
    model = _model(base_lp_data)
    model.lb = [2.0, 0.0]
    model.ub = [1.0, 1.0]
    with pytest.raises(ValueError):
        model.optimize()

    with pytest.raises(ValueError):
        Model(
            [1.0],
            np.array([[1.0]]),
            constraint_lower_bound=[2.0],
            constraint_upper_bound=[1.0],
        )


def test_set_params_is_transactional(base_lp_data):
    model = _model(base_lp_data)
    old_time_limit = model.getParam("TimeLimit")
    with pytest.raises(KeyError):
        model.setParams(TimeLimit=123.0, DefinitelyNotAParam=1)
    assert model.getParam("TimeLimit") == old_time_limit


def test_matrix_row_change_conflicts_with_bounds(base_lp_data):
    """Reassigning A with a different row count than existing bounds raises."""
    model = _model(base_lp_data)
    # existing constraint bounds have 3 rows; a 2-row matrix must be rejected
    # (constraint_lower_bound check)
    two_row = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        model.A = two_row
    # now clear the lower bound so the upper-bound check is the one that fires
    model.constr_lb = None
    with pytest.raises(ValueError):
        model.A = two_row


def test_params_view(base_lp_data):
    """_ParamsView get/set via attribute and item access, plus keys()."""
    model = _model(base_lp_data)
    model.Params.TimeLimit = 123.0
    assert model.Params.TimeLimit == 123.0
    model.Params["OptimalityTol"] = 1e-5
    assert model.Params["OptimalityTol"] == 1e-5
    assert model.getParam("TimeLimit") == 123.0
    assert len(list(model.Params.keys())) > 0
    with pytest.raises(AttributeError):
        _ = model.Params.NoSuchParameter


def test_param_name_validation(base_lp_data):
    """Unknown parameter names are rejected in both set and get."""
    model = _model(base_lp_data)
    with pytest.raises(KeyError):
        model.setParam("DefinitelyNotAParam", 1)
    with pytest.raises(KeyError):
        model.getParam("DefinitelyNotAParam")


def test_non_contiguous_and_typed_inputs(base_lp_data):
    """Helper conversions handle non-contiguous dense and non-float64 sparse."""
    _, _, l, u, _, _ = base_lp_data
    # non-C-contiguous objective vector (a strided column view)
    c_nc = np.ones((2, 4))[:, 1]
    assert not c_nc.flags["C_CONTIGUOUS"]
    # integer-typed sparse matrix (exercises the float64 upcast path)
    A = sp.csr_matrix(np.array([[1, 2], [0, 1], [3, 2]], dtype=np.int64))
    model = Model(c_nc, A, l, u)
    assert model.c.dtype == np.float64
    assert model.A.dtype == np.float64
    assert model.A.indices.dtype == np.int32


def test_int64_index_downcast(base_lp_data):
    """A float64 CSR with int64 indices is downcast to int32 without mutating it."""
    _, _, l, u, _, _ = base_lp_data
    c = np.array([1.0, 1.0])
    A = sp.csr_matrix(np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 2.0]]))
    A.indices = A.indices.astype(np.int64)
    A.indptr = A.indptr.astype(np.int64)
    model = Model(c, A, l, u)
    assert model.A.indices.dtype == np.int32
    assert model.A.indptr.dtype == np.int32
    # caller's matrix must be untouched
    assert A.indices.dtype == np.int64
    assert A.indptr.dtype == np.int64


def test_init_requires_2d_matrix():
    with pytest.raises(ValueError):
        Model(np.ones(2), np.ones(3), None, None)  # 1D "matrix"


def test_solution_attributes_accessible(base_lp_data):
    """Every result attribute is readable after optimize()."""
    model = _model(base_lp_data)
    model.optimize()
    # touching each property exercises its getter
    attrs = [
        model.X, model.Pi, model.RC, model.ObjVal, model.DualObj,
        model.Gap, model.RelGap, model.Status, model.StatusName,
        model.IterCount, model.Runtime, model.RescalingTime,
        model.RelPrimalResidual, model.RelDualResidual,
        model.MaxPrimalRayInfeas, model.MaxDualRayInfeas,
        model.PrimalRayLinObj, model.DualRayObj,
        model.PrimalInfeas, model.DualInfeas,
    ]
    assert model.Status == PDLP.OPTIMAL
    assert model.StatusName == "OPTIMAL"
    assert len(attrs) == 20


def test_optimize_rejects_bad_sense(base_lp_data):
    """The defensive sense check inside optimize() rejects a corrupted sense."""
    model = _model(base_lp_data)
    model._model_sense = 999  # bypass the property to hit optimize()'s guard
    with pytest.raises(ValueError):
        model.optimize()


def test_read_mps_roundtrip():
    """read() loads an MPS file into a usable Model."""
    path = os.path.join(os.path.dirname(__file__), "cplex2.mps")
    model = read(path)
    assert model.num_vars > 0
    assert model.num_constrs > 0
    assert model.A.shape == (model.num_constrs, model.num_vars)
    assert model.ModelSense in (PDLP.MINIMIZE, PDLP.MAXIMIZE)


def test_read_mps_missing_file():
    with pytest.raises(FileNotFoundError):
        read("this_file_does_not_exist.mps")
