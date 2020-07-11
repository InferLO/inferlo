import os

import numpy as np

from inferlo import PairWiseFiniteModel
from inferlo.interop import LibDaiInterop
from inferlo.pairwise.testing import tree_potts_model, assert_results_close


# Warning! If LibDAI is not installed, all tests will silently pass without
# testing anything. This is done to avoid installing LibDAI on remote machine
# and also because it works only on Linux.

def test_to_fg_factor():
    model = PairWiseFiniteModel(3, 2)
    x0, x1, x2 = model.get_symbolic_variables()
    model *= (x0 + x1)
    model *= (2 * x1 + x2)

    tmp_file = os.path.join(LibDaiInterop().tmp_path, 'tmp.fg`')
    LibDaiInterop.write_fg_file(model, tmp_file)
    with open(tmp_file, 'r') as f:
        lines = [x.strip() for x in f.readlines()]
    assert lines == [
        '2', '', '2', '0 1', '2 2', '3',
        '1 1.0000000000', '2 1.0000000000', '3 2.0000000000',
        '', '2', '1 2', '2 2', '3',
        '1 2.0000000000', '2 1.0000000000', '3 3.0000000000', '']


def test_marg_probs_potts_tree_1000x5():
    libdai = LibDaiInterop()
    if not libdai.is_libdai_ready():
        return
    model = tree_potts_model(1000, 5)
    true_result = model.infer()
    libdai_result = libdai.infer(model)
    assert_results_close(true_result, libdai_result, log_pf_tol=1e-8)


def test_max_likelihood_potts_tree_1000x5():
    libdai = LibDaiInterop()
    if not libdai.is_libdai_ready():
        return
    model = tree_potts_model(1000, 5)
    true_ml = model.max_likelihood()
    libdai_ml = libdai.max_likelihood(model)
    assert np.allclose(true_ml, libdai_ml)