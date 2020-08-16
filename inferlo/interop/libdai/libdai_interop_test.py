# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import os

import numpy as np

from inferlo import PairWiseFiniteModel, GenericGraphModel, DiscreteDomain
from inferlo.interop import LibDaiInterop
from inferlo.testing import tree_potts_model, assert_results_close


# Warning! If LibDAI is not installed, all tests will silently pass without
# testing anything. This is done to avoid installing LibDAI on remote machine
# and also because it works only on Linux.

def test_to_fg_factor():
    model = PairWiseFiniteModel(3, 2)
    x0, x1, x2 = model.get_symbolic_variables()
    model *= (x0 + x1)
    model *= (2 * x1 + x2)

    tmp_file = os.path.join(LibDaiInterop().tmp_path, 'tmp.fg')
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
    libdai_result = libdai.infer(model, "BP")
    assert_results_close(true_result, libdai_result, log_pf_tol=1e-8)


def test_max_likelihood_potts_tree_1000x5():
    libdai = LibDaiInterop()
    if not libdai.is_libdai_ready():
        return
    model = tree_potts_model(1000, 5)
    true_ml = model.max_likelihood()
    libdai_ml = libdai.max_likelihood(model, "BP")
    assert np.allclose(true_ml, libdai_ml)


def test_arbitrary_model():
    # LibDAI can handle models with different domains and non-binary models.
    libdai = LibDaiInterop()
    if not libdai.is_libdai_ready():
        return
    model = GenericGraphModel(5)
    model[0].domain = DiscreteDomain.binary()
    model[1].domain = DiscreteDomain.range(3)
    model[2].domain = DiscreteDomain.range(4)
    model[3].domain = DiscreteDomain.range(5)
    model[4].domain = DiscreteDomain([1.2, 3.4, 5.6])

    x = model.get_symbolic_variables()
    model *= np.abs(np.sin(x[0] + x[1] + x[4]))
    model *= (x[1] + x[2] * x[3])
    model *= 10 * x[2] + x[3] + x[4] ** 2

    true_log_pf = np.log(model.part_func_bruteforce())
    libdai_log_pf = libdai.infer(model, "EXACT").log_pf
    assert np.allclose(true_log_pf, libdai_log_pf)
