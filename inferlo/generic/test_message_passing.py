# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import numpy as np

import inferlo
from inferlo.generic.message_passing import infer_generic_message_passing
from inferlo.pairwise.testing import tree_potts_model, assert_results_close


def test_non_pairwise():
    model = inferlo.GenericGraphModel(6, inferlo.DiscreteDomain.range(2))
    model[2].domain = inferlo.DiscreteDomain.range(3)
    model[3].domain = inferlo.DiscreteDomain.range(4)
    x = model.get_symbolic_variables()
    model *= (x[0] + x[1] + x[2])
    model *= (x[2] + x[3] + x[4] + x[5])

    true_log_pf = np.log(model.part_func_bruteforce())
    assert np.allclose(infer_generic_message_passing(model).log_pf,
                       true_log_pf)


def test_pairwise_tree():
    model = tree_potts_model(gr_size=60, al_size=5)
    true_result = model.infer(algorithm='tree_dp')
    result = infer_generic_message_passing(model)
    assert_results_close(result, true_result)
