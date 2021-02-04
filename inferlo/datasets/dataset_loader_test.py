# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import numpy as np
import inferlo


def test_load_promedus():
    dataset = inferlo.datasets.promedus(20)
    assert dataset.model.num_variables == 546
    assert len(dataset.model.factors) == 546
    assert dataset.model.factors[2].var_idx == [80, 544]
    assert np.allclose(dataset.model.factors[2].values,
                       np.array([[0.97, 0], [0.03, 1]]))
    assert len(dataset.true_marginals) == 546
    assert np.allclose(dataset.true_marginals[0],
                       np.array([0.998495, 0.00150482]))
    assert np.allclose(dataset.true_log_z, -7.06065)


def test_load_linkage():
    dataset = inferlo.datasets.linkage(20)
    assert dataset.model.num_variables == 1160
    assert len(dataset.model.factors) == 1160
    assert dataset.model.factors[1].var_idx == [5, 1]
    assert np.allclose(dataset.model.factors[1].values,
                       np.array([[0.9, 0.1], [0.1, 0.9]]))
    assert len(dataset.true_marginals) == 1160
    assert np.allclose(dataset.true_marginals[0], np.array([1, 0, 0, 0, 0]))
    assert np.allclose(dataset.true_marginals[1],
                       np.array([0.505242, 0.494758, 0, 0, 0]))
    assert np.allclose(dataset.true_log_z, -64.2292)
