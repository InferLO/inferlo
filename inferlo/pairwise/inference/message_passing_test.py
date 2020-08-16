# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
import random

import networkx
import numpy as np

from inferlo import PairWiseFiniteModel
from inferlo.testing import (tree_potts_model, assert_results_close,
                             grid_potts_model)


def test_tree_exact():
    model = tree_potts_model(gr_size=50, al_size=3, seed=0)
    assert_results_close(model.infer(algorithm='message_passing'),
                         model.infer(algorithm='tree_dp'))


def test_grid_approx():
    model = grid_potts_model(5, 15, al_size=2, seed=0)
    mp_true = model.infer(algorithm='path_dp').marg_prob
    mp = model.infer(algorithm='message_passing').marg_prob

    assert np.mean(np.square(mp_true - mp)) < 1e-4


def test_small_forest():
    edges = [[0, 1], [1, 2], [1, 3], [4, 5], [6, 7]]
    field = np.random.random((8, 2))
    inter = np.random.random((5, 2, 2))
    model = PairWiseFiniteModel.create(field, edges, inter)
    assert_results_close(model.infer(algorithm='message_passing'),
                         model.infer(algorithm='bruteforce'))


def test_random_forest_compare_with_tree_dp():
    # Generate tree on 50 edges, then leave only 40 random edges.
    gr_size, al_size = 50, 5
    edges = list(networkx.random_tree(gr_size).edges())
    random.shuffle(edges)
    edges = edges[0:40]
    model = PairWiseFiniteModel(gr_size, 5)
    model.set_field(np.random.random((gr_size, al_size)))
    for v1, v2 in edges:
        model.add_interaction(v1, v2, np.random.random((al_size, al_size)))

    assert_results_close(model.infer(algorithm='message_passing'),
                         model.infer(algorithm='tree_dp'))
