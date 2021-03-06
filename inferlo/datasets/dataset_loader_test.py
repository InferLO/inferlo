# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import numpy as np

import inferlo
from inferlo import GenericGraphModel
from inferlo.datasets import Dataset
from inferlo.testing import grid_potts_model

dataset_loader = inferlo.datasets.DatasetLoader()


def test_load_promedus():
    dataset = dataset_loader.load_uai_dataset('Promedus_20.uai')
    assert dataset.model.num_variables == 546
    assert len(dataset.model.factors) == 546
    assert dataset.model.factors[2].var_idx == [80, 544]
    assert np.allclose(dataset.model.factors[2].values,
                       np.array([[0.97, 0], [0.03, 1]]))
    assert len(dataset.true_marginals) == 546
    assert np.allclose(dataset.true_marginals[0],
                       np.array([0.998495, 0.00150482]))
    assert np.allclose(dataset.true_log_pf, -16.26, atol=1e-2)


def test_load_linkage():
    dataset = dataset_loader.load_uai_dataset('linkage_20.uai')
    assert dataset.model.num_variables == 1160
    assert len(dataset.model.factors) == 1160
    assert dataset.model.factors[1].var_idx == [5, 1]
    assert np.allclose(dataset.model.factors[1].values,
                       np.array([[0.9, 0.1], [0.1, 0.9]]))
    assert len(dataset.true_marginals) == 1160
    assert np.allclose(dataset.true_marginals[0], np.array([1, 0, 0, 0, 0]))
    assert np.allclose(dataset.true_marginals[1],
                       np.array([0.505242, 0.494758, 0, 0, 0]))
    assert np.allclose(dataset.true_log_pf, -147.89, atol=1e-2)


def test_custom_dataset():
    model = grid_potts_model(4, 3, al_size=4, seed=1)
    true_result = model.infer(algorithm='path_dp')
    model = GenericGraphModel.from_model(model)
    dataset = Dataset(model=model, true_log_pf=true_result.log_pf,
                      true_marginals=true_result.marg_prob, name='test_dataset')
    dataset_loader.save_custom_dataset(dataset)
    reloaded_dataset = dataset_loader.load_custom_dataset('test_dataset')
    assert reloaded_dataset.model.num_variables == model.num_variables
    assert reloaded_dataset.model.factors[1].var_idx == model.factors[1].var_idx
    assert np.allclose(reloaded_dataset.model.factors[1].values, model.factors[1].values)
    assert np.allclose(reloaded_dataset.true_log_pf, dataset.true_log_pf)
    assert np.allclose(reloaded_dataset.true_marginals, dataset.true_marginals)
    assert reloaded_dataset.name == 'test_dataset'
