import inferlo
from inferlo.generic.inference import BackwardBucketElimination
from inferlo.testing import assert_results_close, grid_potts_model


def test_potts_grid():
    model = grid_potts_model(6, 5, al_size=3)
    true_result = model.infer(algo='parth_dp')
    model = inferlo.GenericGraphModel.from_model(model)
    bw_result = BackwardBucketElimination.infer(model)
    assert_results_close(true_result, bw_result)
