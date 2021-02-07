from inferlo.testing import ExperimentRunner


def test_run_experiment():
    def my_experiment(x=0):
        return {"square": x * x, "cube": x * x * x}

    runner = ExperimentRunner()

    result = runner.run_experiment(my_experiment, {'x': 2})
    assert result['square'] == 4
