import numpy as np

from inferlo.pairwise.testing import tree_potts_model, line_potts_model


def test_empirical_probabilities():
    # In this test case we sample 10000 from randomly generated model and use
    # samples to calculate empirical probabilities for every configuration.
    # Then we explicitly evaluate theoretical probabilities of every state and
    # assert that they are close to empirical probabilities.
    model = tree_potts_model(gr_size=5, al_size=2)
    num_samples = 10000
    samples = model.sample(num_samples=num_samples, algorithm='tree_dp')
    assert samples.shape == (num_samples, model.gr_size)

    # Calculate empirical probabilities of states by counting how many
    # times each state appeared in samples.
    emp_proba = np.zeros(model.al_size ** model.gr_size)
    for sample in samples:
        state_id = model.encode_state(sample)
        emp_proba[state_id] += 1
    emp_proba /= num_samples

    # Calculate true probabilities of states by definition.
    true_proba = [model.evaluate(np.array(model.decode_state(i))) for i in
                  range(model.al_size ** model.gr_size)]
    true_proba /= np.sum(true_proba)

    assert np.max(true_proba - emp_proba) < 0.01


def test_one_very_likely_state():
    # In this test case we generate model on tree and then set very large
    # fields such that they force values in all nodes to be equal to given
    # value with probability very close to 1. Then we assert that this
    # configuration was sampled at least 99% of times.
    gr_size = 100
    al_size = 5
    model = tree_potts_model(gr_size=gr_size, al_size=al_size)
    expected_configuration = np.random.choice(al_size, size=gr_size)
    field = np.zeros((gr_size, al_size))
    for i in range(gr_size):
        field[i][expected_configuration[i]] = 100
    model.set_field(field)

    samples = model.sample(num_samples=100, algorithm='tree_dp')
    good_count = sum(
        [np.all(sample == expected_configuration) for sample in samples])
    assert good_count >= 99


def test_antiferromagnetic_ising_line():
    # We create Ising model on a line with very high interactions forcing
    # samples to be alternate (i.e. 10101010). Then we assert that in at
    # least 99% of cases we got alternating sample.
    model = line_potts_model(gr_size=100, al_size=2,
                             same_j=[[0, 100], [100, 0]], zero_field=True)
    samples = model.sample(num_samples=100)

    def is_alternating(x): return np.all(np.roll(x, 1) == 1 - x)

    good_count = sum([is_alternating(state) for state in samples])
    assert good_count >= 99
