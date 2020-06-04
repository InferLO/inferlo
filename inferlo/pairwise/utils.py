from functools import lru_cache

import numpy as np


# Common utilities for Potts base.


def decode_state(state_id, vert_num, al_size):
    return [(state_id // (al_size ** (vert_num - j - 1))) % al_size for j in
            range(vert_num)]


def encode_state(state, vert_num, al_size):
    return sum(
        [state[j] * (al_size ** (vert_num - j - 1)) for j in range(vert_num)])


@lru_cache(maxsize=None)
def decode_all_states(vert_num, al_size):
    """Enumerates all possible states.

    Returns np.array of shape(states_count, vert_num).
    """
    n = al_size ** vert_num
    return np.array([decode_state(i, vert_num, al_size) for i in range(n)])


@lru_cache(maxsize=None)
def get_marginal_states(vert_num, al_size):
    """Return states to sum over to calculate marginal probabilities.

    Returns tensor swv[vert_num, al_size, al_size**(vert_num-1)], such that
    swv[v][c] contains state codes representing states s, for which s[v] = c.
    """
    ans = [[[] for _ in range(al_size)] for _ in range(vert_num)]
    for state_id in range(al_size ** vert_num):
        state = decode_state(state_id, vert_num, al_size)
        for v in range(vert_num):
            ans[v][state[v]].append(state_id)
    return np.array(ans, dtype=np.int32)
