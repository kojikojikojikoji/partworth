import numpy as np

from partworth.datasets import simulate_cbc


def test_simulate_shapes_and_truth():
    data, truth = simulate_cbc(50, 8, 3, {"brand": ["x", "y", "z"], "price": [1, 2, 3]},
                               price="price", seed=1)
    assert data.X.shape == (50, 8, 4, 4)
    assert truth["beta"].shape == (50, 4)
    assert (truth["beta"][:, 2] < 0).all()
    assert data.codebook.sign_constraints() == {2: "negative"}
    # every task has exactly one choice, and choices are spread over positions
    assert np.bincount(data.y.ravel(), minlength=4).min() > 0
