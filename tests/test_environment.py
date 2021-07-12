import sys

import numpy as np
import pandas as pd

sys.path.append("..")
from env.myenv import MyEnv
import pytest


class TestMyEnv(MyEnv):
    pass


testMyEnv = TestMyEnv()
binary_dir = "../Data/CSV/Binary"
expert_df = pd.read_csv(f"{binary_dir}/BMV772.csv")


@pytest.mark.parametrize(
    "action",
    [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ]
)
def test_action(action):
    env = TestMyEnv()
    env.send_expert_data(expert_df)
    env.reset()
    state, reward, done, _ = env.step(action)

    if action == 0:
        assert (state == np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)).all
        assert reward == 10
        assert done == True
    elif action == 1:
        assert (state == np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)).all
        assert reward == 10
        assert done == True
    elif action == 2:
        assert (state == np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)).all
        assert reward == 10
        assert done == True
    elif action == 3:
        assert (state == np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)).all
        assert reward == -0.1
        assert done == False
    elif action == 4:
        assert (state == np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)).all
        assert reward == 10
        assert done == True
    elif action == 5:
        assert (state == np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8)).all
        assert reward == 10
        assert done == True
    elif action == 6:
        assert (state == np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.uint8)).all
        assert reward == 10
        assert done == True
    elif action == 7:
        assert (state == np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint8)).all
        assert reward == 10
        assert done == True
    elif action == 8:
        assert (state == np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.uint8)).all
        assert reward == 10
        assert done == True
    elif action == 9:
        assert (state == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.uint8)).all
        assert reward == 10
        assert done == True
    elif action == 10:
        assert (state == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.uint8)).all
        assert reward == 10
        assert done == True
    elif action == 11:
        assert (state == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.uint8)).all
        assert reward == 10
        assert done == True
    elif action == 12:
        assert (state == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)).all
        assert reward == -0.1
        assert done == False
