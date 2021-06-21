import sys

import pandas as pd
import pytest

sys.path.append("..")
from toolbox import toolbox


class TestToolbox(toolbox.Toolbox):
    pass


testToolbox = TestToolbox()
midi_dir = "../Data/Midi"
note_dir = "../Data/CSV/Note"


@pytest.mark.note
def test_get_track_data():
    assert testToolbox.get_track_data(f"{midi_dir}/BMV772.mid").iloc[0, 0] == "note_on"


@pytest.mark.note
def test_generate_stave():
    test_df_BMV772 = pd.read_csv(f"{note_dir}/BMV772.csv")

    assert testToolbox.generate_stave(test_df_BMV772).iloc[0, 0] == 0


@pytest.mark.note
def test_generate_str_notation():
    _note = [60]
    _duration = [0.1]
    _channel = [0]
    _test_str_notation = """player.note_on(note=60, velocity=127, channel=0)
time.sleep(0.1)
player.note_off(note=60, velocity=64, channel=0)
"""

    assert testToolbox.generate_str_notation(_note, _duration, _channel) == _test_str_notation


@pytest.mark.parametrize(
    "num",
    [
        0,
        19,
        21,
        37,
        50,
        63,
        76,
        89,
        103,
        108
    ]
)
def test_convert_num2str(num):
    test_str_chord = testToolbox.convert_num2str(num)

    if num == 0:
        assert test_str_chord is None
    elif num == 19:
        assert test_str_chord is None
    elif num == 21:
        assert test_str_chord == "A_M"
    elif num == 37:
        assert test_str_chord == "Db_M"
    elif num == 50:
        assert test_str_chord == "D_M"
    elif num == 63:
        assert test_str_chord == "Eb_M"
    elif num == 76:
        assert test_str_chord == "E_M"
    elif num == 89:
        assert test_str_chord == "F_M"
    elif num == 103:
        assert test_str_chord == "G_M"
    elif num == 108:
        assert test_str_chord == "C_M"
    else:
        raise ValueError


@pytest.mark.group
def test_generate_group_data():
    test_df_BMV772 = pd.read_csv(f"{note_dir}/BMV772.csv")

    assert testToolbox.generate_group_data(test_df_BMV772).iloc[0].tolist() == ["RLRL", "T2", 0]
