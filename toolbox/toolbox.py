import os
import time

import numpy as np
import pandas as pd
import pygame.midi
import requests
from mido import MidiFile
from opycleid.musicmonoids import PRL_Group
from opycleid.musicmonoids import TI_Group_Triads
from tqdm import tqdm


class Toolbox:
    """
    1. tempo: Default value 500000, which means that the duration of a beat is 500000 microseconds, or 0.5 seconds,
        or 120 beats per minute.

    2. beat: A quarter note.

    3. tick: The smallest unit of time, representing the number of parts to divide a beat into.

    4. time: The time attribute of each message. The unit is tick,
        indicating the time distance from the previous message.

    5. time signature: Corresponds to the beat number on the pentatonic scale.

    6. note: C4 is 60.
    """

    def get_mid(self, rootdir: str, convert_dict: dict) -> None:
        """
        Download the midi files
        :param rootdir: The root directory of Midi files
        :param convert_dict: Turning artwork numbers into urls
        :return: None
        """

        # Check if there are folders of rootdir in the environment
        if not os.path.exists(rootdir):
            os.makedirs(rootdir)

        for num, url in tqdm(convert_dict.items()):
            r = requests.get(url, timeout=200)
            with open(f"{rootdir}/{num}.mid", "wb") as f:
                f.write(r.content)
            time.sleep(2)

    def get_track_data(self, filename: str) -> pd.DataFrame:
        """
        Parsing midi files
        :param filename: Path to midi files
        :return: Dataframe of note_on/off, note value, period time, and channel of the notations.
                 C4 of note value is 60.
        """
        mid = MidiFile(filename)
        _type = []
        note = []
        time_by_tick = []
        channel = []

        for num, track in enumerate(mid.tracks):
            for msg in track:
                if type(msg) is not int:
                    if msg.is_meta:
                        if msg.type == "set_tempo":
                            tempo = msg.tempo
                    elif not msg.is_meta:
                        if msg.type == "note_on" or msg.type == "note_off":
                            _type.append(msg.type)
                            note.append(msg.note)
                            time_by_tick.append(msg.time)
                            channel.append(msg.channel)
                    else:
                        raise ValueError

        # The unit of tempo is microsecond
        tick = tempo / (mid.ticks_per_beat * 1000000)
        time_by_second = [j * tick for j in time_by_tick]

        d = {"t.py": _type, "note": note, "time": time_by_second, "channel": channel}
        df = pd.DataFrame(d)

        return df

    def generate_stave(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate datasets containing only notes and durations
        :param df: Dataframes converted from midi files
        :return: Dataframe only include notes value and durations. C4 of note value is 60.
        """

        # Check Parameters
        if not "note" in df.columns:
            raise ValueError("Dataframe is lack of note")
        if not "time" in df.columns:
            raise ValueError("Dataframe is lack of note")
        if not "channel" in df.columns:
            raise ValueError("Dataframe is lack of channel")

        # Separate note_on/off
        df_odd = df[df.index % 2 == 0]
        df_even = df[df.index % 2 == 1]
        d = {
            "odd_note": list(df_odd["note"]),
            "even_note": list(df_even["note"]),
            "odd_time": list(df_odd["time"]),
            "even_time": list(df_even["time"]),
            "channel": list(df_odd["channel"])
        }
        df_diff = pd.DataFrame(d)
        # Since both durations are 0, this is a trill which can be ignored
        remove_index = df_diff.query('((odd_note - even_note) != 0) & (odd_time == 0) & (even_time == 0)').index
        df_drop_time0 = df_diff.drop(index=remove_index)
        df_drop_time0_row_num = df_drop_time0.shape[0]

        # The normal case is the duration of note_on equals to 0
        # If note_on is not 0, we will add a blank beat and its duration
        note = []
        time = []
        channel = []
        for i in range(df_drop_time0_row_num):
            if df_drop_time0.iloc[i]["odd_time"] == 0.0:
                note.append(df_drop_time0.iloc[i]["even_note"])
                time.append(df_drop_time0.iloc[i]["even_time"])
                channel.append(df_drop_time0.iloc[i]["channel"])
            elif df_drop_time0.iloc[i]["even_time"] == 0.0:
                note.append(0)
                time.append(df_drop_time0.iloc[i]["odd_time"])
                channel.append(df_drop_time0.iloc[i]["channel"])
            elif (df_drop_time0.iloc[i]["odd_time"] != 0.0) and (df_drop_time0.iloc[i]["even_time"] != 0.0):
                note.append(0)
                note.append(df_drop_time0.iloc[i]["odd_note"])
                time.append(df_drop_time0.iloc[i]["odd_time"])
                time.append(df_drop_time0.iloc[i]["even_time"])
                channel.append(df_drop_time0.iloc[i]["channel"])
                channel.append(df_drop_time0.iloc[i]["channel"])
            else:
                raise ValueError

        # Convert float of note values and channels to integrate
        note = map(int, note)
        channel = map(int, channel)

        dict_result = {
            "note": note,
            "time": time,
            "channel": channel
        }
        df_result = pd.DataFrame(dict_result)

        return df_result

    def generate_str_notation(self, note: list, duration: list, channel: list) -> str:
        """
        Generate the string notation of pygame.midi form
        :param note: The note values of the music. C4 of note value is 60.
        :param duration: Duration of the note value
        :param channel: Soprano (0) and alto (1) clef channels
        :return: The string of play code
        """
        if len(note) != len(duration):
            raise ValueError("The length of Notes List is different from that of Durations")

        s = ""
        for i in range(len(note)):
            if note[i] == 0:
                s = s + f"time.sleep({duration[i]})\n"
            else:
                s = s + f"player.note_on(note={note[i]}, velocity=127, channel={channel[i]})\ntime.sleep({duration[i]})\n" \
                        f"player.note_off(note={note[i]}, velocity=64, channel={channel[i]})\n"
        return s

    def play(self, str_note: str) -> None:
        """
        Parse the string of notation to play note values and their durations using Pygame.midi
        :param str_note: The string notation of pygame.midi form
        :return: None
        """
        pygame.midi.init()
        player = pygame.midi.Output(0)
        player.set_instrument(1)
        exec(str_note)
        del player
        pygame.midi.quit()

    def convert_num2str(self, num: int) -> str:
        """
        Tune the note to the first minor group to ignore the octave,
            which is used to convert to its major chord.
        :param num: The note values, C4 is 60
        :return: The string form of chord
        """
        d = {
            0: "C_M",
            1: "Db_M",
            2: "D_M",
            3: "Eb_M",
            4: "E_M",
            5: "F_M",
            6: "Gb_M",
            7: "G_M",
            8: "Ab_M",
            9: "A_M",
            10: "Bb_M",
            11: "B_M"
        }

        while True:
            if 0 <= num <= 20:
                break
            elif 60 <= num <= 71:
                break
            else:
                if num < 60:
                    num += 12
                elif num > 71:
                    num -= 12

        if 0 <= num <= 20:
            res = None
        else:
            remainder = num % 60
            res = d[remainder]

        return res

    def generate_group_data(self, df: pd.DataFrame):
        """
        Get group cycle data.
        :param df: Dataframe of midi files information including notes, durations, and channels.
        :return: Dataframe of PRL and T/I group cycle data.
        """

        # Check Parameters
        if not "note" in df.columns:
            raise ValueError("Dataframe is lack of note")
        if not "time" in df.columns:
            raise ValueError("Dataframe is lack of note")
        if not "channel" in df.columns:
            raise ValueError("Dataframe is lack of channel")

        # Convert Notes to Pitch
        df["pitch"] = df["note"].apply(self.convert_num2str)
        # Delete rows including NaN values
        df = df.dropna()
        # Initialise group objects
        prl_group = PRL_Group()
        ti_group = TI_Group_Triads()
        # Declare Variable
        list_prl = []
        list_ti = []
        list_channel = []

        for i in df["channel"].unique():
            for j in range(0, len(df["pitch"]) - 1):
                res_prl = prl_group.get_operation(df.iloc[j]["pitch"], df.iloc[j + 1]["pitch"])
                res_ti = ti_group.get_operation(df.iloc[j]["pitch"], df.iloc[j + 1]["pitch"])
                # Check res is not the null list
                if res_prl:
                    if res_prl[0] != "id_.":
                        list_prl.append(res_prl[0])
                if res_ti:
                    if res_ti[0] != "id_.":
                        list_ti.append(res_ti[0])
                        list_channel.append(i)

        dict_group_data = {
            "prl": list_prl,
            "ti": list_ti,
            "channel": list_channel
        }
        df_group_data = pd.DataFrame(dict_group_data)

        return df_group_data

    def ignore_octave(self, num: int) -> int:
        """
        ignore octave, and convert note value to [0, 11]
        :param num: note value, [21, 108]
        :return: relative value
        """
        while True:
            if 0 <= num <= 20:
                break
            elif 60 <= num <= 71:
                break
            else:
                if num < 60:
                    num += 12
                elif num > 71:
                    num -= 12

        if 0 <= num <= 20:
            res = None
        else:
            remainder = (num % 60)
            res = remainder

        return res

    def trans2binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ignoring airs, octaves and channels, the notes are converted to unison octaves and converted to binary matrix.
        :param df: Dataframe of midi files information including notes, durations, and channels.
        :return: Dataframe with binary form
        """
        # Check Parameters
        if not "note" in df.columns:
            raise ValueError("Dataframe is lack of note")
        if not "time" in df.columns:
            raise ValueError("Dataframe is lack of note")
        if not "channel" in df.columns:
            raise ValueError("Dataframe is lack of channel")

        if df["channel"].unique().shape[0] == 1:
            time_cumsum = []
            s = 0
            for i in df["time"]:
                s += i
                time_cumsum.append(0)

            df["time_cumsum"] = time_cumsum

        elif df["channel"].unique().shape[0] == 2:
            df_0 = df.query("channel == 0")
            df_1 = df.query("channel == 1")

            time_cumsum_0 = []
            s0 = 0
            for i in df_0["time"]:
                s0 += i
                time_cumsum_0.append(s0)

            time_cumsum_1 = []
            s = 0
            for i in df_1["time"]:
                s += i
                time_cumsum_1.append(s)

            df["time_cumsum"] = time_cumsum_0 + time_cumsum_1

        else:
            raise ValueError("Please check the value of channel")

        df_sorted = df.sort_values("time_cumsum")

        df_sorted_delete_quiet = df_sorted[~df_sorted['note'].isin([0])]

        df_sorted_delete_quiet["note_ignore_octave"] = df_sorted_delete_quiet["note"].apply(self.ignore_octave)

        indicator = []

        for i in range(df_sorted_delete_quiet["time_cumsum"].shape[0] - 1):
            if df_sorted_delete_quiet["time_cumsum"].iloc[i] == df_sorted_delete_quiet["time_cumsum"].iloc[i + 1]:
                indicator.append(1)
            else:
                indicator.append(0)

        indicator.append(0)

        df_sorted_delete_quiet["indicator"] = indicator

        note_channel_push_back = []
        skip_list = []

        for i in range(df_sorted_delete_quiet.shape[0]):
            if i not in skip_list:
                if df_sorted_delete_quiet["indicator"].iloc[i] == 1:
                    skip_list.append(i + 1)
                else:
                    note_channel_push_back.append(df_sorted_delete_quiet["note_ignore_octave"].iloc[i])
            else:
                note_channel_push_back.append([df_sorted_delete_quiet["note_ignore_octave"].iloc[i - 1],
                                               df_sorted_delete_quiet["note_ignore_octave"].iloc[i]])

        chord = np.zeros([12, len(note_channel_push_back)], int)

        for i, j in enumerate(note_channel_push_back):
            if type(j) == list:
                for k in j:
                    chord[k, i] = 1
            else:
                chord[j, i] = 1

        return pd.DataFrame(chord)

    def operator(self, state: int, action: int) -> int:
        """
        Define the operator of binary data, this operation can be reversed
        :param state: The state of notes, which includes {0, 1}
        :param action: The action of notes, which includes {0, 1}
        :return: The next state of notes, which includes {0, 1}
        """
        if state not in [0, 1]:
            raise ValueError("State Value Error")
        if action not in [0, 1]:
            raise ValueError("State Value Error")

        if state == 0:
            if action == 0:
                res = 0
            elif action == 1:
                res = 1
        elif state == 1:
            if action == 0:
                res = 1
            elif action == 1:
                res = 0

        return res

    def calculate_list(self, operator: any, l1: list, l2: list) -> list:
        """
        Compute two binary lists of the same length at operator function
        :param operator: The operator of binary data, which is a function object
        :param l1: The binary list, which includes {0, 1}
        :param l2: The binary list, which includes {0, 1}
        :return: The binary list, which includes {0, 1}
        """
        if len(l1) != len(l2):
            raise ValueError("Shape Error")

        res_list = []

        for i in range(len(l1)):
            res_list.append(operator(l1[i], l2[i]))

        return res_list

    def stave2action(self, df: pd.DataFrame, operator: any) -> pd.DataFrame:
        action_matrix = np.zeros([df.shape[0], df.shape[1] - 1])

        for i in range(df.shape[1] - 1):
            action = self.calculate_list(operator, df.iloc[:, i], df.iloc[:, i + 1])
            action_matrix[:, i] = action

        df_res = pd.DataFrame(action_matrix).astype(np.uint8)

        return df_res


if __name__ == "__main__":
    # t = Toolbox()
    df = pd.read_csv("../Data/CSV/Binary/BMV772.csv", header=0)
    # r = t.stave2action(df, t.operator)
    s0 = [1,0,1,0]
    a  = [1,1,0,0]
    s1 = [0,1,1,0]

