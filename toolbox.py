import os
import time

import pandas as pd
import requests
from mido import MidiFile
from tqdm import tqdm
import pygame.midi


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

        d = {"t": _type, "note": note, "time": time_by_second, "channel": channel}
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
