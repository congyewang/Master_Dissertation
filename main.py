import os

import pandas as pd

from toolbox import Toolbox

toolbox = Toolbox()


def generate_data():
    # Generate artwork numbers and their urls. Since BWV 786 is special, we add it separately to the dictionary
    artwork_num = []
    url = []

    for i in range(14):
        artwork_num.append(f"BMV{i + 772}")

    for i in range(14):
        if i < 9:
            url.append(f"https://www.8notes.com/school/midi/piano/bach-invention-0{i + 1}.mid")
        else:
            url.append(f"https://www.8notes.com/school/midi/piano/bach-invention-{i + 1}.mid")

    artwork_num2url = dict(zip(artwork_num, url))

    artwork_num2url.update(
        {
            "BMV786": "https://www.8notes.com/school/midi/piano/bach_invention_15_bwv_786.mid"
        }
    )

    # Download midi files
    midi_dir = "./Midi"
    toolbox.get_mid(midi_dir, artwork_num2url)

    # Parse strings to extracting track data from midi files
    midi_dir = "./Midi"
    for i in list(artwork_num2url.keys()):
        exec(f"df_{i} = toolbox.get_track_data('{midi_dir}/{i}.mid')")

    # Convert dataframe of track to notes and their durations
    dir_data = "./Data"
    # Check if there are folders in the environment
    if not os.path.exists(dir_data):
        os.makedirs(dir_data)
    # Parse strings to generate dataframes and save them to csv
    for i in list(artwork_num2url.keys()):
        exec(f"stave_{i} = toolbox.generate_stave(df_{i})")
        exec(f"stave_{i}.to_csv('{dir_data}/{i}.csv', index=False)")


def play_csv() -> None:
    data_dir = "./Data"
    df = pd.read_csv(f"{data_dir}/BMV772.csv")
    note = df["note"]
    duration = df["time"]
    channel = df["channel"]
    str_notation = toolbox.generate_str_notation(note, duration, channel)
    toolbox.play(str_notation)


def main():
    # generate_data()
    play_csv()


if __name__ == "__main__":
    main()
