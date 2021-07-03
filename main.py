import os

import pandas as pd

from toolbox.toolbox import Toolbox

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
    midi_dir = "./Data/Midi"
    toolbox.get_mid(midi_dir, artwork_num2url)

    # Parse strings to extracting track data from midi files
    midi_dir = "./Data/Midi"
    for i in list(artwork_num2url.keys()):
        exec(f"df_{i} = toolbox.get_track_data('{midi_dir}/{i}.mid')")

    # Convert dataframe of track to notes and their durations
    dir_data = "./Data/CSV/Note"
    # Check if there are folders in the environment
    if not os.path.exists(dir_data):
        os.makedirs(dir_data)
    # Parse strings to generate dataframes and save them to csv
    for i in list(artwork_num2url.keys()):
        exec(f"stave_{i} = toolbox.generate_stave(df_{i})")
        exec(f"stave_{i}.to_csv('{dir_data}/{i}.csv', index=False)")


def play_csv() -> None:
    data_dir = "./Data/CSV/Note"
    df = pd.read_csv(f"{data_dir}/BMV772.csv")
    note = df["note"]
    duration = df["time"]
    channel = df["channel"]
    str_notation = toolbox.generate_str_notation(note, duration, channel)
    toolbox.play(str_notation)


def get_group_data() -> None:
    artwork_num = []
    data_dir = "./Data/CSV/Note"
    group_data_dir = "./Data/CSV/Group"

    # Check if there are folders in the environment
    if not os.path.exists(group_data_dir):
        os.makedirs(group_data_dir)

    for i in range(15):
        artwork_num.append(f"BMV{i + 772}")

    for j in artwork_num:
        exec(f"df_{j} = pd.read_csv('{data_dir}/{j}.csv')")
        exec(f"group_{j} = toolbox.generate_group_data(df_{j})")
        exec(f"group_{j}.to_csv('{group_data_dir}/{j}.csv', index=False)")


def output2binary():
    artwork_num = []
    data_dir = "./Data/CSV/Note"
    binary_data_dir = "./Data/CSV/Binary"
    # Check if there are folders of rootdir in the environment
    if not os.path.exists(binary_data_dir):
        os.makedirs(binary_data_dir)

    for i in range(15):
        artwork_num.append(f"BMV{i + 772}")

    for j in artwork_num:
        exec(f"df_{j} = pd.read_csv('{data_dir}/{j}.csv')")
        exec(f"binary_{j} = toolbox.trans2binary(df_{j})")
        exec(f"binary_{j}.to_csv('{binary_data_dir}/{j}.csv', index=False)")


def main():
    # generate_data()
    # play_csv()
    # get_group_data()
    output2binary()


if __name__ == "__main__":
    main()
