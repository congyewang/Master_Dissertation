import pandas as pd

from toolbox.toolbox import Toolbox
import glob
import statsmodels.stats.weightstats as st
from scipy import stats

toolbox = Toolbox()

df_LSTM_Generation = toolbox.get_track_data("./Model/Complete/LSTM/LSTM_Generation.mid")
notes_LSTM_Generation = toolbox.generate_stave(df_LSTM_Generation)
group_LSTM_Generation = toolbox.generate_group_data(notes_LSTM_Generation)

df_GAIL_Generation = toolbox.get_track_data("./Model/Complete/GAIL/GAIL_Generation.mid")
notes_GAIL_Generation = toolbox.generate_stave(df_GAIL_Generation)
group_GAIL_Generation = toolbox.generate_group_data(notes_GAIL_Generation)

group_LSTM_Generation.head()

d_prl = dict(
    zip(
        group_LSTM_Generation["prl"].unique().tolist(),
        range(len(group_LSTM_Generation["prl"].unique())),
        )
)

d_ti = dict(
    zip(
        [f"T{i}" for i in range(12)],
        range(12)
    )
)

LSTM_prl = group_LSTM_Generation["prl"].apply(lambda x: d_prl[x])
LSTM_ti = group_LSTM_Generation["ti"].apply(lambda x: d_ti[x])

GAIL_prl = group_GAIL_Generation["prl"].apply(lambda x: d_prl[x])
GAIL_ti = group_GAIL_Generation["ti"].apply(lambda x: d_ti[x])

expert_prl = []
expert_ti = []

for file in glob.glob("./Data/CSV/Group/*.csv"):
    df = pd.read_csv(file)
    expert_prl += df["prl"].apply(lambda x: d_prl[x]).tolist()
    expert_ti += df["ti"].apply(lambda x: d_ti[x]).tolist()


t, p_two, df = st.ttest_ind(GAIL_prl, expert_prl)

print('t: ' + str(t))
print('p-value: ' + str(p_two))
print('df: ' + str(df))

# Normal Test
s_expert_prl, p_expert_prl = stats.normaltest(expert_prl, axis=None)
s_expert_ti, p_expert_ti = stats.normaltest(expert_ti, axis=None)
print("-"*30)
print(f"s_expert_prl: {s_expert_prl}\np_expert_prl: {p_expert_prl}")
print(f"s_expert_ti: {s_expert_ti}\np_expert_ti: {p_expert_ti}")
print("-"*30)
s_LSTM_prl, p_LSTM_prl = stats.normaltest(LSTM_prl, axis=None)
s_LSTM_ti, p_LSTM_ti = stats.normaltest(LSTM_ti, axis=None)
print(f"s_LSTM_prl: {s_LSTM_prl}\np_LSTM_prl: {p_LSTM_prl}")
print(f"s_LSTM_ti: {s_LSTM_ti}\np_LSTM_ti: {p_LSTM_ti}")
print("-"*30)
s_GAIL_prl, p_GAIL_prl = stats.normaltest(GAIL_prl, axis=None)
s_GAIL_ti, p_GAIL_ti = stats.normaltest(GAIL_ti, axis=None)
print(f"s_GAIL_prl: {s_GAIL_prl}\np_GAIL_prl: {p_GAIL_prl}")
print(f"s_GAIL_ti: {s_GAIL_ti}\np_GAIL_ti: {p_GAIL_ti}")
print("-"*30)

# Homogeneity of variance test

W_LSTM, p_LSTM = stats.levene(LSTM_prl, expert_prl, center='mean')
print("W_GAIL: " + str(W_LSTM))
print("p-value_GAIL: " + str(p_LSTM))

W_GAIL, p_GAIL = stats.levene(GAIL_prl, expert_prl, center='mean')
print("W_GAIL: " + str(W_GAIL))
print("p-value_GAIL: " + str(p_GAIL))

# Mann-Whitney U rank test

stats.mannwhitneyu(LSTM_prl, expert_prl, alternative='two-sided')

stats.mannwhitneyu(GAIL_prl, expert_prl, alternative='two-sided')

stats.mannwhitneyu(LSTM_ti, expert_ti, alternative='two-sided')

stats.mannwhitneyu(GAIL_ti, expert_ti, alternative='two-sided')
