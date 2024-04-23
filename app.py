from juliacall import Main as jl
import streamlit as st
import pandas as pd
from datetime import datetime # precisa?
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.facecolor'] = 'green'

dic_formações = {"3-5-2": ([53,15,25,25,53,47,47,70,85,85],[8,34,22,46,60,20,48,34,25,43]), "4-3-3": ([35,20,20,35,43,55,55,85,88,85],[10,25,43,58,34,22,46,10,34,58]), "4-1-4-1": ([30,20,20,30,44,60,60,60,60,85],[10,25,43,58,34,7,25,43,61,34]), "4-4-2": ([30,20,20,30,60,50,50,60,85,85],[10,25,43,58,10,25,43,58,25,43]), "3-4-3": ([55,20,27,27,55,50,50,82,88,82],[8,34,22,46,60,23,45,15,34,53])}

def desenha_campo(form, df_orig, team_color = "white", secondary_color = "black"):
    df = df_orig.copy()
    
    plt.figure(figsize=(10, 6))
    plt.plot([0,105,105,0,0,52.5,52.5],[0,0,68,68,0,0,68], color="white")
    plt.plot([0,16.5,16.5,0],[13.84,13.84,54.16,54.16], color = "white")
    plt.plot([105,88.5,88.5,105],[13.84,13.84,54.16,54.16], color = "white")
    plt.plot([0,5.49,5.49,0],[24.86,24.86,43.14,43.14], color = "white")
    plt.plot([105,99.5,99.55,105],[24.86,24.86,43.14,43.14], color = "white")
    plt.plot(9.15*np.sin(np.arange(0, 2*np.pi, 2*np.pi/100)) + 52.5*np.ones(100), 9.15*np.cos(np.arange(0, 2*np.pi, 2*np.pi/100)) + 34*np.ones(100),color="white")
    plt.plot(3.66*np.sin(np.arange(0, np.pi, np.pi/100)) + 16.5*np.ones(100), 3.66*np.cos(np.arange(0, np.pi, np.pi/100)) + 34*np.ones(100), color="white")
    plt.plot(3.66*np.sin(np.arange(np.pi, 2*np.pi, np.pi/100)) + 88.5*np.ones(100), 3.66*np.cos(np.arange(np.pi, 2*np.pi, np.pi/100)) + 34*np.ones(100), color="white")

    plt.scatter(dic_formações[form][0], dic_formações[form][1], color = team_color, s = 80, edgecolors = secondary_color)
    for i in range(0, len(df)-1):
        plt.text(dic_formações[form][0][i], dic_formações[form][1][i] + 1.5, df.Player.iloc[i+1], fontsize = 12, ha = 'center')
        plt.text(dic_formações[form][0][i], dic_formações[form][1][i] - 3.5, df.Apps.iloc[i+1], fontsize = 12, ha = 'center')
    plt.scatter([2], [34], color = "black", s = 80, edgecolors = "white")
    plt.text(2, 35.5, df.Player.iloc[0], fontsize = 12, ha = 'center')
    plt.text(2, 30.5, df.Apps.iloc[0], fontsize = 12, ha = 'center')
    # plt.text(dic_formações[form][0][1], dic_formações[form][1][1] + 2.5, df.Player.iloc[2], fontsize = 12, ha = 'center')

    return plt

st.title('Optimal Signings')

season = st.radio("Season", [2023, 2022], horizontal=True)
data = pd.read_csv(f"dados/dados{season}.csv")
df_means = pd.read_csv(f"dados/medias{season}.csv")
data['Position'] = data['Position'].replace({
    "midfield": "Centre-Back",
    "Second Striker": "Centre-Forward",
    "Right Midfield": "Right Winger",
    "Left Midfield": "Left Winger",
    "attack": "Left Winger"
})
data = data[data['Mins_Per_90'] >= 5]
if season >= 2023:
    data['Birth'] = pd.to_datetime(data['Birth'], format="%m/%d/%Y")
else:
    data['Birth'] = pd.to_datetime(data['Birth'])
data['Age'] = (datetime(season, 7, 1) - data['Birth']).dt.days / 365
data['Value'] = data['Value'] / 10**6

jl.include("python.jl")


starting = st.checkbox("Only starters", value = False)
teams = data['Squad'].unique()
teams.sort()
team = st.selectbox("Team", teams)
# dict_stats_default = {"Clr": 0.1,"SCA_SCA": 0.1,"PrgDist_Carries": 0.1,"Gls": 0.1, "PSxG+_per__minus__Expected": 0.1,
#     "PrgR_Receiving": 0.1,"Tkl+Int": 0.1,"xA": 0.1,"PrgDist_Total": 0.1,"Succ_Take": 0.1}
dict_stats = {}
stats = st.multiselect("Attributes of interest", options=data.columns[7:-5] )
for stat in stats:
    pct = st.slider(f"Percentile of {stat}", 0, 100)
    dict_stats.update({stat: pct/100})

formation = st.selectbox("Formation", ['Any', '3-4-3', '3-5-2', '4-1-4-1', '4-3-3', '4-4-2', '5-4-1'])
budget = float(st.slider("Budget (mi of euros)", 0, 1500))
age = float(st.slider("Maximum average age", 17, 42))
keep = st.number_input("Minimum percentage of players to be kept", 0, 100)/100 # ajeitar quando tiver segundo estágio
own_val = 1.0
time_limit = 60.0
if not starting:
    # scenarios=scenarios, max_players=max_players, gap=gap, foreigners=foreigners, healthy=healthy)
    scenarios = st.slider("Number of scenarios", 1, 100) # aumentar o limite máximo?
else:
    scenarios = 0
    

if st.button("Run"):

    x = jl.recommended_signings(team, season, dict_stats, time_limit = time_limit, age_limit = age, pct_keep = keep, starting11 = starting, own_players_val = own_val, formation = formation, budget = budget, scenarios = scenarios)    
    df = pd.DataFrame(jl.eachrow(x[0]))
    # df[df.columns[:4]] = df[df.columns[:4]].map(lambda x: str(x))
    df = df.map(lambda x: str(x))
    df.columns = jl.names(x[0])

    if starting:
        df['Apps'] = ''
        filtered = df
    else:
        df['Apps'] = df['Apps'].map(lambda x: int(x))
        sorted_apps = df['Apps'].sort_values(ascending=False)
        limit = sorted_apps.iloc[10]
        filtered = df[df['Apps'] >= limit]

    plt = desenha_campo(x[3], filtered)
    st.pyplot(plt)

    st.markdown("Recommended squad:")
    st.write(df)

    st.markdown("Squad cost (millions of Euros):")
    st.write(x[1])

    st.markdown("Score:")
    st.write(x[2])
