from juliacall import Main as jl
import streamlit as st
import pandas as pd
from datetime import datetime

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


starting = st.checkbox("Only starters", value = True)
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
budget = float(st.slider("Budget", 0, 1500))
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

    st.markdown("Recommended team:")
    st.write(df)

    st.markdown("Squad cost (millions of Euros):")
    st.write(x[1])

    st.markdown("Score:")
    st.write(x[2])
