import streamlit as st
import pandas as pd
from datetime import datetime # precisa?
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from ortools.math_opt.python import mathopt
import random


all_positions = ["Goalkeeper", "Right-Back", "Centre-Back", "Left-Back", "Defensive Midfield", "Central Midfield", "Attacking Midfield", "Right Winger", "Centre-Forward", "Left Winger"]

def position_sort(x):
    return all_positions.index(x)

dict_formations = {}  # [gks, rbs, cbs, lbs, cdms, cms, cams, rws, sts, lws]
dict_formations["4-4-2"] = [1, 1, 2, 1, 1, 1, 0, 1, 2, 1]
dict_formations["4-1-4-1"] = [1, 1, 2, 1, 1, 2, 0, 1, 1, 1]
dict_formations["4-3-3"] = [1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
dict_formations["3-5-2"] = [1, 1, 3, 1, 1, 1, 1, 0, 2, 0]
dict_formations["5-4-1"] = [1, 1, 3, 1, 1, 1, 0, 1, 1, 1]
dict_formations["3-4-3"] = [1, 1, 3, 1, 1, 1, 0, 1, 1, 1]

def starters(roster, formation):
    formation = dict_formations[formation]
    starters = pd.DataFrame()
    for pos, n in zip(all_positions, formation):
        options = roster[roster['Position'] == pos].sort_values('Apps', ascending=False)
        starters = pd.concat([starters, options.iloc[:int(n)]])
    return starters

df_regs = pd.read_csv("dados/regs.csv", index_col=0)

dict_positions = {"Goalkeeper": 0, "Right-Back": 1, "Centre-Back": 2, "Left-Back": 3, "Defensive Midfield": 4, "Central Midfield": 5, "Attacking Midfield": 6, "Right Winger": 7, "Centre-Forward": 8, "Left Winger": 9}


def recommend_signings_single_stage(team, data_orig, df_means, dict_stats, time_limit=60.0, age_limit=45, min_keep=11, starting11=True, own_players_val=1.0, formation="", budget=0.0):
    data = data_orig.copy()

    if formation == "Qualquer":
        formation_results = [recommend_signings_single_stage(team, data_orig, df_means, dict_stats, time_limit=time_limit, age_limit=age_limit, min_keep=min_keep, own_players_val=own_players_val, formation=formation, budget=budget) for formation,v in dict_formations.items()]
        val, idx = max([(x[3], i) for i, x in enumerate(formation_results)])
        return formation_results[idx]

    I = list(range(1, data.shape[0]+1))
    idx_current = [i for i, row in data.iterrows() if row["Squad"] == team]
    S = [df_means.columns.get_loc(k) + 1 for k in dict_stats.keys()]

    data.loc[idx_current, "Value"] = np.round(own_players_val * data.loc[idx_current, "Value"]).astype(int)

    model = mathopt.Model()

    #model.SetTimeLimit(int(time_limit))

    x = {}
    for i in I:
        x[i] = model.IntVar(0, 1, "x[%i]" % i)

    if budget == 0:
        objective = model.Objective()
        for i in I:
            objective.SetCoefficient(x[i], data.loc[i, "Value"])
        objective.SetMinimization()
    else:
        dict_stats_normalized = {}
        for stat, pct in dict_stats.items():
            mini, maxi = data[stat].min(), data[stat].max()
            dict_stats_normalized[stat] = (data[stat] - mini) / (maxi - mini)

        objective = model.Objective()
        for i in I:
            for stat, norm in dict_stats_normalized.items():
                objective.SetCoefficient(x[i], norm[i])
        objective.SetMaximization()

        budget_constraint = model.Constraint(-model.infinity(), budget)
        for i in I:
            budget_constraint.SetCoefficient(x[i], data.loc[i, "Value"])

    for stat, pct in dict_stats.items():
        constraint = model.Constraint(pct * df_means[stat].quantile(), model.infinity())
        for i in I:
            constraint.SetCoefficient(x[i], data.loc[i, stat])

    if min_keep > 0:
        constraint = model.Constraint(min_keep, model.infinity())
        for j in idx_current:
            constraint.SetCoefficient(x[j], 1)

    constraint = model.Constraint(-model.infinity(), age_limit)
    for i in I:
        constraint.SetCoefficient(x[i], data.loc[i, "Age"])

    if starting11:
        constraint = model.Constraint(11, 11)
        for i in I:
            constraint.SetCoefficient(x[i], 1)

    positions = dict_formations[formation]
    for i, qtd_pos in enumerate(positions):
        pos_name = all_positions[i]
        bool_pos = [name == pos_name for name in data["Position"]]

        constraint = model.Constraint(qtd_pos, qtd_pos)
        for i in I:
            constraint.SetCoefficient(x[i], bool_pos[i])

    model.Solve()

    # if model.Status() == pywraplp.Solver.INFEASIBLE:
    #     print("It is impossible to build a team respecting such constraints")
    #     return pd.DataFrame(), 0.0, 0.0, ""

    rs = data.loc[[i for i in I if abs(x[i].solution_value()) > 10**(-12)]].sort_values(by="Position", key=position_sort)

    score = []
    for stat, pct in dict_stats.items():
        mini, maxi = data[stat].min(), data[stat].max()
        score.append(np.mean((rs[stat] - mini) / (maxi - mini)))

    ret = rs[["Player", "Squad", "Position", "Age", "Value"] + list(dict_stats.keys())]
    ret.loc[:, "Age":] = np.round(ret.loc[:, "Age":], decimals=2)
    return ret, sum(rs["Value"]), round(np.mean(score), 4), formation, starters(rs, formation)

def recommend_signings_multi_stage(team, data_orig, df_means, dict_stats, time_limit=60.0, age_limit=45, pct_keep=0.0, own_players_val=1.0, formation="", budget=0.0, scenarios=100, max_players=1922, gap=0.01, foreigners=4, healthy=np.array([[]]), pred_method="Naïve"):
    if formation == "Qualquer":
        formation_results = [recommend_signings_multi_stage(team, data_orig, df_means, dict_stats, time_limit=time_limit, age_limit=age_limit, pct_keep=pct_keep, own_players_val=own_players_val, formation=formation, budget=budget, scenarios=scenarios, max_players=max_players, gap=gap, healthy=healthy, pred_method=pred_method) for formation,v in dict_formations.items()]
        val, idx = max([(x[3], i) for i, x in enumerate(formation_results)])
        return formation_results[idx]

    data = data_orig.copy()

    I = list(range(data.shape[0]))
    idx_current = [i for i, row in data.iterrows() if row["Squad"] == team]
    S = [df_means.columns.get_loc(k) + 1 for k in dict_stats.keys()]
    C = list(range(scenarios))

    data.loc[idx_current, "Value"] = np.round(own_players_val * data.loc[idx_current, "Value"]).astype(int)

    model = mathopt.Model()
    #model.SetTimeLimit(int(time_limit))
    #model.SetMIPGap(gap)

    x = {}
    for i in I:
        x[i] = model.add_binary_variable(name="x[%i]" % i)

    y = {}
    for i in I:
        for c in C:
            y[i, c] = model.add_binary_variable(name=f"y[{i},{c}]")

    dict_stats_normalized = {}
    for stat, pct in dict_stats.items():
        mini, maxi = data[stat].min(), data[stat].max()
        dict_stats_normalized[stat] = (data[stat] - mini) / (maxi - mini)

    random.seed(21)
    healthy = np.random.rand(data.shape[0], scenarios) > data["Injury_Prob"].values[:, np.newaxis]
    healthy = np.zeros((data.shape[0], scenarios))
    for i in range(data.shape[0]):
        prob = data.loc[i, "Injury_Prob"]
        num_scen_healthy = int(round(scenarios*(1-prob)))
        v = np.zeros(scenarios)
        v[:num_scen_healthy] = np.ones(num_scen_healthy)
        random.shuffle(v)
        healthy[i,:] = v

    pre_game_stats = np.zeros((len(I), len(S), scenarios))

    for k, stat in enumerate([stat for stat in dict_stats_normalized.keys()]):
        if pred_method == "Naïve":
            pre_game = np.column_stack([np.repeat([el], scenarios) for el in data.loc[:,stat]]) # SxI (inverter?)
        elif pred_method == "Normal":
            pre_game = np.column_stack([np.zeros(scenarios) if el == 0 else np.random.normal(el, 2/3*np.std(data.loc[:,stat]), scenarios) for el in data.loc[:,stat]]) # SxI (inverter?)
        elif pred_method == "Séries temporais":
            stat_ = "Prog_Receiving" if stat == "PrgR_Receiving" else stat # generalize with get(dict)
            stat_ = dict({"PrgR_Receiving": "Prog_Receiving", "Succ_Take": "Succ_Dribbles"}).get(stat, stat)
            pos_ = [dict_positions[pos] for pos in data.Position]
            X = np.column_stack([np.ones(len(data)), data.loc[:,stat], data.Age, data.Age ** 2, [posit == "FW" for posit in pos_], [posit == "FW" for posit in pos_], [posit == "FW" for posit in pos_]])
            coefs_vcov = df_regs[stat_]
            eta = X @ coefs_vcov.iloc[:7]
            vc = coefs_vcov.iloc[7:].values.reshape((7,7))
            vcovXnewT = vc @ X.T
            stdeta = [np.sqrt(np.dot(X[i,:], vcovXnewT[:,i])) for i in range(X.shape[0])]
            lower, upper = eta - 1.96*np.array(stdeta), eta + 1.96*np.array(stdeta)
            dists = [norm(loc=e, scale=(e-l)/1.96) for e, l in zip(eta, lower)]
            pre_game = np.column_stack([dist.rvs(scenarios) for dist in dists]) # SxI (inverter?)

        pre_game_stats[:,k,:] = pre_game.T

    objective = 0
    for i in I:
        for c in C:
            for k, stat in enumerate([stat for stat in dict_stats_normalized.keys()]):
                objective += y[i, c] * pre_game_stats[i, k, c]
    model.maximize(objective)

    model.add_linear_constraint(sum(x[i]*data.Value.loc[i] for i in I) <= budget)

    for stat in dict_stats.keys():
        pct = dict_stats[stat]
        # sum(y[i,c]*data[i,stat] for i in I, c in C) >= quantile(df_means[:,stat], pct) * sum(y[i,c] for i in I, c in C)
        model.add_linear_constraint(sum(y[i, c]*data.loc[i, stat] for i in I for c in C) >= df_means[stat].quantile(pct) * sum(y[i, c] for i in I for c in C))

    model.add_linear_constraint(sum(y[j, c] for j in idx_current for c in C) >= sum(pct_keep for c in C) * 11)

    model.add_linear_constraint(sum(y[i, c]*data.loc[i, "Age"] for i in I for c in C) <= age_limit * sum(y[i, c] for i in I for c in C))

    positions = dict_formations[formation]
    for i, qtd_pos in enumerate(positions):
        pos_name = all_positions[i]
        bool_pos = [name == pos_name for name in data["Position"]]

        for c in C:
            model.add_linear_constraint(sum(y[i, c]*bool_pos[i] for i in I) == qtd_pos)

    for i in I:
        for c in C:
            model.add_linear_constraint(y[i, c] <= x[i])
            model.add_linear_constraint(y[i, c] <= healthy[i,c])

    for i in I:
        model.add_linear_constraint(x[i] <= sum(y[i, c] for c in C))

    model.add_linear_constraint(sum(x[i] for i in I) <= max_players)

    result = mathopt.solve(model, mathopt.SolverType.GSCIP)

    if result.termination.reason != mathopt.TerminationReason.OPTIMAL:
        return pd.DataFrame(), 0.0, 0.0, "", pd.DataFrame()

    data["Position"] = pd.Categorical(data["Position"], categories=all_positions, ordered=True)
    rs = data.loc[[i for i in I if abs(result.variable_values()[x[i]]) > 10**(-12)]]
    rsy = [data.loc[[i for i in I if abs(result.variable_values()[y[i, c]]) > 10**(-12)]] for c in C]
    rs["Apps"] = [sum([p in rsy[c].Player.values for c in range(scenarios)]) for p in rs.Player]

    score = []
    for stat, pct in dict_stats.items():
        mini, maxi = data[stat].min(), data[stat].max()
        score.append(np.mean((rs[stat] - mini) / (maxi - mini)))

    ret = rs[["Player", "Squad", "Position", "Apps", "Age", "Value"] + list(dict_stats.keys())]
    ret.loc[:, "Age":] = np.round(ret.loc[:, "Age":], decimals=2)
    ret = ret.sort_values("Position")
    return ret, sum(rs["Value"]), round(np.mean(score), 4), formation, starters(rs, formation)

def recommended_signings(team, season, dict_stats, time_limit=60.0, age_limit=45, pct_keep=0.0, starting11=False, own_players_val=1.0, formation="", budget=0.0, scenarios=100, max_players=1922, gap=0.01, foreigners=4, healthy=np.array([[]]), pred_method="Naïve"):
    data = pd.read_csv(f"dados/dados{season}.csv")
    data.replace({"Position": {"midfield": "Centre-Back", "Second Striker": "Centre-Forward", "Right Midfield": "Right Winger", "Left Midfield": "Left Winger", "attack": "Left Winger"}}, inplace=True)
    df_means = pd.read_csv(f"dados/medias{season}.csv")
    data = data[data["Mins_Per_90"] >= 5]
    data.reset_index(drop=True, inplace=True)
    data["Age"] = (pd.to_datetime(f"{season}-07") - pd.to_datetime(data["Birth"], format="%m/%d/%Y")).dt.days / 365
    data["Value"] /= 10**6

    if starting11:
        return recommend_signings_single_stage(team, data, df_means, dict_stats, time_limit=time_limit, age_limit=age_limit, min_keep=int(pct_keep*11), own_players_val=own_players_val, formation=formation, budget=budget)
    else:
        prob_inj = pd.read_csv("dados/injury_prob.csv")
        dict_prob_inj = dict(zip(prob_inj["Url"], prob_inj["Prob"]))
        data["Injury_Prob"] = data["Url"].map(lambda x: dict_prob_inj.get(x, prob_inj["Prob"].mode()[0]))
        yell, red = 3/df_means["CrdY"].quantile(.5), 1/df_means["CrdR"].quantile(.5)
        data["Injury_Prob"] += 1/yell + 1/red

        return recommend_signings_multi_stage(team, data, df_means, dict_stats, time_limit=time_limit, age_limit=age_limit, pct_keep=pct_keep, own_players_val=own_players_val, formation=formation, budget=budget, scenarios=scenarios, max_players=50, gap=gap, foreigners=foreigners, healthy=healthy, pred_method=pred_method)






plt.rcParams['axes.facecolor'] = 'green'

dic_formações = {"3-5-2": ([53,15,25,25,53,47,47,70,85,85],[8,34,22,46,60,20,48,34,25,43]), "4-3-3": ([35,20,20,35,43,55,55,85,88,85],[10,25,43,58,34,22,46,10,34,58]), "4-1-4-1": ([30,20,20,30,44,60,60,60,85,60],[10,25,43,58,34,25,43,7,34,61]), "4-4-2": ([30,20,20,30,50,50,60,85,85,60],[10,25,43,58,25,43,10,25,43,58]), "3-4-3": ([55,20,27,27,55,50,50,82,88,82],[8,34,22,46,60,23,45,15,34,53]), "5-4-1" : ([55,20,27,27,55,50,50,82,88,82],[8,34,22,46,60,23,45,15,34,53])}

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

st.title('VITÓRIA')
st.markdown('Value-based Intelligence for Transfers Optimization & Roster Improvement Analysis')

season = st.radio("Temporada", [2023, 2022], horizontal=True)
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

teste = recommended_signings("Real Madrid", 2023, {"Tkl+Int": 0.1, "SCA_SCA": 0.1}, time_limit=60.0, age_limit=45, pct_keep=0.0, starting11=False, own_players_val=1.0, formation="4-3-3", budget=900.0, scenarios=2)

col_names_eng = data.columns

col_names_pt = [
        "Elenco",
        "Competição",
        "Jogador",
        "País",
        "Idade",
        "Nascimento",
        "Minutos/Partida",
        "Desarmes",
        "Desarmes certos",
        "Desarmes no terço defensivo",
        "Desarmes no terço médio",
        "Desarmes no terço ofensivo",
        "Dribles desarmados",
        "Dribles enfrentados",
        "Porcentagem de sucesso nos dribles enfrentados",
        "Dribles sofridos",
        "Bloqueios",
        "Bloqueios de chute",
        "Bloqueios de passe",
        "Interceptações",
        "Desarmes + Interceptações",
        "Afastamentos de bola",
        "Erros",
        "Chances criadas",
        "Chances criadas por partida",
        "Chances criadas por passes em jogo",
        "Chances criadas por bolas paradas",
        "Chances criadas por dribles",
        "Chances criadas por chutes",
        "Chances criadas por faltas sofridas",
        "Chances criadas por ações defensivas",
        "Gols criados",
        "Gols criados por partida",
        "Gols criados por passes em jogo",
        "Gols criados por bolas paradas",
        "Gols criados por dribles",
        "Gols criados por chutes",
        "Gols criados por faltas sofridas",
        "Gols criados por ações defensivas",
        "Gols sofridos",
        "Gols sofridos por partida",
        "Chutes a gol contra",
        "Defesas",
        "Porcentagem de defesa",
        "Empates",
        "Derrotas",
        "Jogos sem sofrer gols",
        "Porcentagem de jogos sem sofrer gols",
        "Pênaltis enfrentados",
        "Gols de pênalti sofridos",
        "Pênaltis defendidos",
        "Pênaltis enfrentados perdidos",
        "Porcentagem de defesa de pênalti",
        "Gols sofridos",
        "Gols de pênalti sofridos",
        "Gols de falta sofridos",
        "Gols de escanteio sofridos",
        "Gols contra",
        "Gols sofridos esperados (PSxG)",
        "Gols sofridos esperados por chute no gol (PSxG por chute no gol)",
        "Gols salvos acima do esperado",
        "Gols salvos acima do esperado por partida",
        "Lançamentos certos",
        "Tentativas de lançamentos",
        "Porcentagem de acerto no lançamento",
        "Passes tentados por goleiro",
        "Lançamentos com a mão",
        "Porcentagem de lançamentos sobre passes",
        "Distância média dos passes",
        "Tiros de meta cobrados",
        "Porcentagem de lançamentos em tiros de meta",
        "Distância média dos tiros de meta cobrados",
        "Cruzamentos sofridos",
        "Cruzamentos interceptados",
        "Porcentagem de cruzamentos interceptados",
        "Ações de goleiro fora da área",
        "Ações de goleiro fora da área por partida",
        "Distância média das ações de líbero",
        "Cartões amarelos",
        "Cartões vermelhos",
        "Dois cartões amarelos no jogo",
        "Faltas cometidas",
        "Faltas sofridas",
        "Impedimentos",
        "Cruzamentos",
        "Desarmes certos",
        "Pênaltis sofridos",
        "Pênaltis cometidos",
        "Gols contra",
        "Bolas recuperadas",
        "Disputas áereas vencidas",
        "Disputas aéreas perdidas",
        "Porcentagem de sucesso nas disputas aéreas",
        "Passes certos",
        "Passes tentados",
        "Porcentagem de acerto no passe",
        "Distância total dos passes",
        "Distância progressiva dos passes",
        "Passes curtos certos",
        "Passes curtos tentados",
        "Porcentagem de acerto no passe curto",
        "Passes médios certos",
        "Passes médios tentados",
        "Porcentagem de acerto no passe médio",
        "Passes longos certos",
        "Passes longos tentados",
        "Porcentagem de acerto no passe longo",
        "Assistências",
        "Gols esperados de chutes assistidos (xAG)",
        "Assistências esperadas (xA)",
        "Diferença entre assistências e assistências esperadas",
        "Passes para finalizações",
        "Passes no terço final",
        "Passes para dentro da área",
        "Cruzamentos para dentro da área",
        "Passes progressivos",
        "Tentativas de passes",
        "Passes durante o jogo",
        "Passes em bola parada",
        "Passes em cobranças de faltas",
        "Passes em profundidade",
        "Inversões de jogo",
        "Cruzamentos",
        "Laterais cobrados",
        "Escanteios cobrados",
        "Escanteios cobrados com curva por dentro",
        "Escanteios cobrados com curva por fora",
        "Escanteios cobrados sem curva",
        "Passes completos",
        "Passes para impedimento",
        "Passes bloqueados",
        "Toques na bola",
        "Toques na própria área",
        "Toques no terço defensivo",
        "Toques no terço médio",
        "Toques no terço ofensivo",
        "Toques na área adversária",
        "Toques durante o jogo",
        "Tentativas de drible",
        "Dribles certos",
        "Porcentagem de sucesso nos dribles",
        "Dribles errados",
        "Porcentagem de insucesso nos dribles",
        "Conduções de bola",
        "Distância percorrida conduzindo a bola",
        "Distância progressiva conduzindo a bola",
        "Conduções de bola progressivas",
        "Conduções de bola no terço final",
        "Conduções de bola para dentro da área adversária",
        "Perdas de controle de bola",
        "Conduções desarmadas",
        "Passes recebidos",
        "Passes progressivos recebidos",    
        "Gols",
        "Finalizações",
        "Finalizações no gol",
        "Porcentagem de acerto no chute",
        "Finalizações por partida",
        "Finalizações no gol por partida",
        "Gols por finalização",
        "Gols por finalização no gol",
        "Distância média das finalizações",
        "Finalizações de falta",
        "Gols de pênalti",
        "Pênaltis cobrados",
        "Gols esperados (xG)",
        "Gols esperados excluindo pênaltis (npxG)",
        "Gols esperados por finalização excluindo pênaltis (npxG por finalização)",
        "Diferença acima do esperado (Gols - xG)",
        "Diferença acima do esperado excluindo pênaltis",
        "Min/Partida",
        "Gols",
        "Gols + Assistências",
        "Gols (sem pênaltis)",
        "Pênaltis convertidos",
        "Pênaltis cobrados",
        "Assistências esperadas (xAG)",
        "Gols esperados (xG) + Assistência esperada (xAG)",
        "Conduções progressivas",
        "Passes progressivos",
        "Passes progressivos recebidos",
        "Gols por partida",
        "Assistências por partida",
        "Gols + Assistências por partida",
        "Gols (sem pênaltis) por partida",
        "Gols + Assistências (sem pênaltis) por partida",
        "Gols esperados por partida",
        "Assistências esperadas por partida",
        "Gols esperados (xG) + Assistência esperada (xAG) por partida",
        "Gols esperados não pênalti (npxG) por partida",
        "Gols esperados (xG) + Assistência esperada (xAG) não pênalti por partida",
        "URL",
        "Nome",
        "Nascimento",
        "Posição",
        "Valor",
    ]
    
pt_to_eng = {pt: eng for pt, eng in zip(col_names_pt, col_names_eng)}
data.columns = col_names_pt
data = data.loc[:,~data.columns.duplicated()]

positions_pt = {'Goalkeeper': 'Goleiro', 'Centre-Back': 'Zagueiro', 'Right-Back': 'Lateral Direito', 'Left-Back': 'Lateral Esquerdo', 'Defensive Midfield': 'Volante', 'Central Midfield': 'Meia Central', 'Right Midfield': 'Meia Direita', 'Left Midfield': 'Meia Esquerda', 'Attacking Midfield': 'Meia Ofensiva', 'Centre-Forward': 'Centroavante', 'Second Striker': 'Segundo Atacante', 'Right Winger': 'Ponta Direita', 'Left Winger': 'Ponta Esquerda'}
data['Posição'] = data['Posição'].map(positions_pt)

prioridades = pd.read_csv("dados/prioridades de atributos.csv")
idx_pri = [prioridades['Prioridade'] == i for i in range(prioridades['Prioridade'].max()+1)]

new_cols = data.columns.copy().to_list()
new_cols[7:23] = [f"{col} (defesa)" for col in data.columns[7:23]]
new_cols[23:39] = [f"{col} (criação de chances)" for col in data.columns[23:39]]
new_cols[39:76] = [f"{col} (goleiro)" for col in data.columns[39:76]]
new_cols[76:89] = [f"{col} (outros)" for col in data.columns[76:89]]
new_cols[89:126] = [f"{col} (passe)" for col in data.columns[89:126]]
new_cols[126:148] = [f"{col} (condução)" for col in data.columns[126:148]]
new_cols[148:182] = [f"{col} (finalização e assistência)" for col in data.columns[148:182]]

new_to_old = {new: old for new, old in zip(new_cols, data.columns)}
data.columns = new_cols
final_cols = [data.columns[idx_pri[i]] for i in range(prioridades['Prioridade'].max()+1)]
data = pd.concat([data[final_cols[4]], data[final_cols[3]], data[final_cols[2]], data[final_cols[1]], data[final_cols[0]]], axis = 1)

# jl.include("python.jl")

starting = False # st.checkbox("Apenas titulares", value = False)
teams = data['Elenco'].unique()
teams.sort()
team = st.selectbox("Time", teams)
# dict_stats_default = {"Clr": 0.1,"SCA_SCA": 0.1,"PrgDist_Carries": 0.1,"Gls": 0.1, "PSxG+_per__minus__Expected": 0.1,
#     "PrgR_Receiving": 0.1,"Tkl+Int": 0.1,"xA": 0.1,"PrgDist_Total": 0.1,"Succ_Take": 0.1}
dict_stats = {}
stats = st.multiselect("Atributos de interesse", options=data.columns[:-7])
for stat in stats:
    pct = st.slider(f"Percentil de {stat}", 0, 100)
    dict_stats.update({pt_to_eng[new_to_old[stat]]: pct/100})

pred_method = st.radio("Método de previsão", ['Naïve', 'Normal', 'Séries temporais'], index = 0)
formation = st.selectbox("Esquema tático", ['Qualquer', '3-4-3', '3-5-2', '4-1-4-1', '4-3-3', '4-4-2', '5-4-1'])
budget = float(st.slider("Orçamento (milhões de euros)", 0, 1500))
age = float(st.slider("Valor máximo para a idade média", 17, 42))
keep = st.number_input("Percentagem mínima de jogadores a serem mantidos", 0, 100)/100 # ajeitar quando tiver segundo estágio
own_val = 1.0
time_limit = 60.0
if not starting:
    # scenarios=scenarios, max_players=max_players, gap=gap, foreigners=foreigners, healthy=healthy)
    scenarios = st.slider("Número de cenários", 1, 100) # aumentar o limite máximo?
else:
    scenarios = 0
    
execution_time = scenarios / 2.5
execution_time *= 5 if formation == 'Qualquer' else 1
st.markdown(f"Tempo estimado de execução: {int(execution_time)} segundos")
if st.button("Otimizar"):

    x = recommended_signings(team, season, dict_stats, time_limit = time_limit, age_limit = age, pct_keep = keep, starting11 = starting, own_players_val = own_val, formation = formation, budget = budget, scenarios = scenarios, pred_method = pred_method)    
    # x = jl.recommended_signings(team, season, dict_stats, time_limit = time_limit, age_limit = age, pct_keep = keep, starting11 = starting, own_players_val = own_val, formation = formation, budget = budget, scenarios = scenarios, pred_method = pred_method)    
    df = x[0]
    if len(df) == 0:
        st.markdown("Não foi possível encontrar um elenco que satisfaça esses critérios.")
        st.stop()
    # df[df.columns[:4]] = df[df.columns[:4]].map(lambda x: str(x))
    df = df.map(lambda x: str(x))
    df.columns = ["Jogador", "Elenco", "Posição", "Jogos", "Idade", "Valor"] + stats
    starters_ = x[4]

    # if starting:
    #     df['Apps'] = ''
    #     filtered = df
    # else:
    #     df['Apps'] = df['Apps'].map(lambda x: int(x))
    #     sorted_apps = df['Apps'].sort_values(ascending=False)
    #     limit = sorted_apps.iloc[10]
    #     filtered = df[df['Apps'] >= limit]

    plt = desenha_campo(x[3], starters_)
    st.pyplot(plt)

    st.markdown("Elenco recomendado:")
    st.write(df)

    st.markdown("Custo total (milhões de euros):")
    st.write(x[1])

    st.markdown("Score:")
    st.write(x[2])
