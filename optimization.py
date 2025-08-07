import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from utils import lineup, starters, all_positions, dict_formations, dict_formations_rev, df_regs, dict_positions, sort_order

def recommend_signings_single_stage(team, data_orig, df_means, dict_stats, time_limit=60.0,
                                     age_limit=45, min_keep=11, starting11=True, own_players_val=1.0, formation="Qualquer", budget=0.0):

    data = data_orig.copy()

    I = list(range(len(data)))
    idx_current = [i for i in range(len(data)) if data.iloc[i].isin(lineup(team, data)).any()] if min_keep > 0 else []
    S = [data.columns.get_loc(k) for k in dict_stats.keys()]
    F = dict_formations.keys()
    P = list(range(len(all_positions)))

    data.loc[idx_current, 'Value'] = (own_players_val * data.loc[idx_current, 'Value']).round().astype(int)

    solver = pywraplp.Solver.CreateSolver('GLOP')

    solver.SetTimeLimit(int(time_limit * 1000))

    x = [solver.BoolVar(f'x[{i}]') for i in I]
    z = [solver.BoolVar(f'z[{f}]') for f in F] # escolher o melhor esquema tático

    if budget == 0:
        solver.Minimize(solver.Sum(x[i] * data['Value'].iloc[i] for i in I))
    else:
        dict_stats_norm = {}
        for stat, pct in dict_stats.items():
            mini, maxi = data[stat].min(), data[stat].max()
            dict_stats_norm[stat] = (data[stat] - mini) / (maxi - mini)

        solver.Maximize(solver.Sum(x[i] * norm[i] for i in I for stat, norm in dict_stats_norm.items()))

        budget_constraint = solver.Add(solver.Sum(x[i] * data['Value'].iloc[i] for i in I) <= budget)

    for stat, pct in dict_stats.items():
        solver.Add(solver.Sum(x[i] * data.iloc[i][stat] for i in I) >= np.quantile(df_means[stat], pct) * solver.Sum(x[i] for i in I))

    solver.Add(solver.Sum(x[j] for j in idx_current) >= min_keep)

    solver.Add(solver.Sum(x[i] * data.iloc[i]['Age'] for i in I) <= age_limit * solver.Sum(x[i] for i in I))

    if starting11:
        solver.Add(solver.Sum(x[i] for i in I) == 11)

    bool_pos = [[int(data.iloc[i]['Position'] == all_positions[p]) for p in P] for i in I]

    for p in P:
        solver.Add(solver.Sum(x[i] * bool_pos[i][p] for i in I) <= sum(dict_formations[f][p] * z[f] for f in F))
    solver.Add(solver.Sum(z[f] for f in F) == 1)

    if formation != "Qualquer":
        solver.Add(z[formation] == 1)    

    status = solver.Solve()

    if status != pywraplp.Solver.OPTIMAL:
        print("It is impossible to build a team respecting such constraints")
        return data.iloc[[], :], 0.0, 0.0, ""

    rs = data.iloc[[i for i in range(len(data)) if abs(x[i].solution_value()) > 1e-12]].sort_values(by='Position', key = lambda x: x.map(sort_order))

    score = []
    for stat, pct in dict_stats.items():
        mini, maxi = data[stat].min(), data[stat].max()
        score.append(((rs[stat] - mini) / (maxi - mini)).mean())

    ret = rs[['Player', 'Squad', 'Position', 'Age', 'Value'] + list(dict_stats.keys())]
    ret.iloc[:, 4:] = ret.iloc[:, 4:].round(2)

    formation = dict_formations_rev[tuple([sum(rs['Position'] == pos) for pos in all_positions])]

    return ret, rs['Value'].sum(), round(np.mean(score), 4), formation, starters(rs, formation), {formation: 1}

def recommend_signings_multi_stage(team, data_orig, df_means, dict_stats, time_limit=60.0,
                                    age_limit=45, pct_keep=0.0, own_players_val=1.0, formation="", 
                                    budget=0.0, scenarios=100, max_players=1922, gap=0.01, 
                                    pred_method="Naïve"):

    data = data_orig.copy()

    I = list(range(len(data)))
    idx_current = [i for i in I if pct_keep > 0 and data.iloc[i]['Squad'] == team]
    S = [data.columns.get_loc(k) for k in dict_stats.keys()]
    C = list(range(scenarios))
    F = dict_formations.keys()
    P = list(range(len(all_positions)))

    data.iloc[idx_current].Value = (own_players_val * data.iloc[idx_current].Value).round().astype(int)

    solver = pywraplp.Solver.CreateSolver('GLOP')
    solver.SetTimeLimit(int(time_limit * 1000))

    x = {i: solver.BoolVar(f'x[{i}]') for i in I}
    y = {i: {c: solver.BoolVar(f'y[{i},{c}]') for c in C} for i in I}
    z = {f: {c: solver.BoolVar(f'z[{f},{c}]') for c in C} for f in F} # escolher o melhor esquema tático

    dict_stats_norm = {}
    for stat, pct in dict_stats.items():
        mini, maxi = data[stat].min(), data[stat].max()
        dict_stats_norm[stat] = (data[stat] - mini) / (maxi - mini)

    healthy = np.zeros((len(data), scenarios), dtype=bool)
    for i in range(len(data)):
        prob = data.iloc[i]['Injury_Prob']
        num_scen_healthy = int(round(scenarios * (1 - prob)))
        v = np.zeros(scenarios, dtype=bool)
        v[:num_scen_healthy] = True
        np.random.shuffle(v)
        healthy[i, :] = v
    healthy = healthy.astype(int)

    solver.Add(sum(x[i] * data.iloc[i]['Value'] for i in I) <= budget)

    for stat, pct in dict_stats.items():
        solver.Add(sum(y[i][c] * data.iloc[i][stat] for i in I for c in C) >= 
                   np.quantile(df_means[stat], pct) * sum(y[i][c] for i in I for c in C))

    pre_game_stats = np.zeros((len(I), len(S), scenarios))

    for k, stat in enumerate(dict_stats_norm.keys()):
        if pred_method == "Naïve":
            pre_game = np.hstack([np.repeat(data.iloc[i][stat], scenarios).reshape(-1, 1) for i in range(len(data))])
        elif pred_method == "Normal":
            pre_game = np.hstack([np.random.normal(data.iloc[i][stat], 2/3 * data[stat].std(), scenarios) 
                                   if data.iloc[i][stat] != 0 else np.zeros(scenarios) 
                                   for i in range(len(data))]).reshape(-1, scenarios).T
        elif pred_method == "Séries temporais":
            stat_ = {"PrgR_Receiving": "Prog_Receiving", "Succ_Take": "Succ_Dribbles"}.get(stat, stat)
            pos_ = np.array([dict_positions[pos] for pos in data['Position']])
            X = np.column_stack([np.ones(len(data)), data[stat].values, data['Age'].values, 
                                 data['Age'].values ** 2, pos_ == "FW", pos_ == "GK", pos_ == "MF"])
            coefs_vcov = df_regs[stat_].values
            eta = np.dot(X, coefs_vcov[:7])
            vc = coefs_vcov[7:].reshape(7, 7)
            vcovXnewT = np.dot(vc, X.T)
            stdeta = np.array([np.sqrt(np.dot(X[i, :], vcovXnewT[:, i])) for i in range(X.shape[0])])
            lower, upper = eta - 1.96 * stdeta, eta + 1.96 * stdeta
            dists = [np.random.normal(eta[i], (upper[i] - lower[i]) / 1.96, scenarios) for i in range(len(data))]
            pre_game = np.column_stack(dists)  # SxI (inverter?)
        
        pre_game_stats[:, k, :] = pre_game.T

    solver.Maximize(sum(y[i][c] * pre_game_stats[i, s, c] for i in I for c in C for s in range(len(dict_stats_norm))))

    solver.Add(sum(y[j][c] for j in idx_current for c in C) >= sum(pct_keep for c in C) * 11)

    solver.Add(sum(y[i][c] * data.iloc[i]['Age'] for i in I for c in C) <= age_limit * sum(y[i][c] for i in I for c in C))

    for c in C:
        solver.Add(sum(y[i][c] for i in I) == 11)

    bool_pos = np.array([[int(data.iloc[i]['Position'] == all_positions[p]) for p in P] for i in I])

    for p in P:
        for c in C:
            solver.Add(sum(y[i][c] * bool_pos[i, p] for i in I) == sum(dict_formations[f][p] * z[f][c] for f in F))

    for c in C:
        solver.Add(sum(z[f][c] for f in F) == 1)

    if formation != "Qualquer":
        for c in C:
            solver.Add(z[formation][c] == 1)

    for i in I:
        for c in C:
            solver.Add(y[i][c] <= x[i])
            solver.Add(y[i][c] <= healthy[i, c])

    for i in I:
        solver.Add(x[i] <= sum(y[i][c] for c in C))

    solver.Add(sum(x[i] for i in I) <= max_players)

    solver.Solve()

    if solver.Solve() != pywraplp.Solver.OPTIMAL:
        print("It is impossible to build a team respecting such constraints")
        return data.iloc[[], :], 0.0, 0.0, ""

    rs = data.loc[np.abs([x[i].solution_value() for i in I]) > 1e-12].sort_values(by='Position', key = lambda x: x.map(sort_order))
    rsy = [data.loc[np.abs([y[i][c].solution_value() for i in I]) > 1e-12].sort_values(by='Position', key = lambda x: x.map(sort_order)) for c in C]
    rs['Apps'] = [sum(x == p for x in np.concatenate([rsy[c]['Player'].values for c in C])) for p in rs['Player']]

    score = []
    for stat, pct in dict_stats.items():
        mini, maxi = data[stat].min(), data[stat].max()
        score.append(((rs[stat] - mini) / (maxi - mini)).mean())

    ret = rs[['Player', 'Squad', 'Position', 'Apps', 'Age', 'Value'] + list(dict_stats.keys())]
    ret.iloc[:, 4:] = ret.iloc[:, 4:].round(2)

    formations_count = {k: 0 for k in dict_formations.keys()}
    for y in rsy:
        position_counts = [sum(y['Position'] == pos) for pos in all_positions]
        formations_count[dict_formations_rev[tuple(position_counts)]] += 1

    main_formation = max(formations_count, key=formations_count.get)

    return ret, rs['Value'].sum(), round(np.mean(score), 4), main_formation, starters(rs, main_formation), formations_count

def recommended_signings(team: str, season: int, dict_stats, time_limit: float = 60.0,
                         age_limit = 45, pct_keep: float = 0.0, starting11: bool = False, 
                         own_players_val: float = 1.0, formation: str = "", budget: float = 0.0, 
                         scenarios: int = 100, max_players: int = 1922, gap: float = 0.01, 
                         foreigners: int = 4, healthy=None, pred_method: str = "Naïve"):
    
    data = pd.read_csv(f"dados/dados{season}.csv")
    data['Position'].replace({"midfield": "Centre-Back", "Second Striker": "Centre-Forward", 
                                "Right Midfield": "Right Winger", "Left Midfield": "Left Winger", 
                                "attack": "Left Winger"}, inplace=True)
    
    df_means = pd.read_csv(f"dados/medias{season}.csv")
    data = data[data['Mins_Per_90'] >= 5]
    
    if season >= 2023:
        data['Age'] = (pd.to_datetime(f"{season}-07-01") - pd.to_datetime(data['Birth'], format="%m/%d/%Y")).dt.days / 365
    else:
        data['Age'] = (pd.to_datetime(f"{season}-07-01") - pd.to_datetime(data['Birth'])).dt.days / 365
    
    data['Value'] = data['Value'] / 10**6

    if starting11:
        return recommend_signings_single_stage(team, data, df_means, dict_stats, time_limit=time_limit,
                                               age_limit=age_limit, min_keep=int(pct_keep * 11), 
                                               own_players_val=own_players_val, formation=formation, 
                                               budget=budget)
    else:
        prob_inj = pd.read_csv("dados/injury_prob.csv")
        dict_prob_inj = {row['Url']: row['Prob'] for _, row in prob_inj.iterrows()}
        data['Injury_Prob'] = [dict_prob_inj.get(row['Url'], prob_inj['Prob'].mode()[0]) for _, row in data.iterrows()]
        
        yell = 3 / df_means['CrdY'].quantile(0.5)
        red = 1 / df_means['CrdR'].quantile(0.5)
        data['Injury_Prob'] += (1/yell + 1/red)

        return recommend_signings_multi_stage(team, data, df_means, dict_stats, time_limit=time_limit,
                                              age_limit=age_limit, pct_keep=pct_keep, 
                                              own_players_val=own_players_val, formation=formation, 
                                              budget=budget, scenarios=scenarios, max_players=50, 
                                              gap=gap, pred_method=pred_method)

