import pandas as pd

def lineup(team: str, df: pd.DataFrame) -> pd.DataFrame:
    squad = df[df['Squad'] == team]
    return squad.nlargest(11, 'Mins_Per_90')

def mean_stat(team: str, stat: str, df: pd.DataFrame) -> float:
    lup = lineup(team, df)
    return lup[stat].mean()

def all_lineups(df: pd.DataFrame) -> dict:
    d = {}
    for team in df['Squad'].unique():
        d[str(team)] = lineup(str(team), df)
    return d

def create_input(path: str) -> tuple:
    df = pd.read_csv(path)
    team = str(df.iloc[0, 1])
    formation = str(df.iloc[1, 1])
    budget = float(df.iloc[2, 1])
    time = float(df.iloc[3, 1])
    age = float(df.iloc[4, 1])
    pct = float(df.iloc[5, 1])
    starting = bool(int(df.iloc[6, 1]))
    own_val = float(df.iloc[7, 1])

    d = {str(df.iloc[i, 0]): float(df.iloc[i, 1]) for i in range(8, len(df))}

    return d, team, formation, budget, time, age, pct, starting, own_val

all_positions = ["Goalkeeper", "Right-Back", "Centre-Back", "Left-Back", "Defensive Midfield", "Central Midfield", "Attacking Midfield", "Right Winger", "Centre-Forward", "Left Winger"]

sort_order = {"Goalkeeper": 0, "Right-Back": 1, "Centre-Back": 2, "Left-Back": 3, "Defensive Midfield": 4, "Central Midfield": 5, "Attacking Midfield": 6, "Right Winger": 7, "Centre-Forward": 8, "Left Winger": 9}

dict_formations = {
    "4-4-2": [1, 1, 2, 1, 1, 1, 0, 1, 2, 1],
    "4-1-4-1": [1, 1, 2, 1, 1, 2, 0, 1, 1, 1],
    "4-3-3": [1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
    "3-5-2": [1, 1, 3, 1, 1, 1, 1, 0, 2, 0],
    "5-4-1": [1, 1, 3, 1, 1, 1, 0, 1, 1, 1],
    "3-4-3": [1, 1, 3, 1, 1, 1, 0, 1, 1, 1]
}

assert all(sum(v) == 11 for v in dict_formations.values())

def starters(roster: pd.DataFrame, formation: str) -> pd.DataFrame:
    formation = dict_formations[formation]
    starters = pd.DataFrame()
    for pos, n in zip(all_positions, formation):
        options = roster[roster['Position'] == pos]
        options = options.sort_values(by='Apps', ascending=False)
        starters = pd.concat([starters, options.head(int(n))])
    return starters

DEFAULT_STATS = ["Tkl+Int", "Clr", "SCA_SCA", "PrgDist_Total", "xA", "Succ_Take", "PrgDist_Carries", "PrgR_Receiving", "Gls", "PSxG+_per__minus__Expected"]

europe = ["GER", "ESP", "DEN", "FRA", "POR", "ITA", "SUI", "ENG", "NED", "ROU", "CZE", "SVN", "CRO", "POL", "HUN", "BEL", "SCO", "WAL", "MNE", "SWE", "SRB", "RUS", "NOR", "KVX", "ALB", "GRE", "AUT", "BIH", "UKR", "MKD", "IRL", "FIN", "CYP", "BUL", "NIR", "GEO", "LUX", "SVK", "MAD"]
africa = ["SEN", "MAR", "CIV", "MLI", "GHA", "NGA", "CMR", "ALG", "EQG", "GAM", "TUN", "GNB", "CGO", "BFA", "BEN", "TOG", "COD", "CTA", "GUI", "GAB", "ZIM", "ANG", "ZAM", "EGY", "COM", "CPV", "MOZ", "SLE", "BDI"]
america = ["BRA", "ARG", "CHI", "MEX", "URU", "CAN", "HON", "USA", "PAR", "COL", "ECU", "MTQ", "CRC", "GLP", "VEN", "JAM", "PER", "GRN", "SUR", "GUF"]
asia = ["JPN", "TUR", "AUS", "UZB", "PHI", "ARM", "KOR", "NZL", "ISR", "IRN"]

df_regs = pd.read_csv("dados/regs.csv")
dict_positions = {
    "Goalkeeper": "GK",
    "Left-Back": "DF",
    "Centre-Back": "DF",
    "Right-Back": "DF",
    "Defensive Midfield": "MF",
    "Central Midfield": "MF",
    "Attacking Midfield": "MF",
    "Right Winger": "FW",
    "Left Winger": "FW",
    "Centre-Forward": "FW"
}