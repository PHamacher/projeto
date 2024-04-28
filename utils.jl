using Statistics

function lineup(team::String, df::DataFrame)
    squad = filter(x -> x.Squad == team, df)
    return sort(squad, :Mins_Per_90, rev=true)[1:11,:]
end

function mean_stat(team::String, stat::String, df::DataFrame)
    lup = lineup(team, df)
    return mean(lup[:,stat])
end

function all_lineups(df::DataFrame)
    d = Dict{String, DataFrame}()
    for team in unique(df.Squad)
        team = String(team)
        d[team] = lineup(team, df)
    end
    return d
end

function create_input(path::String)
    df = CSV.read(path, DataFrame)
    team = String(df[1,2])
    formation = String(df[2,2])
    budget = parse(Float64, df[3,2])
    time = parse(Float64, df[4,2])
    age = parse(Float64, df[5,2])
    pct = parse(Float64, df[6,2])
    starting = Bool(parse(Int64, df[7,2]))
    own_val = parse(Float64, df[8,2])

    d = Dict([String(df[i,1]) => parse(Float64, df[i,2]) for i in 9:size(df,1)])

    return d, team, formation, budget, time, age, pct, starting, own_val
end

all_positions = ["Goalkeeper", "Right-Back", "Centre-Back", "Left-Back", "Defensive Midfield", "Central Midfield", "Attacking Midfield", "Right Winger", "Centre-Forward", "Left Winger"]

position_sort(x,y) = findfirst(el -> el == x, all_positions) < findfirst(el -> el == y, all_positions)

dict_formations = Dict{String, Vector{Float64}}() # [gks, rbs, cbs, lbs, cdms, cms, cams, rws, sts, lws]
dict_formations["4-4-2"] = [1,1,2,1,1,1,0,1,2,1]
dict_formations["4-1-4-1"] = [1,1,2,1,1,2,0,1,1,1]
dict_formations["4-3-3"] = [1,1,2,1,1,1,1,1,1,1]
dict_formations["3-5-2"] = [1,1,3,1,1,1,1,0,2,0]
dict_formations["5-4-1"] = [1,1,3,1,1,1,0,1,1,1]
dict_formations["3-4-3"] = [1,1,3,1,1,1,0,1,1,1]


[@assert(sum(v) == 11) for (k,v) in dict_formations];

function starters(roster::DataFrame, formation::String)
    formation = dict_formations[formation]
    starters = DataFrame()
    for (pos, n) in zip(all_positions, formation)
        options = filter(x -> x.Position == pos, roster)
        sort!(options, :Apps, rev=true)
        starters = vcat(starters, options[1:Int64(n),:])
    end
    return starters
end

# const DEFAULT_STATS = ["Tkl+Int", "Succ_Pressures", "SCA_SCA", "PrgDist_Total", "xA", "Succ_Dribbles", "PrgDist_Carries", "Prog_Receiving", "np:G_minus_xG_Expected"] # 2022
const DEFAULT_STATS = ["Tkl+Int", "Clr", "SCA_SCA", "PrgDist_Total", "xA", "Succ_Take", "PrgDist_Carries", "PrgR_Receiving", "Gls", "PSxG+_per__minus__Expected"]

# mean_stat("Sevilla", "Tkl_Tackles", teste)
# mean_stat("Sevilla", "Value", teste)


# d = all_lineups(teste)

# df_means = DataFrame(vcat([hcat(team, mean(Matrix(d[team][:,8:end-5]), dims=1), mean(d[team].Value)) for (team, lup) in d]...),
#             vcat("Team", names(teste)[8:end-5], "Value"))

# CSV.write("C:\\Users\\admin\\OneDrive\\Documentos antigo\\Projetos_jl/Matérias PUC/23.1/prog mat/projeto/medias2022.csv", df_means)


europe = ["GER", "ESP", "DEN", "FRA", "POR", "ITA", "SUI", "ENG", "NED", "ROU", "CZE", "SVN", "CRO", "POL", "HUN", "BEL", "SCO", "WAL", "MNE", "SWE", "SRB", "RUS", "NOR", "KVX", "ALB", "GRE", "AUT", "BIH", "UKR", "MKD", "IRL", "FIN", "CYP", "BUL", "NIR", "GEO", "LUX", "SVK", "MAD"]
africa = ["SEN", "MAR", "CIV", "MLI", "GHA", "NGA", "CMR", "ALG", "EQG", "GAM", "TUN", "GNB", "CGO", "BFA", "BEN", "TOG", "COD", "CTA", "GUI", "GAB", "ZIM", "ANG", "ZAM", "EGY", "COM", "CPV", "MOZ", "SLE", "BDI"]
america = ["BRA", "ARG", "CHI", "MEX", "URU", "CAN", "HON", "USA", "PAR", "COL", "ECU", "MTQ", "CRC", "GLP", "VEN", "JAM", "PER", "GRN", "SUR", "GUF"]
asia = ["JPN", "TUR", "AUS", "UZB", "PHI", "ARM", "KOR", "NZL", "ISR", "IRN"]
