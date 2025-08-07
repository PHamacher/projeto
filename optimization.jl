
using JuMP, Gurobi, CSV, DataFrames, Statistics, Dates, Random

# include("Matérias PUC\\23.1\\prog mat\\projeto\\utils.jl")
include("utils.jl")
include("forecasting.jl")
function recommend_signings(team::String, data_orig::DataFrame, df_means::DataFrame, dict_stats::Dict{String, Float64}; time_limit::Float64 = 60.0,
                            age_limit = 45, min_keep::Int64 = 11, starting11::Bool = true, own_players_val::Float64 = 1.0, formation::String="", budget::Float64=0.0)

    data = deepcopy(data_orig)
    # budget = budget*10^6 # millions of Euros

    # Sets
    I = collect(1:size(data,1))
    idx_current = min_keep > 0 ? findall(x -> x in eachrow(lineup(team, data)), eachrow(data)) : Int64[]
    S = [findfirst(x -> x == k, names(df_means)) for k in keys(dict_stats)]
    F = keys(dict_formations)
    P = collect(1:length(all_positions))

    data[idx_current, :Value] = Int64.(round.(own_players_val*data[idx_current, :Value]))

    model = Model(Gurobi.Optimizer)

    set_optimizer_attribute(model, "time_limit", time_limit)

    @variable(model, x[I], binary=true)
    @variable(model, Z[F], Bin) # escolher o melhor esquema tático

    if budget == 0 # FO: minimizar custo
        @objective(model, Min, sum(x[i]*data.Value[i] for i in I))

    else # FO: maximizar stats dado budget
        dict_stats_normalized = Dict{String, Vector{Float64}}()
        for (stat, pct) in dict_stats
            mini, maxi = minimum(data[:,stat]), maximum(data[:,stat])
            dict_stats_normalized[stat] = (data[:,stat] .- mini) / (maxi - mini) # normalização min-max
        end

        @objective(model, Max, sum(x[i]*norm[i] for i in I, (stat,norm) in dict_stats_normalized))

        @constraint(model, budget_constraint, sum(x[i]*data.Value[i] for i in I) <= budget)
    end

    for (stat, pct) in dict_stats # deixar apenas no caso de budget == 0?
        @constraint(model, sum(x[i]*data[i,stat] for i in I) >= quantile(df_means[:,stat], pct) * sum(x[i] for i in I)) # assuming initial starters are maintained
    end

    # @constraint(model, sum(x[j] for j in idx_current) == round(pct_keep*length(idx_current)))
    @constraint(model, sum(x[j] for j in idx_current) >= min_keep)

    @constraint(model, max_age, sum(x[i]*data[i,:Age] for i in I) <= age_limit * sum(x[i] for i in I))

    # pct_keep > 0.0 || @constraint(model, sum(x[i] for i in I) >= 1)

    starting11 && @constraint(model, sum(x[i] for i in I) == 11)

    bool_pos = [data[i,:Position] == all_positions[p] for i in I, p in P]
    @constraint(model, [p in P], sum(x[i]*bool_pos[i,p] for i in I) == sum(dict_formations[f][p] * Z[f] for f in F))
    @constraint(model, sum(Z[f] for f in F) == 1) # escolher apenas uma formação
    formation != "" && @constraint(model, Z[formation] == 1)

    optimize!(model)

    if termination_status(model) == MOI.INFEASIBLE
        @warn "It is impossible to build a team respecting such constraints"
        return data[[],:], 0.0, 0.0
    end

    rs = sort(data[findall(x -> abs(x) > 10^(-12), JuMP.value.(x).data),:], [:Position], lt=position_sort)

    score = Float64[]
    for (stat, pct) in dict_stats
        mini, maxi = minimum(data[:,stat]), maximum(data[:,stat])
        push!(score, mean((rs[:,stat] .- mini) / (maxi - mini)))
    end

    return rs, sum(rs.Value) / 10^6, mean(score)
end

# Parameters
season = 2023
# data = CSV.read("Matérias PUC\\23.1\\prog mat\\projeto\\dados/dados2022.csv", DataFrame)
data = CSV.read("dados/dados$season.csv", DataFrame)
replace!(data.Position, "midfield" => "Centre-Back", "Second Striker" => "Centre-Forward", "Right Midfield" => "Right Winger", "Left Midfield" => "Left Winger", "attack" => "Left Winger")
# df_means = CSV.read("Matérias PUC\\23.1\\prog mat\\projeto\\dados/medias2022.csv", DataFrame)
df_means = CSV.read("dados/medias$season.csv", DataFrame)
filter!(x -> x.Mins_Per_90 >= 5, data)
data.Age = season >= 2023 ? map(x -> x.value/365, Date(season,7) .- Date.(data.Birth, "mm/dd/yyyy")) : map(x -> x.value/365, Date(season,7) .- data.Birth)
data.Value = data.Value ./ 10^6

# dict_stats, team, formation, budget, time_limit, age, pct, starting, own_val = create_input("Matérias PUC\\23.1\\prog mat\\projeto\\dados/input.csv")
dict_stats, team, formation, budget, time_limit, age, keep, starting, own_val = create_input("dados/input.csv")
keep = Int64(keep)
rs, spent, score = recommend_signings(team, data, df_means, dict_stats; time_limit = time_limit, age_limit = age, min_keep = keep, starting11 = starting,
                        own_players_val = own_val, formation = formation, budget = budget)
rs[:, vcat("Player", "Squad", [k for k in keys(dict_stats)], "Position", "Age", "Value")]
filter(x -> x.Squad != team, rs)[:, vcat("Player", "Squad", [k for k in keys(dict_stats)], "Age", "Value")] # bought players
filter(x -> !(x.Player in rs.Player), lineup(team,data))[:, vcat("Player", "Squad", [k for k in keys(dict_stats)], "Position", "Age", "Value")] # sold players

# ---------

# dict_stats_all = Dict([name => .5 for name in names(df_means)[2:end-1]])



dict_stats_defaults1 = Dict([name => .1 for name in DEFAULT_STATS])
rs1, spent1, score1 = recommend_signings(team, data, df_means, dict_stats_defaults1; min_keep = 0, time_limit = 60.0*1, formation = "4-3-3", budget = sum(sort(df_means.Value)[end-10:end]))
rs1[:, vcat("Player", [k for k in keys(dict_stats_defaults1)], "Position", "Age", "Value")]

dict_stats_defaults2 = Dict([name => .975 for name in DEFAULT_STATS])
rs2, spent2, score2 = recommend_signings(team, data, df_means, dict_stats_defaults2; min_keep = 0, time_limit = 60.0*5, formation = "4-3-3")
rs2[:, vcat("Player", [k for k in keys(dict_stats_defaults2)], "Position", "Age", "Value")]

dict_stats_defaults3 = Dict([name => .975 for name in DEFAULT_STATS])
rs3, spent3, score3 = recommend_signings(team, data, df_means, dict_stats_defaults3; min_keep = 0, time_limit = 60.0*1, formation = "5-4-1", budget = sum(sort(df_means.Value)[end-10:end]))
rs3[:, vcat("Player", [k for k in keys(dict_stats_defaults3)], "Position", "Age", "Value")]


DataFrame(hcat(DEFAULT_STATS,[quantile(df_means[:,name], .975) for name in DEFAULT_STATS], transpose(mean(Matrix(rs[:,DEFAULT_STATS]), dims=1)), transpose(mean(Matrix(rs2[:,DEFAULT_STATS]), dims=1))), [:Stat, :Quantile, :Max_Score, :Min_Cost])


[(stat,mean((rs[:,stat] .- minimum(data[:,stat])) / (maximum(data[:,stat]) - minimum(data[:,stat])))) for stat in keys(dict_stats)]


rs_budget, spent_budget, score_budget = recommend_signings(team, data, df_means, dict_stats_defaults1; min_keep = 0, time_limit = 60.0*1, formation = "4-3-3", budget = 20.0, age_limit = 27)
rs_budget[:, vcat("Player", [k for k in keys(dict_stats_defaults1)], "Position", "Age", "Value")]


best_team = Dict{String, Dict}()
for formation in keys(dict_formations)
    rs1, spent1, score1 = recommend_signings(team, data, df_means, dict_stats_defaults1; min_keep = 0, time_limit = 60.0*1, formation = formation, budget = sum(sort(df_means.Value)[end-10:end]))
    rs2, spent2, score2 = recommend_signings(team, data, df_means, dict_stats_defaults2; min_keep = 0, time_limit = 60.0*5, formation = formation)
    rs3, spent3, score3 = recommend_signings(team, data, df_means, dict_stats_defaults3; min_keep = 0, time_limit = 60.0*1, formation = formation, budget = sum(sort(df_means.Value)[end-10:end]))
    best_team[formation] = Dict("rs" => [rs1,rs2,rs3], "spent" => [spent1,spent2,spent3], "score" => [score1,score2,score3])
end





# ----------------------
df=innerjoin(df2022, df2023, on="Url", makeunique=true)

idx2022 = setdiff(1:159, vcat(1, 17:22, 83, 84, 98:105))
idx2023 = setdiff(1:196, vcat(40:78, 109))
xAG
89:96

df_stab = rand(0,2)
for (i,name) in enumerate(names(df))
    if "$(name)_1" in names(df) && eltype.(eachcol(df))[i] in [Float64, Int64]
        # println(name, " = ", round(cor(df[:,name], df[:,"$(name)_1"]), digits=2))
        df_stab = vcat(df_stab, permutedims([name, cor(df[:,name], df[:,"$(name)_1"])]))
    end
end

df_stab = DataFrame(df_stab, [:Name, :Correlation])


sort(DataFrame(vcat([hcat(names(data)[8:end-5][i], cor(col, groupby(data, :Position)[7].Value)) for (i,col) in enumerate(eachcol(groupby(data, :Position)[7][:,8:end-5]))]...), [:Name, :Correlation]), :Correlation, rev=true)[41:69,:]


gk22 = CSV.read("C:/Users/admin/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/2022/big5_player_keepers.csv", DataFrame)
gk22_adv = CSV.read("C:/Users/admin/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/2022/big5_player_keepers_adv.csv", DataFrame)
filter!(x -> x.Season_End_Year == 2022, gk22)
filter!(x -> x.Season_End_Year == 2022, gk22_adv)
df_gk = innerjoin(gk22, gk22_adv, df2023, on="Url", makeunique=true)

df_stab_gk = rand(0,2)
for name in names(df_gk)[11:end]
    if "$(name)_1" in names(df_gk)
        col = isa(df_gk[:,name], Vector{Float64}) ? df_gk[:,name] : map(x -> x == "NA" ? 0 : parse(Float64, x), df_gk[:,name])
        df_stab_gk = vcat(df_stab_gk, permutedims([name, cor(col, df_gk[:,"$(name)_1"])]))
    end
end

df_stab_gk = DataFrame(df_stab_gk, [:Name, :Correlation])



# =========== 2o estágio ===========
using StatsBase, Distributions
prob_inj = CSV.read("dados/injury_prob.csv", DataFrame)
dict_prob_inj = Dict([row.Url => row.Prob for row in eachrow(prob_inj)])
data[!, :Injury_Prob] = [get(dict_prob_inj, row.Url, StatsBase.mode(prob_inj.Prob)) for row in eachrow(data)]
yell, red = 3/quantile(df_means[:,"CrdY"],.5), 1/quantile(df_means[:,"CrdR"],.5)
data.Injury_Prob = data.Injury_Prob .+ (1/yell + 1/red)


# stats in-game só são conhecidos pros titulares, pros reservas tb são expectativas -> como modelar?
# dentro do modelo, criar known_stats = [y[i,s]==1 ? stats_in_game[i,s] : stats_pre_game[i,s] for i in 1:I, s in 1:S] # revisar código

# crio S2 cenários de 3o estágio pra CADA cenário de 2o estágio ou só 1 pra cada??
# stats_in_game teria 3 dimensões: I, S e S2

# passo pra FO no 2o estágio a expectativa ou já o realizado in-game? (pergunta análoga pros substitutos no 3o estágio)

# adicionar: expectativa do treinador vs desempenho real

# polivalência -> PR no worldfootballR ou fazer alguma lógica com base de dados do Fifa
# add 3o estágio
# normalizar score?

# comparações: com polivalência e sem, com 3o estágio e sem, comparar com time real, com/sem limite de idade
# análises de sensibilidade: probabilidade lesão, esquemas táticos

function recommend_signings_multi_stage(team::String, data_orig::DataFrame, df_means::DataFrame, dict_stats::Dict{String, Float64}; time_limit::Float64 = 60.0,
    age_limit = 45, pct_keep::Float64 = 0., own_players_val::Float64 = 1.0, formation::String="", budget::Float64=0.0, scenarios::Int64=100, max_players::Int64=1922, gap::Float64=0.01, foreigners::Int64=4)

    data = deepcopy(data_orig)
    # budget = budget*10^6 # millions of Euros

    # Sets
    I = collect(1:size(data,1))
    # idx_current = pct_keep > 0 ? findall(x -> x in eachrow(lineup(team, data)), eachrow(data)) : Int64[]
    idx_current = pct_keep > 0 ? findall(x -> x.Squad == "Monaco", eachrow(data)) : Int64[]
    idx_foreigners = foreigners > 0 ? findall(x -> !(x.Nation in vcat(europe,africa)), eachrow(data)) : Int64[] # Ligue 1
    S = [findfirst(x -> x == k, names(df_means)) for k in keys(dict_stats)]
    C = collect(1:scenarios) # S?
    F = keys(dict_formations)
    P = collect(1:length(all_positions))

    data[idx_current, :Value] = Int64.(round.(own_players_val*data[idx_current, :Value]))
    # data[!, :Value] = data.Value ./ 10^6 # millions of Euros

    model = Model(Gurobi.Optimizer)

    # set_optimizer_attribute(model, "seconds", time_limit)
    set_time_limit_sec(model, time_limit)
    set_optimizer_attribute(model, "MIPGap", gap)

    @variable(model, x[I], binary=true) # contratação
    @variable(model, y[I,C], binary=true) # escalação
    # @variable(model, z[I,C,]) # falta um terceiro índice
    @variable(model, Z[F,C], Bin) # escolher o melhor esquema tático

    
    dict_stats_normalized = Dict{String, Vector{Float64}}()
    for (stat, pct) in dict_stats
        mini, maxi = minimum(data[:,stat]), maximum(data[:,stat])
        dict_stats_normalized[stat] = (data[:,stat] .- mini) / (maxi - mini) # normalização min-max
    end

    # Cenários
    Random.seed!(21)
    healthy = rand(size(data,1), scenarios) .> data.Injury_Prob
    healthy = BitMatrix(zeros(size(data,1), scenarios))
    for i in 1:size(data,1)
        prob = data[i,:Injury_Prob]
        num_scen_healthy = Int64(round(scenarios*(1-prob)))
        v = BitArray(zeros(scenarios))
        v[1:num_scen_healthy] = ones(num_scen_healthy)
        healthy[i,:] = shuffle(v)
    end

    @constraint(model, budget_constraint, sum(x[i]*data.Value[i] for i in I) <= budget)

    for (stat, pct) in dict_stats # fazer em y e em z?
        # @constraint(model, [c in C], sum(y[i,c]*data[i,stat] for i in I) >= quantile(df_means[:,stat], pct) * sum(y[i,c] for i in I))
        # @constraint(model, sum(x[i]*data[i,stat] for i in I) >= quantile(df_means[:,stat], pct) * sum(x[i] for i in I))
        @constraint(model, sum(y[i,c]*data[i,stat] for i in I, c in C) >= quantile(df_means[:,stat], pct) * sum(y[i,c] for i in I, c in C))
    end

    pre_game_stats = zeros(length(I), length(S), scenarios)
    # in_game_stats = zeros(length(I), length(S), scenarios)

    for (k,stat) in enumerate([stat for stat in keys(dict_stats_normalized)])
        pre_game = hcat([repeat([el], scenarios) for el in data[:,stat]]...) # SxI (inverter?)

        # stat_ = stat == "PrgR_Receiving" ? "Prog_Receiving" : stat # generalizar com get(dict)
        # stat_ = get(Dict("PrgR_Receiving" => "Prog_Receiving", "Succ_Take" => "Succ_Dribbles"), stat, stat)
        # X = DataFrame(Lag = data[:,stat], Age = data.Age, Pos = [dict_positions[pos] for pos in data.Position])
        # bounds = predict(dict_reg[stat_], X, interval = :confidence, level = .95)
        # dists = Normal.(bounds.prediction,(bounds.prediction-bounds.lower)/1.96)
        # pre_game = hcat([rand(dist, scenarios) for dist in dists]...) # SxI (inverter?)

        # pre_game = hcat([max.(0,rand(Normal(el,3/8), scenarios)) for el in data[:,stat]]...) # SxI (inverter?)
        # in_game = [rand(Normal(pre,3/4),1)[1] for pre in pre_game]

        pre_game_stats[:,k,:] = permutedims(pre_game) 
        # in_game_stats[:,k,:] = permutedims(in_game) 
    end

    @objective(model, Max, sum(y[i,c]*pre_game_stats[i,s,c] for i in I, c in C, s in 1:length(dict_stats_normalized)))

    # @constraint(model, sum(x[j] for j in idx_current) >= min_keep)
    @constraint(model, sum(y[j,c] for j in idx_current, c in C) >= sum(pct_keep for c in C) * 11)

    # @constraint(model, [c in C], sum(y[k,c] for k in idx_foreigners) <= foreigners)

    # @constraint(model, max_age, sum(x[i]*data[i,:Age] for i in I) <= age_limit * sum(x[i] for i in I)) # fazer do time titular tb?
    @constraint(model, max_age, sum(y[i,c]*data[i,:Age] for i in I, c in C) <= age_limit * sum(y[i,c] for i in I, c in C))

    @constraint(model, [c in C], sum(y[i,c] for i in I) == 11) # redundante com a restrição de posições, mas whatever

    bool_pos = [data[i,:Position] == all_positions[p] for i in I, p in P]
    @constraint(model, [p in P, c in C], sum(y[i,c]*bool_pos[i,p] for i in I) == sum(dict_formations[f][p] * Z[f,c] for f in F))
    @constraint(model, [c in C], sum(Z[f,c] for f in F) == 1) # escolher apenas uma formação
    formation != "" && @constraint(model, [c in C], Z[formation,c] == 1)

    @constraint(model, [i in I, c in C], y[i,c] <= x[i]) # só pode escalar se tiver contratado
    @constraint(model, [i in I, c in C], y[i,c] <= healthy[i,c]) # só pode escalar quem está disponível

    @constraint(model, [i in I], x[i] <= sum(y[i,c] for c in C)) # só contrata se for de fato usar

    @constraint(model, sum(x[i] for i in I) <= max_players) # opcional, só pra ajudar no tempo computacional

    optimize!(model)

    if termination_status(model) == MOI.INFEASIBLE
    @warn "It is impossible to build a team respecting such constraints"
    return data[[],:], 0.0, 0.0
    end

    rs = sort(data[findall(x -> abs(x) > 10^(-12), JuMP.value.(x).data),:], [:Position], lt=position_sort)
    rsy = [sort(data[findall(x -> abs(x) > 10^(-12), JuMP.value.(y).data[:,c]),:], [:Position], lt=position_sort) for c in C]
    rs[!, :Apps] = [count(x->x==p, vcat(map(x->x.Player, rsy)...)) for p in rs.Player]

    score = Float64[]
    for (stat, pct) in dict_stats
        mini, maxi = minimum(data[:,stat]), maximum(data[:,stat])
        push!(score, mean((rs[:,stat] .- mini) / (maxi - mini)))
    end

    idx(url) = findfirst(x -> x.Url == url, eachrow(data))
    return rs, sum(rs.Value), mean(score), rsy, healthy[[idx(url) for url in rs.Url],:], pre_game_stats[[idx(url) for url in rs.Url],:,:]
end

dict_stats, team, formation, budget, time_limit, age, keep, starting, own_val = create_input("dados/input monaco.csv")
# keep = Int64(keep)

@time rs, spent, obj, y, health, stats = recommend_signings_multi_stage(team, data, df_means, dict_stats; time_limit = time_limit, age_limit = age, pct_keep = keep,
                        own_players_val = own_val, formation = formation, budget = budget, scenarios = 10, max_players=50, gap=0.05);

rs[:, vcat("Player", "Squad", [k for k in keys(dict_stats)], "Position", "Age", "Value")]
rs[:,[:Player,:Position,:Age,:Value,:Apps]]
filter(x->x.Apps>=50, rs[:,[:Player,:Position,:Age,:Squad,:Value,:Apps]])
filter(x->x.Apps>=50, rs)[:,vcat("Player", [k for k in keys(dict_stats)], "Position", "Age", "Value")]



# dict = Dict()
scatter()
for formation in ["5-3-2", "5-4-1", "4-1-4-1", "4-3-3", "4-4-2"]
    rs, spent, score, y, health = recommend_signings_multi_stage(team, data, df_means, dict_stats; time_limit = time_limit, age_limit = age, min_keep = keep,
    own_players_val = own_val, formation = formation, budget = budget, scenarios = 100, max_players=50);
    scatter!([spent], [score], xlabel = "millions of €", ylabel = "Score", label = formation)
    # dict[formation] = [spent, score]
end
scatter!()

scatter(map(x->x[1], [v for v in values(dict)]), map(x->x[2], [v for v in values(dict)]), xlabel = "millions of €", ylabel = "Score", label = [k for k in keys(dict)])



dict_stats_defaults3 = Dict([name => .95 for name in DEFAULT_STATS])
rs, spent, score, y, health, stats = recommend_signings_multi_stage(team, data, df_means, dict_stats_defaults3; pct_keep = 0., time_limit = 300.0, formation = "3-4-3", budget = sum(sort(data.Value)[end-30:end]), scenarios = 100, gap=0.01)
rs[:, vcat("Player", [k for k in keys(dict_stats_defaults3)], "Position", "Age", "Value")]

all_forms = [recommend_signings_multi_stage(team, data, df_means, dict_stats_defaults3; pct_keep = 0., time_limit = 60.0*5, formation = form, budget = sum(sort(data.Value)[end-50:end]), scenarios = 100, gap=0.01) for form in keys(dic_formações)]



data[!,:Url_fb] = [fb_tm[x] for x in data.Url]

int = intersect(forecasted_players, data.Url_fb)
filter!(x -> x.Url_fb in forecasted_players, data)
idx_in = findall(x -> x in data.Url_fb, forecasted_players)
ordem = sortperm(forecasted_players[idx_in])
# ordem = sortperm(forecasted_players[findall(x -> x in data.Url_fb, forecasted_players)])
sort!(data, :Url_fb)

dict_stats_defaults3 = Dict([name => .05 for name in DEFAULT_STATS])
rs_mc = recommend_signings_multi_stage(team, data, df_means, dict_stats_defaults3; pct_keep = 0., time_limit = 300.0, formation = "3-4-3", budget = sum(sort(data.Value)[end-30:end]), scenarios = 100, healthy = [el < rand() for el in mc[idx_in,:][ordem,:]])

