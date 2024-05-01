import Pkg
Pkg.add(["DataFrames", "CSV", "JuMP", "Gurobi", "Statistics", "Dates", "Random", "StatsBase", "Distributions"])
using JuMP, Gurobi, CSV, DataFrames, Statistics, Dates, Random, StatsBase, Distributions

include("utils.jl")
function recommend_signings_single_stage(team::String, data_orig::DataFrame, df_means::DataFrame, dict_stats; time_limit::Float64 = 60.0,
                            age_limit = 45, min_keep::Int64 = 11, starting11::Bool = true, own_players_val::Float64 = 1.0, formation::String="", budget::Float64=0.0)

    data = deepcopy(data_orig)

    if formation == "Qualquer" # test all formations
        formation_results = [recommend_signings_single_stage(team, data_orig, df_means, dict_stats; time_limit = time_limit,
                                age_limit = age_limit, min_keep = min_keep, own_players_val = own_players_val,
                                formation=formation, budget=budget) for (formation,v) in dict_formations]
        val, idx = findmax(map(x -> x[3], formation_results))
        return formation_results[idx]
    end

    # Sets
    I = collect(1:size(data,1))
    idx_current = min_keep > 0 ? findall(x -> x in eachrow(lineup(team, data)), eachrow(data)) : Int64[]
    S = [findfirst(x -> x == k, names(df_means)) for k in keys(dict_stats)]

    data[idx_current, :Value] = Int64.(round.(own_players_val*data[idx_current, :Value]))

    model = Model(Gurobi.Optimizer)

    set_optimizer_attribute(model, "time_limit", time_limit)

    @variable(model, x[I], binary=true)

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

    positions = dict_formations[formation]
    for (i, qtd_pos) in enumerate(positions)
        pos_name = all_positions[i]
        bool_pos = [name == pos_name for name in data.Position]

        @constraint(model, sum(x[i]*bool_pos[i] for i in I) == qtd_pos)
    end

    optimize!(model)

    if termination_status(model) == MOI.INFEASIBLE
        @warn "It is impossible to build a team respecting such constraints"
        return data[[],:], 0.0, 0.0, ""
    end

    rs = sort(data[findall(x -> abs(x) > 10^(-12), JuMP.value.(x).data),:], [:Position], lt=position_sort)

    score = Float64[]
    for (stat, pct) in dict_stats
        mini, maxi = minimum(data[:,stat]), maximum(data[:,stat])
        push!(score, mean((rs[:,stat] .- mini) / (maxi - mini)))
    end

    ret = rs[:, vcat("Player", "Squad", "Position", "Age", "Value", [k for k in keys(dict_stats)])]
    ret[:,4:end] = round.(ret[:,4:end], digits=2)
    return ret, sum(rs.Value), round(mean(score), digits=4), formation, starters(rs, formation)
end

function recommend_signings_multi_stage(team::String, data_orig::DataFrame, df_means::DataFrame, dict_stats; time_limit::Float64 = 60.0,
    age_limit = 45, pct_keep::Float64 = 0., own_players_val::Float64 = 1.0, formation::String="", budget::Float64=0.0, scenarios::Int64=100, max_players::Int64=1922, gap::Float64=0.01, foreigners::Int64=4, healthy::Matrix, pred_method::String="Naïve")

    if formation == "Qualquer" # test all formations
        formation_results = [recommend_signings_multi_stage(team, data_orig, df_means, dict_stats; time_limit = time_limit,
                                age_limit = age_limit, pct_keep = pct_keep, own_players_val = own_players_val,
                                formation=formation, budget=budget, scenarios=scenarios, max_players=max_players, gap=gap, healthy=healthy, pred_method=pred_method) for (formation,v) in dict_formations]
        val, idx = findmax(map(x -> x[3], formation_results))
        return formation_results[idx]
    end

    data = deepcopy(data_orig)

    # Sets
    I = collect(1:size(data,1))
    idx_current = pct_keep > 0 ? findall(x -> x.Squad == "Monaco", eachrow(data)) : Int64[]
    # idx_foreigners = foreigners > 0 ? findall(x -> !(x.Nation in vcat(europe,africa)), eachrow(data)) : Int64[] # Ligue 1
    S = [findfirst(x -> x == k, names(df_means)) for k in keys(dict_stats)]
    C = collect(1:scenarios) # S?

    data[idx_current, :Value] = Int64.(round.(own_players_val*data[idx_current, :Value]))

    model = Model(Gurobi.Optimizer)

    set_time_limit_sec(model, time_limit)
    set_optimizer_attribute(model, "MIPGap", gap)

    @variable(model, x[I], binary=true) # contratação
    @variable(model, y[I,C], binary=true) # escalação
    
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
        @constraint(model, sum(y[i,c]*data[i,stat] for i in I, c in C) >= quantile(df_means[:,stat], pct) * sum(y[i,c] for i in I, c in C))
    end

    pre_game_stats = zeros(length(I), length(S), scenarios)

    for (k,stat) in enumerate([stat for stat in keys(dict_stats_normalized)])
        if pred_method == "Naïve"
            pre_game = hcat([repeat([el], scenarios) for el in data[:,stat]]...) # SxI (inverter?)
        elseif pred_method == "Normal"
            # if minimum(data[:,stat]) >= 0 # valores inteiros, mais realista, mas variação está muito grande
            #     pre_game = hcat([rand(Poisson(el), scenarios) for el in data[:,stat]]...) # SxI (inverter?)
            # else
                pre_game = hcat([el == 0 ? zeros(scenarios) : rand(Normal(el, 2/3*std(data[:,stat])), scenarios) for el in data[:,stat]]...) # SxI (inverter?)
            # end
        elseif pred_method == "Séries temporais"
            stat_ = stat == "PrgR_Receiving" ? "Prog_Receiving" : stat # generalizar com get(dict)
            stat_ = get(Dict("PrgR_Receiving" => "Prog_Receiving", "Succ_Take" => "Succ_Dribbles"), stat, stat)
            pos_ = [dict_positions[pos] for pos in data.Position]
            X = [ones(size(data,1)) data[:,stat] data.Age data.Age .^ 2 pos_ .== "FW" pos_ .== "GK" pos_ .== "MF"]
            coefs_vcov = df_regs[:,stat_]
            eta = X * coefs_vcov[1:7]
            vc = reshape(coefs_vcov[8:end], 7, 7)
            vcovXnewT = vc*X'
            stdeta = [sqrt(GLM.dot(view(X, i, :), view(vcovXnewT, :, i))) for i in axes(X,1)]
            lower, upper = eta .- 1.96*stdeta, eta .+ 1.96*stdeta
            dists = Normal.(eta,(eta-lower)/1.96)
            pre_game = hcat([rand(dist, scenarios) for dist in dists]...) # SxI (inverter?)
        end

        pre_game_stats[:,k,:] = permutedims(pre_game) 
    end

    @objective(model, Max, sum(y[i,c]*pre_game_stats[i,s,c] for i in I, c in C, s in 1:length(dict_stats_normalized)))

    @constraint(model, sum(y[j,c] for j in idx_current, c in C) >= sum(pct_keep for c in C) * 11)

    @constraint(model, max_age, sum(y[i,c]*data[i,:Age] for i in I, c in C) <= age_limit * sum(y[i,c] for i in I, c in C))

    @constraint(model, [c in C], sum(y[i,c] for i in I) == 11) # redundante com a restrição de posições, mas whatever

    positions = dict_formations[formation]
    for (i, qtd_pos) in enumerate(positions)
        pos_name = all_positions[i]
        bool_pos = [name == pos_name for name in data.Position]

        @constraint(model, [c in C], sum(y[i,c]*bool_pos[i] for i in I) == qtd_pos)
    end

    @constraint(model, [i in I, c in C], y[i,c] <= x[i]) # só pode escalar se tiver contratado
    @constraint(model, [i in I, c in C], y[i,c] <= healthy[i,c]) # só pode escalar quem está disponível

    @constraint(model, [i in I], x[i] <= sum(y[i,c] for c in C)) # só contrata se for de fato usar

    @constraint(model, sum(x[i] for i in I) <= max_players) # opcional, só pra ajudar no tempo computacional

    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
    #@warn "It is impossible to build a team respecting such constraints"
    return data[[],:], 0.0, 0.0, ""
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
    ret = rs[:, vcat("Player", "Squad", "Position", "Apps", "Age", "Value", [k for k in keys(dict_stats)])]
    ret[:,5:end] = round.(ret[:,5:end], digits=2)
    return ret, sum(rs.Value), round(mean(score), digits=4), formation, starters(rs, formation)

    # return rs, sum(rs.Value), mean(score), rsy, healthy[[idx(url) for url in rs.Url],:], pre_game_stats[[idx(url) for url in rs.Url],:,:]
end


function recommended_signings(team::String, season::Int64, dict_stats; time_limit::Float64 = 60.0,
    age_limit = 45, pct_keep::Float64 = 0., starting11::Bool = false, own_players_val::Float64 = 1.0, formation::String="", budget::Float64=0.0, scenarios::Int64=100, max_players::Int64=1922, gap::Float64=0.01, foreigners::Int64=4, healthy::Matrix=Matrix(undef, 1, 1), pred_method::String="Naïve")
    
    data = CSV.read("dados/dados$season.csv", DataFrame)
    replace!(data.Position, "midfield" => "Centre-Back", "Second Striker" => "Centre-Forward", "Right Midfield" => "Right Winger", "Left Midfield" => "Left Winger", "attack" => "Left Winger")
    df_means = CSV.read("dados/medias$season.csv", DataFrame)
    filter!(x -> x.Mins_Per_90 >= 5, data)
    data.Age = season >= 2023 ? map(x -> x.value/365, Date(season,7) .- Date.(data.Birth, "mm/dd/yyyy")) : map(x -> x.value/365, Date(season,7) .- data.Birth)
    data.Value = data.Value ./ 10^6

    if starting11
        return recommend_signings_single_stage(team, data, df_means, dict_stats; time_limit = time_limit,
            age_limit = age_limit, min_keep = Int64(pct_keep*11), own_players_val = own_players_val, formation=formation, budget=budget)
    else
        prob_inj = CSV.read("dados/injury_prob.csv", DataFrame)
        dict_prob_inj = Dict([row.Url => row.Prob for row in eachrow(prob_inj)])
        data[!, :Injury_Prob] = [get(dict_prob_inj, row.Url, StatsBase.mode(prob_inj.Prob)) for row in eachrow(data)]
        yell, red = 3/quantile(df_means[:,"CrdY"],.5), 1/quantile(df_means[:,"CrdR"],.5)
        data.Injury_Prob = data.Injury_Prob .+ (1/yell + 1/red)

        return recommend_signings_multi_stage(team, data, df_means, dict_stats; time_limit = time_limit,
            age_limit = age_limit, pct_keep = pct_keep, own_players_val = own_players_val, formation=formation, budget=budget, scenarios=scenarios, max_players=50, gap=gap, foreigners=foreigners, healthy=healthy, pred_method=pred_method)
    end
end
