using Plots, Colors, StatsPlots

scatter(df_means.Value, df_means.SCA_SCA, series_annotations = text.(df_means.Team, 8, :bottom), legend = false)

wins = [filter(x->x.Season_End_Year==2022 && x.Squad == team && x.Team_or_Opponent=="team", df_teams).W[1] for team in df_means.Team]
draws = [filter(x->x.Season_End_Year==2022 && x.Squad == team && x.Team_or_Opponent=="team", df_teams).D[1] for team in df_means.Team]
losses = [filter(x->x.Season_End_Year==2022 && x.Squad == team && x.Team_or_Opponent=="team", df_teams).L[1] for team in df_means.Team]

aproveitamento = [(3*wins[i]+draws[i])/(wins[i]+draws[i]+losses[i])/3 for i in 1:98]
cor(aproveitamento, df_means.Value)
scatter(df_means.Value, aproveitamento, series_annotations = text.(df_means.Team, 8, :bottom), legend = false)

idx_league = findall(el -> el.Team in filter(x -> x.Comp == "Ligue 1", df_teams).Squad, eachrow(df_means))
scatter(df_means[idx_league, :Value], aproveitamento[idx_league], series_annotations = text.(df_means[idx_league, :Team], 8, :bottom), legend = false)
cor(aproveitamento[idx_league], df_means[idx_league, :Value])

team = "Burnley"
rankings = [findfirst(x -> x ==  team, el.Team) for el in [sort(df_means, i, rev=true) for i in 2:size(df_means,2)]]
hcat(names(df_means)[2:end][sortperm(rankings)], sort(rankings))

idxs_posse = vcat(25:40,58:63,70:75,81:88)


brn_rk = [findfirst(x->x<brn_means[i], sort(df_means[:,k],rev=true)) for (i,k) in enumerate([k for k in keys(dict_stats)])]
rs_rk = [findfirst(x->x<rs_means[i], sort(df_means[:,k],rev=true)) for (i,k) in enumerate([k for k in keys(dict_stats)])]
brn_rk[2] = 98
brn_rk = Int64.(brn_rk)

rs_means=mean(Matrix(rs[:,[k for k in keys(dict_stats)]]), dims=1)[1,:]
brn_means=mean(Matrix(lineup("Burnley", data)[:,[k for k in keys(dict_stats)]]), dims=1)[1,:]

groupedbar(vcat([k for k in keys(dict_stats)], [k for k in keys(dict_stats)]), vcat(brn_rk, rs_rk), group = vcat(repeat(["Previous"], 6), repeat(["Now"], 6)), ylabel = "Ranking", ylim = (0,115))

mon_means=mean(Matrix(lineup("Monaco", data)[:,[k for k in keys(dict_stats)]]), dims=1)[1,:]
mon_rk = [findfirst(x->x<mon_means[i], sort(df_means[:,k],rev=true)) for (i,k) in enumerate([k for k in keys(dict_stats)])]

# ---------------------------------

scatter([454.7,274.2,387.7], [.462,.325,.438], series_annotations = text.(["Modelo 1", "Modelo 2", "Modelo 3"], :bottom, 8), legend = false,
xlabel = "millions of €", ylabel = "Score")


scatter([66880, 99280, 117840, 169360, 123280]/10^3, [.3479, .3421, .3442, .3399, .3385], label = "Minimizing cost", xlabel = "millions of €", ylabel = "Score",
            series_annotations = text.(["4-3-3", "4-1-4-1", "4-4-2", "5-4-1", "5-3-2"], 8, :bottom)) 

scatter!([77480, 105280, 117840, 169360, 129840]/10^3, [.3567, .3447, .3442, .3399, .3391], label = "Maximizing stats",
series_annotations = text.(["4-3-3", "4-1-4-1", "4-4-2", "5-4-1", "5-3-2"], 8, :top))


# rodar depois do for de best_team
plot([185.3, 232.05, 296.5, 319.5, 418], [0.360298, .360948, .367523, .37324, .403987], color = :black, label = "Efficient Frontier", linestyle = :dash, linealpha = 1)
[scatter!(dict["spent"], dict["score"], label = formation, markershape = [:circle, :diamond, :hexagon]) for (formation, dict) in best_team];
scatter!(xlabel = "millions of €", ylabel = "Score")
# scatter!(xlim=(100,530), ylim = (0.42,0.475), legend = :topleft, xlabel = "millions of €", ylabel = "Score")

savefig("img/best team")



# análise de sensibilidade em um experimento 'aleatório'
team = String(rand(unique(data.Squad)))
# formation = rand(keys(dict_formations))
time_limit, starting11 = 60.0,true
age_limit = rand(26:35)
pct_keep = rand(.2:.1:.6)
own_players_val = rand(.5:.1:1)
dict_stats = Dict([stat => 0.05 for stat in rand(DEFAULT_STATS, 5)])

dict_spent = Dict{AbstractString, Vector}()
dict_score = Dict{AbstractString, Vector}()
plot()

for formation in keys(dic_formações)

    spenditure = Float64[]
    scores = Float64[]
    lineups = DataFrame[]

    # _rs, max_spent, max_score = recommend_signings(team, data, df_means, dict_stats; time_limit = time_limit, age_limit = age_limit, pct_keep = pct_keep, starting11 = starting11,
    # own_players_val = own_players_val, formation = formation, budget = sum(sort(data.Value)[end-10:end])*1.0)
    _rs, max_spent, max_score = recommend_signings_multi_stage(team, data, df_means, dict_stats; time_limit = time_limit, age_limit = age, pct_keep = keep,
    formation = formation, budget = sum(sort(data.Value)[end-50:end])*1.0, scenarios = 23)

    # current_team_value = sum(sort(lineup(team, data).Value)[1:Int64(round(pct_keep*11))])/1000000
    # budget_millions = current_team_value + 20
    budget_millions = 200.
    # budget_millions = max_spent/2

    while budget_millions <= max_spent # /1000
        actual_budget = budget_millions # *10^6.0
        try
            rs, spent, score = recommend_signings_multi_stage(team, data, df_means, dict_stats; time_limit = time_limit, age_limit = age, pct_keep = keep,
                                                    formation = formation, budget = actual_budget, scenarios = 23)

            push!(spenditure, spent)
            push!(scores, score)
            push!(lineups, rs)
        catch
        end

        budget_millions *= 1.1
    end
    @show spenditure, scores
    plot!(spenditure, scores, label = formation, xlabel = "millions of €", ylabel = "Score", markershape = :circle,)
            # title = "$team age $age keep $pct_keep own $own_players_val")
    dict_spent[formation] = spenditure
    dict_score[formation] = scores
end

plot!()


plot()
for (form, score) in dict_score
    # idx_pos = vcat(1,findall(x->x>=0, diff(score) .+ 1))
    score = form == "4-1-4-1" ? score[2:end] : score
    idx_pos = findall(x->x>0, vcat(1,[score[i] > maximum(score[1:i-1]) for i in 2:length(score)]))
    @show form, score[idx_pos]
    @assert all(diff(score[idx_pos]) .> 0)
    plot!(dict_spent[form][idx_pos], score[idx_pos], label = form, xlabel = "millions of €", ylabel = "Score", markershape = :circle)
end

vline!([344.2], color = :black, linestyle = :dash, label = "Monaco's budget")
savefig("C:/Users/pedrohamacher/OneDrive/PC 2022/Documents/PUC/23.1/mod mat/projeto/Gráficos/sensibility sem bug")
# idx_valid = findall(x -> x > 0, scores)

# scatter(spenditure[idx_valid]/10^3, scores[idx_valid], legend = false, xlabel = "millions of €", ylabel = "Score",
#         series_annotations = text.(collect(current_team_value:20:500)[idx_valid], 8, :bottom))
# plot!(spenditure[idx_valid]/10^3, scores[idx_valid])



# -------------------------

dict_team_league = Dict([team => data[findfirst(x->x==team,data.Squad),:Comp] for team in unique(data.Squad)])

# idx_stats = filter(x -> x in keys(dict_stats))
league = dict_team_league[team]
league_teams = [team for (team,comp) in dict_team_league if comp == league]
league_mean = mean(Matrix(df_means[findall(x -> x in league_teams, df_means.Team), [k for k in keys(dict_stats)]]), dims=1)
league_max = maximum(Matrix(df_means[findall(x -> x in league_teams, df_means.Team), [k for k in keys(dict_stats)]]), dims=1)
max_all = maximum(Matrix(df_means[:, [k for k in keys(dict_stats)]]), dims=1)
min_all = minimum(Matrix(df_means[:, [k for k in keys(dict_stats)]]), dims=1)
# opt = mean(Matrix(rs[:, [k for k in keys(dict_stats)]]), dims=1) # solução
opt = permutedims([mean(vcat([y[c][:,stat] for c in 1:100]...)) for stat in keys(dict_stats)]) # new, includes 2nd stage
prev = Matrix(df_means[findall(x -> x == team, df_means.Team), [k for k in keys(dict_stats)]])

plot([k for k in keys(dict_stats)], transpose((opt .- min_all) ./ (max_all .- min_all)), markershape = :circle, label = "Model", legend = :topright)
plot!([k for k in keys(dict_stats)], transpose((league_max .- min_all) ./ (max_all .- min_all)), markershape = :circle, label = "$league best")
plot!([k for k in keys(dict_stats)], transpose((league_mean .- min_all) ./ (max_all .- min_all)), markershape = :circle, label = "$league average")
plot!([k for k in keys(dict_stats)], transpose((prev .- min_all) ./ (max_all .- min_all)), markershape = :circle, label = "Previous rankings")

savefig("img/Burnley comparison")

comp_names = ["Def SCA", "3rd Tackles", "PSxG+-", "Goals", "PrgRec", "KP", "PrgDist", "3rd Carries"]
comp_vals = vcat([permutedims((v .- min_all) ./ (max_all .- min_all))[:,1] for v in [opt, league_max, league_mean, prev]]...)
groupedbar(repeat(comp_names,4), comp_vals, group = vcat([repeat([nm],8) for nm in ["Model", "Ligue 1 best", "Ligue 1 average", "Previous"]]...), ylim = (0,1.3))

savefig("C:/Users/pedrohamacher/OneDrive/PC 2022/Documents/PUC/23.1/mod mat/projeto/Gráficos/monaco comparison")


max_all = maximum(Matrix(df_means[:, [k for k in keys(dict_stats_defaults3)]]), dims=1)
min_all = minimum(Matrix(df_means[:, [k for k in keys(dict_stats_defaults3)]]), dims=1)
m1 = mean(Matrix(rs1[:, [k for k in keys(dict_stats_defaults3)]]), dims=1) # solução
m2 = mean(Matrix(rs2[:, [k for k in keys(dict_stats_defaults3)]]), dims=1) # solução
m3 = mean(Matrix(rs3[:, [k for k in keys(dict_stats_defaults3)]]), dims=1) # solução
names_def = ["Clr", "SCA", "PrgCarr", "Gls", "PSxG+-", "PrgRec", "Tkl+Int", "xA", "PrgDist", "Drib"]

plot(names_def, transpose((m1 .- min_all) ./ (max_all .- min_all)), markershape = :circle, label = "Model 1", legend = :topright, ylim = (0,2))
plot!(names_def, transpose((m2 .- min_all) ./ (max_all .- min_all)), markershape = :circle, label = "Model 2", legend = :topright)
plot!(names_def, transpose((m3 .- min_all) ./ (max_all .- min_all)), markershape = :circle, label = "Model 3", legend = :topright)

savefig("img/best team comparison")




m = mean(Matrix(rs[:, [k for k in keys(dict_stats_defaults3)]]), dims=1) # solução
m = permutedims([mean(vcat([all_forms[1][4][c][:,stat] for c in 1:100]...)) for stat in keys(dict_stats_defaults3)])
max_all = maximum(Matrix(df_means[:, [k for k in keys(dict_stats_defaults3)]]), dims=1)
min_all = minimum(Matrix(df_means[:, [k for k in keys(dict_stats_defaults3)]]), dims=1)
q95 = permutedims(quantile.(eachcol(Matrix(df_means[:, [k for k in keys(dict_stats_defaults3)]])), .95))


plot(names_def, transpose((m .- min_all) ./ (max_all .- min_all)), markershape = :circle, label = "Model", legend = :topright, ylim = (0,2))

bar(names_def, transpose((m .- min_all) ./ (max_all .- min_all)), label = "Model", legend = :topright, ylim = (0,2))
hline!([0,.5,.95,1], color = :black, linestyle = :dash)

m_norm = (m .- min_all) ./ (max_all .- min_all)
q95_norm = (q95 .- min_all) ./ (max_all .- min_all)
groupedbar(vcat(names_def,names_def), permutedims(hcat(m_norm,q95_norm))[:,1], group = vcat(repeat(["Model"],10),repeat(["95% quantile"],10)))
hline!([0,1], color = :black, linestyle = :dash, label = "Best team")
savefig("C:/Users/pedrohamacher/OneDrive/PC 2022/Documents/PUC/23.1/mod mat/projeto/Gráficos/95 quantile comparison")

dict_league_team = Dict([v => k for (k,v) in dict_team_league])



league_mean = mean(Matrix(df_means[:, [k for k in keys(dict_stats)]]), dims=1)
league_q95 = permutedims(quantile.(eachcol(Matrix(df_means[:, [k for k in keys(dict_stats)]])), .95))
league_max = maximum(Matrix(df_means[:, [k for k in keys(dict_stats)]]), dims=1)
max_all = maximum(Matrix(df_means[:, [k for k in keys(dict_stats)]]), dims=1)
min_all = minimum(Matrix(df_means[:, [k for k in keys(dict_stats)]]), dims=1)
opt = mean(Matrix(rs[:, [k for k in keys(dict_stats)]]), dims=1) # solução

plot([k for k in keys(dict_stats)], transpose((opt .- min_all) ./ (max_all .- min_all)), markershape = :circle, label = "Model", legend = :topright)
plot!([k for k in keys(dict_stats)], transpose((league_max .- min_all) ./ (max_all .- min_all)), markershape = :circle, label = "Best team")
plot!([k for k in keys(dict_stats)], transpose((league_mean .- min_all) ./ (max_all .- min_all)), markershape = :circle, label = "Average team")
plot!([k for k in keys(dict_stats)], transpose((league_q95 .- min_all) ./ (max_all .- min_all)), markershape = :circle, label = "top 5%")











player_apps = vcat([hcat(pl,sum(pl in el for el in all_teams)) for pl in unique(vcat(all_teams...))]...)
bar(player_apps[:,2], orientation = :h, yticks = (1:size(player_apps,1), player_apps[:,1]), yflip=true)

bar(1:12, orientation=:h, yticks=(1:12, ticklabel), yflip=true)




# --------- jogadores no campinho ---------------
dic_formações = Dict{String,Tuple}()
dic_formações["3-5-2"] = ([53,15,25,25,53,47,47,70,85,85],[8,34,22,46,60,20,48,34,25,43])
dic_formações["4-3-3"] = ([35,20,20,35,43,55,55,85,88,85],[10,25,43,58,34,22,46,10,34,58])
dic_formações["4-1-4-1"] = ([30,20,20,30,44,60,60,60,60,85],[10,25,43,58,34,7,25,43,61,34])
dic_formações["4-4-2"] = ([30,20,20,30,60,50,50,60,85,85],[10,25,43,58,10,25,43,58,25,43])
dic_formações["3-4-3"] = ([55,20,27,27,55,50,50,82,88,82],[8,34,22,46,60,23,45,15,34,53])

function desenha_campo(form::AbstractString, df_orig::DataFrame; team_color = :white, secondary_color = :black)
    df = deepcopy(df_orig)

    plot([0,105,105,0,0,52.5,52.5],[0,0,68,68,0,0,68], color="white", legend=false, bg="green")
    plot!([0,16.5,16.5,0],[13.84,13.84,54.16,54.16], color = "white")
    plot!([105,88.5,88.5,105],[13.84,13.84,54.16,54.16], color = "white")
    plot!([0,5.49,5.49,0],[24.86,24.86,43.14,43.14], color = "white")
    plot!([105,99.5,99.55,105],[24.86,24.86,43.14,43.14], color = "white")
    plot!(9.15*sin.(0:2pi/100:2pi) + 52.5ones(100), 9.15*cos.(0:2pi/100:2pi) + 34ones(100),color="white")
    plot!(3.66*sin.(0:pi/100:pi) + 16.5ones(100), 3.66*cos.(0:pi/100:pi) + 34ones(100), color="white")
    plot!(3.66*sin.(pi:pi/100:2pi) + 88.5ones(101), 3.66*cos.(pi:pi/100:2pi) + 34ones(101), color="white")

    # scatter!(dic_formações[form], series_annotations = text.(names[2:end], 10, :bottom), color = team_color, markersize = 8, msc = secondary_color)
    # scatter!(dic_formações[form], series_annotations = text.(rand(1:100,10), 10, :top), color = team_color, markersize = 8, msc = secondary_color)
    # scatter!([5], [34], series_annotations = text.(names[1], 10, :top), color = :black, markersize = 8, msc = :white)

    df[3,:Player] = ""
    scatter!(dic_formações[form], color = team_color, markersize = 8, msc = secondary_color)
    annotate!(dic_formações[form][1], dic_formações[form][2] .+ 1.5, text.(df.Player[2:end], 10, :bottom))
    annotate!(dic_formações[form][1], dic_formações[form][2] .- 1.5, text.(df.Apps[2:end], 10, :top))
    # annotate!(dic_formações[form][1], dic_formações[form][2] .+ 1.5, text.(df.Player[2:end], 10, :bottom))
    # annotate!(dic_formações[form][1], dic_formações[form][2] .- 1.5, text.(df.Apps[2:end], 10, :top))
    scatter!([2], [34], color = :black, markersize = 8, msc = :white)
    annotate!([2], [35], text.(df.Player[1], 10, :bottom))
    annotate!([2], [32.5], text.(df.Apps[1], 10, :top))
    annotate!([dic_formações[form][1][2]], [dic_formações[form][2][2] + 2.5], text(df_orig.Player[3], 10, :bottom))
end


scatter(dic_formações["3-5-2"])
scatter(dic_formações["3-4-3"])

desenha_campo("3-5-2", [Char(i) for i in 97:107])
desenha_campo("3-4-3", [Char(i) for i in 97:107])

desenha_campo(formation, filter(x->x.Apps>40,rs); team_color = RGB(229/255, 27/255, 34/255), secondary_color = :white)
desenha_campo(formation, rs[vcat(3,findall(x->x>=40,rs.Apps)),:]; team_color = RGB(229/255, 27/255, 34/255), secondary_color = :white)
savefig("C:/Users/pedrohamacher/OneDrive/PC 2022/Documents/PUC/23.1/mod mat/projeto/Gráficos/monaco")

desenha_campo("3-4-3", filter(x->x.Apps>=40,rs))
savefig("C:/Users/pedrohamacher/OneDrive/PC 2022/Documents/PUC/23.1/mod mat/projeto/Gráficos/best team")


for scenarios in 20:30
    Random.seed!(21)
    # @show i
    # @show std(mean(eachcol((rand(size(data,1), i) .> data.Injury_Prob))) .+ data.Injury_Prob .- 1)

    healthy = BitMatrix(zeros(size(data,1), scenarios))
    for i in 1:size(data,1)
        prob = data[i,:Injury_Prob]
        num_scen_healthy = Int64(round(scenarios*(1-prob)))
        v = BitArray(zeros(scenarios))
        v[1:num_scen_healthy] = ones(num_scen_healthy)
        healthy[i,:] = shuffle(v)
    end

    @show scenarios
    @show std(mean(eachcol(healthy)) .+ data.Injury_Prob .- 1)
end

