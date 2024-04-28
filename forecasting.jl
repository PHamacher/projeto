using GLM, CSV, DataFrames
df_all = CSV.read("C:/Users/pedrohamacher/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/2022 - all years/all_players.csv", DataFrame)
filter!(x -> !ismissing(x.Age) && x.Age != "", df_all)
df_all[!,"Age"] = map(x -> isa(x,Int64) ? x : parse.(Int64, x[1:2]), df_all.Age)
filter!(x -> x.Mins_Per_90 >= 5, df_all)
idx_stats = vcat(11:33,35:224, 232:247)

mtx_all = Matrix(df_all[:, idx_stats])
replace!(mtx_all, "NA" => "0")
replace!(mtx_all, missing => "0")
df_all[!, idx_stats] = [(isa(el, Real) ? el : parse(Float64, el)) for el in mtx_all]

sort!(df_all, :Url)
players = groupby(df_all, :Url)

# dict_reg = Dict{String, StatsModels.TableRegressionModel}()
df_regs = DataFrame()
for stat in names(df_all)[idx_stats]
    sizes_ = [size(player,1) for player in players]
    sizes(i::UnitRange{Int64}) = collect(i) != Int64[] ? sizes_[i] : 1
    idxX = vcat([collect(cumsum(sizes(1:i-1))[end]+1:cumsum(sizes(1:i))[end]-1) for i in 1:length(players) if sizes_[i] > 1]...)
    idxY = vcat([collect(cumsum(sizes(1:i-1))[end]+2:cumsum(sizes(1:i))[end]) for i in 1:length(players) if sizes_[i] > 1]...)

    df_reg = DataFrame(Y = Float64.(df_all[idxY, stat]), Lag = Float64.(df_all[idxX,stat]), Age = Float64.(df_all[idxY,:Age]), Pos = map(x->x[1:2], df_all[idxY,:Pos]))
    reg = lm(@formula(Y~Lag+Age+Age^2+Pos), df_reg)
    # if DataFrame(coeftable(reg))[4,5] > .05
    #     reg = lm(@formula(Y~Lag+Age+Pos), df_reg)
    # end
    # if DataFrame(coeftable(reg))[3,5] > .05
    #     reg = lm(@formula(Y~Lag+Pos), df_reg)
    # end
    # re-estimar sem coeficientes não-relevantes

    vc = vcat(vcov(reg)...)
    df_regs[!, stat] = vcat(coef(reg), vc)
end

dict_positions = Dict("Goalkeeper"=>"GK","Left-Back"=>"DF","Centre-Back"=>"DF","Right-Back"=>"DF","Defensive Midfield"=>"MF","Central Midfield"=>"MF","Attacking Midfield"=>"MF","Right Winger"=>"FW","Left Winger"=>"FW","Centre-Forward"=>"FW")

CSV.write("dados/regs.csv", df_regs)