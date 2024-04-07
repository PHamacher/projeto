# ToDo geral: polivalência -> Fotmob? (outra vantagem: Nota SofaScore deles)
using CSV, DataFrames

dados2022 = CSV.read("dados/dados2022.csv", DataFrame)
dados2023 = CSV.read("dados/dados2023.csv", DataFrame)


# ToDo: remover interseções de data pra cada jogador (missed games está sendo contado dobrado) ex: Pogba
# preditor (glm, rede neural)
# feature engineering (agrupar tipos de lesão)
# pegar todas as temporadas que o jogador existiu para calcular média de lesões por temporada na carreira

# pq missed e data_injury[:,end] não estão iguais??

inj = CSV.read("dados/injuries.csv", DataFrame)[:,2:end]
inj = unique(inj, names(inj)[1:end-1])

filter!(x -> !("NA" in x[[:injured_since, :injured_until]]), inj)
replace!(inj.games_missed, "NA" => "0")
inj[!,:games_missed] = parse.(Int64, inj.games_missed)
replace!(inj.duration, "NA" => "0 days")
inj[!,:duration] = map(x -> parse(Int64, split(x, " ")[1]), inj.duration)
strip(split(inj[:,2][1], "\r\n")[end])
inj[!,:player_name] = map(x -> strip(split(x, "\r\n")[end]), inj.player_name)

gp_inj = groupby(inj, :player_url)

last_season = filter(x -> x.season_injured == "22/23", inj)
missed = combine(groupby(last_season, :player_url), :games_missed => sum)

function missed_days_season(sb::SubDataFrame, season::String)
    df_season = filter(x -> x.season_injured == season, sb)
    return sum(df_season.duration) # ToDo: fazer conta na mão pra evitar interseção
end

function player_injury_history(row::DataFrameRow)
    # idx_gp = findfirst(x -> x.player_url == algo, gp_inj)
    idx_gp = findfirst(x -> x.player_name[1] == row.Name, gp_inj) # ToDo: fazer cruzando URLs
    is_first = row.Url in dados2022.Url
    if isnothing(idx_gp)
        return hcat(row.Age, 0, 0, is_first, 0)
    end
    gp = gp_inj[idx_gp]
    last_season = missed_days_season(gp, "21/22")
    current_season = missed_days_season(gp, "22/23")
    career_avg = mean(combine(groupby(gp, :season_injured), :duration => sum).duration_sum) # ToDo: não estou considerando temporadas com 0 lesões (tá prejudicando a média do cara)
    return hcat(row.Age, last_season, career_avg, is_first, current_season)
end

# data_injury = vcat([player_injury_history(row) for row in eachrow(data)]...)
data_injury = vcat([player_injury_history(row) for row in eachrow(filter(x->x.Player in inj.player_name, data))]...)
data_injury = data_injury[setdiff(1:size(data_injury,1), 474),:] # removendo outlier
data_injury = data_injury[findall(x->x>0, data_injury[:,end]),:]
X,Y = data_injury[:,1:end-1], data_injury[:,end]

using GLM
glm = lm(hcat(ones(size(X, 1)), X), Y)


scatter(predict(glm), Y)
plot!(1:250, 1:250)


using Lasso

lasso = fit(LassoModel, @formula(x5 ~ x1+x2+x3+x4), DataFrame(data_injury, :auto))


scatter(predict(lasso), Y)
plot!(1:100,1:100)


histogram(missed[:,2])
histogram(vcat(missed[:,2], zeros(2562-1314)))
# takeaway: primeiro sortear se o cara lesionará; depois, quanto tempo (talvez só é válido pra simulação sequencial?)
# rede neural pra 1a etapa (se o cara lesiona ou não), regressão pra 2a (dado que lesionou, quanto tempo ficou fora)
# simula cenários pra tudo isso, mas no final só tô interessado na média (probabilidade de estar disponível num jogo qlqr, sem o sequencial)
# cenários -> output da rede neural é a probabilidade de lesionar ou não; intervalo de 95% da regressão é usado pra geral uma Normal/LogNormal

mapping = CSV.read("C:/Users/admin/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/mapping.csv", DataFrame)






scatter(inj.duration, inj.games_missed, xlim=(0,700)) # fazer regressão
# dá uns 7 dias/jogo perdido (as expected)




using Dates
setdiff(Date(2022):Day(1):Date(2023), Date(2022,6):Day(1):Date(2022,10), Date(2022,7):Day(1):Date(2022,8))

healthy = Date(2022,8,11):Day(1):Date(2023,6,10) # meio chutado, verificar qnd começa e termina temporada
healthy = Date(2022,8,15):Day(1):Date(2023,6,4) # Serie A
len_temp = length(healthy)
for row in eachrow(gp_inj[1655])
    healthy = setdiff(healthy, Date(row.injured_since):Day(1):Date(row.injured_until))
end
len_temp - length(healthy)
(len_temp - length(healthy))/len_temp


function missing_days(sdf::Union{SubDataFrame, DataFrame})
    size(sdf,1) == 0 && return 0
    y = 2000 + parse(Int64, sdf[1,:season_injured][1:2])
    healthy = Date(y,8,11):Day(1):Date(y+1,6,10) # meio chutado, verificar qnd começa e termina temporada
    len_temp = length(healthy)
    for row in eachrow(sdf)
        healthy = setdiff(healthy, Date(row.injured_since):Day(1):Date(row.injured_until))
    end
    return len_temp - length(healthy)
end

sort(unique(vcat([collect(Date(row.injured_since):Day(1):Date(row.injured_until)) for row in eachrow(gp_inj[1655])]...)))

# agrupar por jogador e temporada
# criar df com: jogador, temporada, dias fora
gp = groupby(inj, [:player_url, :season_injured])
df_inj = DataFrame(vcat([permutedims(vcat(Matrix(g)[1,1:3], missing_days(g))) for g in gp]...), [:Name, :Url, :Season, :Missed_days])
CSV.write("dados/injuries_treated.csv", df_inj)


function fill_missing_seasons!(df::DataFrame) # se o cara não machucou, preenche a temp com 0
    # for name in unique(df.Name)
    for url in unique(df.Url)
        # df_name = filter(x->x.Name==name, df)
        df_name = filter(x->x.Url==url, df)
        y1 = parse(Int64, sort(df_name.Season)[1][1:2])
        for y in y1+1:22
            str = "$y/$(y+1)"
            if !(str in df_name.Season)
                # push!(df, [df_name[1,1], df_name[1,2], str, 0])
                push!(df, vcat([df_name[1,1], df_name[1,2], str], zeros(size(df,2)-3))) # account outras colunas de missing days
            end
        end
    end
end

df_inj = CSV.read("dados/injuries_treated.csv", DataFrame); fill_missing_seasons!(df_inj)
gp = groupby(df_inj, :Url)

current, previous, means = [], [], []
for g in [g for g in gp]
    dict = Dict()
    for row in eachrow(g)
        dict[parse(Int64, row.Season[1:2])] = row.Missed_days
    end

    sort!(g, :Season, rev=true)
    y1 = parse(Int64, g[end,:Season][1:2])

    len = length(previous)
    for i in y1:21
        val_prev = get(dict, i, 0)
        val_cur = get(dict, i+1, 0)
        push!(previous, val_prev)
        push!(current, val_cur)

        mean_prev = mean(previous[len+1:end])
        push!(means, mean_prev)
    end
end
cor(current, previous)
cor(current, means)

idx = findall(x -> x >0, current)
cor(current[idx], previous[idx])
cor(current[idx], means[idx])





older = filter(x -> x.Season != "22/23", df_inj) 
new = filter(x -> x.Season == "22/23", df_inj) 
gp_old = groupby(older,:Url)
mean_old = combine(gp_old, :Missed_days => mean, :Url=>length)
ip = filter(x->x[2] >= 66.5 && x[3] >= 3, mean_old).Url # .Name
mean(filter(x->!(x.Name in ip), new).Missed_days)
mean(filter(x->x.Name in ip, new).Missed_days)

gp_new = groupby(new,:Url)
mean_new = combine(gp_new, :Missed_days => mean)
joined = sort(innerjoin(mean_old, mean_new, on = :Url, makeunique=true),2,rev=true)

histogram(filter(x->!(x.Url in ip), new).Missed_days)
histogram(filter(x->x.Url in ip, new).Missed_days)

plot();
for num in 20:10:110
    ip = filter(x->x[2] >= num && x[3] >= 3, mean_old).Url
    scatter!([num], [mean(filter(x->!(x.Url in ip), new).Missed_days), mean(filter(x->x.Url in ip, new).Missed_days)], color=1, label = nothing)
    scatter!([num], [mean(filter(x->x.Url in ip, new).Missed_days)], color=2, label = nothing)
end
plot!(xlabel = "Avg days out previous seasons", ylabel = "Days out last season")

[length(filter(x->x[2] >= num && x[3] >= 3, mean_old).Url) for num in 20:10:150]

for i in 2:21
    reg = lm(@formula(Missed_days_mean_1~Missed_days_mean), filter(x->x.Url_length >= i, joined))
    rookies_sum = sum(filter(x->x.Url_length<i,joined).Missed_days_mean) / (size(filter(x->x.Url_length<i,joined), 1) + 495) # 495 jogadores de dados2023 nunca se machucaram
    pred = [row.Url_length < i ? rookies_sum : predict(reg, [1 row.Missed_days_mean])[1] for row in eachrow(joined)]
    mse = mean((pred .- joined[:,end]) .^ 2)
    @show i
    @show mse
    @show mean(pred)/3.04
    println()
end
# mse mínimo: i=4
i = 4
reg = lm(@formula(Missed_days_mean_1~Missed_days_mean), filter(x->x.Url_length >= i, joined))
rookies_sum = sum(filter(x->x.Url_length<i,joined).Missed_days_mean) / (size(filter(x->x.Url_length<i,joined), 1) + 495) # 495 jogadores de dados2023 nunca se machucaram
pred = [row.Url_length < i ? rookies_sum : predict(reg, [1 row.Missed_days_mean])[1] for row in eachrow(joined)]
CSV.write("dados/injury_prob.csv", DataFrame(hcat([dict_tm_fb[url] for url in joined.Url], pred/304), [:Url, :Prob]))


mapping = CSV.read("C:/Users/admin/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/mapping.csv", DataFrame)
dict_fb_tm = Dict([row.UrlFBref => row.UrlTmarkt for row in eachrow(mapping)])
dict_tm_fb = Dict([row.UrlTmarkt => row.UrlFBref for row in eachrow(mapping)])

# ============= tipo de lesão ===============

idx_musc = []
push!(idx_musc, findall(x->occursin("musc",x), lowercase.(unique(inj.injury))))
push!(idx_musc, findall(x->occursin("hamstring",x), lowercase.(unique(inj.injury))))
push!(idx_musc, findall(x->occursin("thigh",x), lowercase.(unique(inj.injury))))
push!(idx_musc, findall(x->occursin("adductor",x), lowercase.(unique(inj.injury))))
push!(idx_musc, findall(x->occursin("calf",x), lowercase.(unique(inj.injury))))
push!(idx_musc, findall(x->occursin("groin",x), lowercase.(unique(inj.injury))))

idx_bone = []
push!(idx_bone, findall(x->occursin("bone",x), lowercase.(unique(inj.injury))))

idx_ankle = []
push!(idx_ankle, findall(x->occursin("ankle",x), lowercase.(unique(inj.injury))))
push!(idx_ankle, findall(x->occursin("achilles",x), lowercase.(unique(inj.injury))))

idx_foot = []
push!(idx_foot, findall(x->occursin("foot",x), lowercase.(unique(inj.injury))))
push!(idx_foot, findall(x->occursin("toe",x), lowercase.(unique(inj.injury))))
push!(idx_foot, findall(x->occursin("metatars",x), lowercase.(unique(inj.injury))))

idx_knee = []
push!(idx_knee, findall(x->occursin("knee",x), lowercase.(unique(inj.injury))))
push!(idx_knee, findall(x->occursin("menisc",x), lowercase.(unique(inj.injury))))
push!(idx_knee, findall(x->occursin("patella",x), lowercase.(unique(inj.injury))))
push!(idx_knee, findall(x->occursin("ligament",x) && !occursin("ankle",x), lowercase.(unique(inj.injury))))

idx_back = []
push!(idx_back, findall(x->occursin("back",x), lowercase.(unique(inj.injury))))
push!(idx_back, findall(x->occursin("lumb",x), lowercase.(unique(inj.injury))))
push!(idx_back, findall(x->occursin("hip",x), lowercase.(unique(inj.injury)))) # tem a ver mesmo?

idx_musc = vcat(idx_musc...)
idx_bone = vcat(idx_bone...)
idx_ankle = vcat(idx_ankle...)
idx_foot = vcat(idx_foot...)
idx_knee = vcat(idx_knee...)
idx_back = vcat(idx_back...)

unique(inj.injury)[setdiff(1:307, idx_bone, idx_ankle, idx_foot, idx_knee, idx_musc, idx_back)]


gp_type = groupby(filter(x->x.injury in unique(inj.injury)[idx_ankle], inj), [:player_url, :season_injured])
df_inj = DataFrame(vcat([permutedims(vcat(Matrix(g)[1,1:3], missing_days(g))) for g in gp_type]...), [:Name, :Url, :Season, :Missed_days])

gp = groupby(df_inj, :Url)

current, previous, means = [], [], []
for g in [g for g in gp]
    dict = Dict()
    for row in eachrow(g)
        dict[parse(Int64, row.Season[1:2])] = row.Missed_days
    end

    sort!(g, :Season, rev=true)
    y1 = parse(Int64, g[end,:Season][1:2])

    len = length(previous)
    for i in y1:21
        val_prev = get(dict, i, 0)
        val_cur = get(dict, i+1, 0)
        push!(previous, val_prev)
        push!(current, val_cur)

        mean_prev = mean(previous[len+1:end])
        push!(means, mean_prev)
    end
end

cor(current, previous)
cor(current, means)

idx = findall(x -> x >0, current)
cor(current[idx], previous[idx])
cor(current[idx], means[idx])



gp = groupby(inj, [:player_url, :season_injured])
df_inj = DataFrame(vcat([permutedims(vcat(Matrix(g)[1,1:3], missing_days(g))) for g in gp_type]...), [:Name, :Url, :Season, :Missed_days])

all_mds = []
for g in gp
    mds = []
    for idx_type in [idx_ankle,idx_back,idx_bone,idx_foot,idx_knee,idx_musc]
        filtered = filter(x->x.injury in unique(inj.injury)[idx_type], g)
        if size(filtered,1) > 0
            md = missing_days(filtered)
        else
            md = 0
        end
        push!(mds, md)
    end
    push!(all_mds, mds)
end


df_type = DataFrame(vcat([permutedims(vcat(Matrix(g)[1,1:3], missing_days(g), all_mds[i])) for (i,g) in enumerate(gp)]...), [:Name, :Url, :Season, :Missed_days, :Missed_ankle, :Missed_back, :Missed_bone, :Missed_foot, :Missed_knee, :Missed_muscle])
CSV.write("dados/injuries_type.csv", df_type)



using GLM

df_type = CSV.read("dados/injuries_type.csv", DataFrame)


gp = groupby(df_type, :Url)
current, previous, means = [], [], []
for g in [g for g in gp]
    dict = Dict()
    for row in eachrow(g)
        dict[parse(Int64, row.Season[1:2])] = Vector{Int64}(row[4:end])
    end

    sort!(g, :Season, rev=true)
    y1 = parse(Int64, g[end,:Season][1:2])

    len = length(previous)
    for i in y1:21
        val_prev = get(dict, i, Int64.(zeros(7)))
        val_cur = get(dict, i+1, Int64.(zeros(7)))
        val_prev[1] = val_prev[1] - sum(val_prev[2:end]) # pegar missing days de 'outros', não do total (multicolinearidade)
        push!(previous, hcat(val_prev...))
        push!(current, val_cur[1])

        mean_prev = mean(vcat(previous[len+1:end]...), dims=1)
        push!(means, mean_prev)
    end
end

X = vcat(means...)

glm = lm(hcat(ones(size(X, 1)), X), Int64.(current))

new_X = hcat(sum(X, dims=2), [x > 0 for x in X[:,2:end]])
glm = lm(hcat(ones(size(new_X, 1)), new_X), Int64.(current))

all_X = hcat(X, new_X)
glm = lm(hcat(ones(size(all_X, 1)), all_X), Int64.(current))

# ===============================
inj_det = CSV.read("dados/injuries_detailed.csv", DataFrame)
inj_det[!,"lesao"] = lowercase.(inj_det[:,"Lesão"])
inj_det[!,"sitio_macro"] = String.(replace(inj_det[:,"Sitio macro"], missing => "Outros", "Tronco" => "tronco"))
inj_det[!,"sitio"] = String.(replace(inj_det[:,"Sítio"], missing => "Outros"))
inj_det[!,"macro"] = String.(replace(inj_det[:,"Macro"], missing => "Outros"))
inj_det[!,"cirurgico"] = Int64.(replace(inj_det[:,"Micro"], missing => 0, "Cirúrgico" => 1))
inj_det[!,"trauma"] = Int64.(replace(inj_det[:,"Coluna1"], missing => 0, "Trauma" => 1))
inj_det = inj_det[:,9:end]


inj_comp = hcat(inj, vcat([filter(x->x.lesao==lowercase(name), inj_det)[:,2:end] for name in inj.injury]...))

gp_det = groupby(inj_comp, [:player_url, :season_injured])
det = DataFrame(vcat([permutedims(vcat(Matrix(g)[1,1:3], missing_days(g))) for g in gp_det]...), [:Name, :Url, :Season, :Missed_days])


df_trauma = DataFrame(vcat([permutedims(vcat(Matrix(g)[1,1:3], missing_days(filter(x->x.trauma==1,g)))) for g in gp_det]...), [:Name, :Url, :Season, :Missed_days])
df_no_trauma = DataFrame(vcat([permutedims(vcat(Matrix(g)[1,1:3], missing_days(filter(x->x.trauma==0,g)))) for g in gp_det]...), [:Name, :Url, :Season, :Missed_days])
df_cirurgico = DataFrame(vcat([permutedims(vcat(Matrix(g)[1,1:3], missing_days(filter(x->x.cirurgico==1,g)))) for g in gp_det]...), [:Name, :Url, :Season, :Missed_days])
df_no_cirurgico = DataFrame(vcat([permutedims(vcat(Matrix(g)[1,1:3], missing_days(filter(x->x.cirurgico==0,g)))) for g in gp_det]...), [:Name, :Url, :Season, :Missed_days])

df_sitio = [DataFrame(vcat([permutedims(vcat(Matrix(g)[1,1:3], missing_days(filter(x->x.sitio_macro==sit,g)))) for g in gp_det]...), [:Name, :Url, :Season, :Missed_days]) for sit in unique(inj_comp.sitio_macro)]
df_sitio = innerjoin(df_sitio..., on = [:Name, :Url, :Season], makeunique=true)
df_macro = [DataFrame(vcat([permutedims(vcat(Matrix(g)[1,1:3], missing_days(filter(x->x.macro==mac,g)))) for g in gp_det]...), [:Name, :Url, :Season, :Missed_days]) for mac in unique(inj_comp.macro)]
df_macro = innerjoin(df_macro..., on = [:Name, :Url, :Season], makeunique=true)

joined = innerjoin(df_trauma, df_no_trauma, df_cirurgico, df_no_cirurgico, df_sitio, df_macro, on = [:Name, :Url, :Season], makeunique=true)
rename!(joined, :Missed_days=>:Missed_trauma,:Missed_days_1=>:Missed_no_trauma,:Missed_days_2=>:Missed_cir,:Missed_days_3=>:Missed_no_cir)
fill_missing_seasons!(joined)


X = [hcat(mean(Matrix(filter(x->x.Season!="22/23",gp)[:,4:end]), dims=1), Matrix(filter(x->x.Season=="21/22",gp)[:,4:end])) for gp in groupby(joined, :Url) if size(gp,1)>3]
X = Float64.(vcat(X...))
y = [sum(Matrix(filter(x->x.Season=="22/23",gp)[:,4:5])) for gp in groupby(joined, :Url) if size(gp,1)>3]

glm = lm(hcat(ones(size(X, 1)), X), y)

glm = lm(hcat(ones(size(X[:,1:29], 1)), X[:,1:29]), y)
glm = lm(hcat(ones(size(X[:,30:end], 1)), X[:,30:end]), y)


# ===================================

using MultivariateStats
lda = fit(MulticlassLDA, new_X', [x>0 for x in current])
Ylda = predict(lda, new_X')

pca = fit(PCA, X'; maxoutdim=2)
Ypca = predict(pca, X')

scatter(Ypca[1,[x==0 for x in current]], Ypca[2,[x==0 for x in current]])
scatter!(Ypca[1,[x>0 for x in current]], Ypca[2,[x>0 for x in current]])




# jogador genérico
prob_healthy = 4750/11847 # count(x->x==0, current) / length(current)
(sum(rand(LogNormal(3.419311221638169, 1.0909291785311028), 6*10^6) .- 1) / 10^7) / 304
