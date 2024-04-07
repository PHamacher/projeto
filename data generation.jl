using CSV, DataFrames, Dates, Statistics

cd("Futebol")

files = readdir("transfermarkt data")

cd("transfermarkt data")

df = Dict{String,DataFrame}()

for file in files
    if endswith(file, ".csv")
        df[file[1:end-4]] = CSV.read(file, DataFrame)
    end
end

dict_players = Dict([row.player_id => Player(row.name, row.date_of_birth, row.sub_position) for row in eachrow(df["players"])])
dict_clubs = Dict([row.club_id => row.name for row in eachrow(df["clubs"])])
dict_leagues = Dict([row.competition_id => row.name for row in eachrow(df["competitions"])])

player_vals = groupby(df["player_valuations"], :player_id)


cd("../../Futebol/fbref data")

df_players = CSV.read("Big 5/2023/all_players.csv", DataFrame)

function explanatory(df_players::DataFrame)
    vars = findall(x -> x in [String3, String7, Int64, Float64, Union{Missing, String3}, Union{Missing, String7}], eltype.(eachcol(df_players)))
    x = df_players[:,vars][:,Not(["plus_per__minus_90_Team.Success", "Nation", "Pos"])]

    types = eltype.(eachcol(df_players))
    x, names_ = DataFrame(), String[]
    for (i,col) in enumerate(eachcol(df_players))
        if types[i] in [Int64, Float64]
            push!(names_, names(df_players)[i])
            x = hcat(x, col)
            rename!(x, names_)
        elseif types[i] in [String3, String7] && length(intersect(map(x->x[1:1],col), ["$i" for i in 0:9])) > 0 # numerical only
            x = hcat(x, [el == "NA" ? 0 : parse(Float64, el) for el in col])
            push!(names_, names(df_players)[i])
            rename!(x, names_)
        elseif types[i] in [Union{Missing, String3}, Union{Missing, String7}] && !startswith(names(df_players)[i], "Nation")
            x = hcat(x, [(ismissing(el) || el == "NA") ? 0 : parse(Float64, el) for el in col])
            push!(names_, names(df_players)[i])
            rename!(x, names_)
        end
    end

    x = x[:,findall(x -> x !== Missing, eltype.(eachcol(x)))]
    x = x[:, Not(unique(vcat(filter(x->length(x)>0,[findall(el->ismissing(el), x[i,:]) for i in 1:size(x,1)])...)))]
    x = x[:, Not(["Column1", "Season_End_Year"])]
    playing_time = x[:,"Min_Playing.Time"]
    del = ["playing.time", "starts", "subs", "team.success", "playing_time", "team.success"]
    x = x[:, Not(filter(n -> maximum(occursin.(del, lowercase(n))), names(x)))]

    # convert everything to per match
    keep = ["percent", "per", "90"]
    idx_per_match = findall(n -> !maximum(occursin.(keep, lowercase(n))), names(x))
    x[!,idx_per_match] = x[:,idx_per_match] ./ (playing_time/90)

    return Array{Float64}(x), names(x)
end

x,nms = explanatory(df_players)

df["players"].url = map(x -> replace(x, "co.uk" => "com"), df["players"].url)

# tm_fb = CSV.read("C:\\Users\\admin\\Downloads\\worldfootballR_data-master\\raw-data\\fbref-tm-player-mapping\\output/fbref_to_tm_mapping.csv", DataFrame)
tm_fb = CSV.read("C:/Users/admin/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/mapping.csv", DataFrame)[:,2:end]


idx_players = intersect(findall(x -> x in df_players.Url, tm_fb.UrlFBref), findall(x -> x in df["players"].url, tm_fb.UrlTmarkt))

mutable struct Player
    name::String
    birth::Union{Date, Missing}
    position::Union{String, Missing}
    nation::Union{String, Missing}
    stats::DataFrame
    valuations::DataFrame
end

function Player(tm_fb::DataFrameRow, df_fb::DataFrame, df_trm::DataFrame, df_vals::DataFrame)
    name, url_fb, url_tm, pos = tm_fb
    pos = ismissing(pos) ? pos : String(pos)

    stats = filter(x -> x.Url == url_fb, df_fb)
    nat = stats.Nation[end]
    nat = ismissing(nat) ? nat : String(nat)

    df_tm = filter(x -> x.url == url_tm, df_trm)
    @assert size(df_tm,1) == 1
    name, birth = df_tm.name[1], df_tm.date_of_birth[1]
    id_tm = df_tm.player_id[1]

    vals = filter(x -> x.player_id == id_tm, df_vals)

    return Player(name, birth, pos, nat, stats, vals)
end


df_per_game = DataFrame(x,nms)
df_per_game.Url = df_players.Url
df_per_game = hcat(df_players[:,3:6], df_per_game)

idx_players = intersect(findall(x -> x in df_per_game.Url, tm_fb.UrlFBref), findall(x -> x in df["players"].url, tm_fb.UrlTmarkt))
@time players = [Player(row, df_per_game, df["players"], df["player_valuations"]) for row in eachrow(tm_fb[idx_players,:])]

filter!(x -> 2023 in x.stats.Season_End_Year, players)


const M = 10^9

function value(player::Player, date::DateTime)
    @show player.name
    issorted(player.valuations.date) || sort!(player.valuations.date)
    idx_date = findlast(dt -> dt <= date, player.valuations.date)
    idx_date = isnothing(idx_date) ? 1 : idx_date
    val = size(player.valuations,1) > 0 ? player.valuations[idx_date, :market_value_in_eur] : M
    return val
end

function season2023(player::Player)
    stats = player.stats
    # stats = filter(x -> x.Season_End_Year == 2023, player.stats) # necessário só se o df tiver múltiplas temporadas
    stats = DataFrame(stats[end, :]) # ToDo: estou pegando somente o último clube na temporada; pegar o que mais jogou? média ponderada?

    stats.Name = [player.name]
    stats.Birth = [player.birth]
    stats.Nation = [player.nation]
    stats.Position = [player.position]
    stats.Value = [value(player, DateTime(2023,7))]

    return stats
end

df2023 = vcat(season2023.(players)...)

CSV.write("C:\\Users\\admin\\OneDrive\\Documentos antigo\\Projetos_jl/Matérias PUC/23.1/prog mat/projeto/dados2023.csv", df2023)



# ===================== medias 2023 (elenco inteiro) =====================

df = CSV.read("C:/Users/pedrohamacher/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/2023/all_teams.csv", DataFrame)[:,3:end]
filter!(x -> x.Team_or_Opponent == "team", df)
df[!, :Save_percent_Penalty] = parse.(Float64, df.Save_percent_Penalty)
teams = df.Squad
df = df[:, filter(nm -> eltype(df[:,nm]) in [Int64, Float64], names(df))]
mins = df.Mins_Per_90
df_means_roster = DataFrame(hcat(teams, Matrix(df[:,3:end]) ./ mins ./ 11), vcat("Team", names(df)[3:end]))
df_means_roster[:,findall(x->occursin("percent", x), names(df_means_roster))] .*= mins .* 11/100
df_means_roster[:,findall(x->occursin("90", x), names(df_means_roster))] .*= mins
df_means_roster[:,["PSxG+_per__minus__Expected"]] .*= mins
CSV.write("dados/medias2023.csv", df_means_roster)
