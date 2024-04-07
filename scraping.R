library(worldfootballR)

# 2023
for (year in 2013:2021) {
    print(year)
    standard <- fb_big5_advanced_season_stats(season_end_year= year, stat_type= "standard", team_or_player= "player")
    keeper <- fb_big5_advanced_season_stats(season_end_year= year, stat_type= "keepers", team_or_player= "player")
    keeper_adv <- fb_big5_advanced_season_stats(season_end_year= year, stat_type= "keepers_adv", team_or_player= "player")
    shooting <- fb_big5_advanced_season_stats(season_end_year= year, stat_type= "shooting", team_or_player= "player")
    passing <- fb_big5_advanced_season_stats(season_end_year= year, stat_type= "passing", team_or_player= "player")
    passing_types <- fb_big5_advanced_season_stats(season_end_year= year, stat_type= "passing_types", team_or_player= "player")
    goal_shot_creation <- fb_big5_advanced_season_stats(season_end_year= year, stat_type= "gca", team_or_player= "player")
    defense <- fb_big5_advanced_season_stats(season_end_year= year, stat_type= "defense", team_or_player= "player")
    possession <- fb_big5_advanced_season_stats(season_end_year= year, stat_type= "possession", team_or_player= "player")
    playing_time <- fb_big5_advanced_season_stats(season_end_year= year, stat_type= "playing_time", team_or_player= "player")
    misc <- fb_big5_advanced_season_stats(season_end_year= year, stat_type= "misc", team_or_player= "player")

    write.csv(standard, paste("C:/Users/pedrohamacher/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/", year, "/player_standard.csv", sep=""), fileEncoding = "UTF-8")
    write.csv(keeper, paste("C:/Users/pedrohamacher/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/", year, "/player_keeper.csv", sep=""), fileEncoding = "UTF-8")
    write.csv(keeper_adv, paste("C:/Users/pedrohamacher/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/", year, "/player_keeper_adv.csv", sep=""), fileEncoding = "UTF-8")
    write.csv(shooting, paste("C:/Users/pedrohamacher/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/", year, "/player_shooting.csv", sep=""), fileEncoding = "UTF-8")
    write.csv(passing, paste("C:/Users/pedrohamacher/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/", year, "/player_passing.csv", sep=""), fileEncoding = "UTF-8")
    write.csv(passing_types, paste("C:/Users/pedrohamacher/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/", year, "/player_passing_types.csv", sep=""), fileEncoding = "UTF-8")
    write.csv(goal_shot_creation, paste("C:/Users/pedrohamacher/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/", year, "/player_goal_shot_creation.csv", sep=""), fileEncoding = "UTF-8")
    write.csv(defense, paste("C:/Users/pedrohamacher/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/", year, "/player_defense.csv", sep=""), fileEncoding = "UTF-8")
    write.csv(possession, paste("C:/Users/pedrohamacher/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/", year, "/player_possession.csv", sep=""), fileEncoding = "UTF-8")
    write.csv(playing_time, paste("C:/Users/pedrohamacher/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/", year, "/player_playing_time.csv", sep=""), fileEncoding = "UTF-8")
    write.csv(misc, paste("C:/Users/pedrohamacher/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/", year, "/player_misc.csv", sep=""), fileEncoding = "UTF-8")
}

standard <- fb_big5_advanced_season_stats(season_end_year= 2023, stat_type= "standard", team_or_player= "team")
keeper <- fb_big5_advanced_season_stats(season_end_year= 2023, stat_type= "keepers", team_or_player= "team")
keeper_adv <- fb_big5_advanced_season_stats(season_end_year= 2023, stat_type= "keepers_adv", team_or_player= "team")
shooting <- fb_big5_advanced_season_stats(season_end_year= 2023, stat_type= "shooting", team_or_player= "team")
passing <- fb_big5_advanced_season_stats(season_end_year= 2023, stat_type= "passing", team_or_player= "team")
passing_types <- fb_big5_advanced_season_stats(season_end_year= 2023, stat_type= "passing_types", team_or_player= "team")
goal_shot_creation <- fb_big5_advanced_season_stats(season_end_year= 2023, stat_type= "gca", team_or_player= "team")
defense <- fb_big5_advanced_season_stats(season_end_year= 2023, stat_type= "defense", team_or_player= "team")
possession <- fb_big5_advanced_season_stats(season_end_year= 2023, stat_type= "possession", team_or_player= "team")
playing_time <- fb_big5_advanced_season_stats(season_end_year= 2023, stat_type= "playing_time", team_or_player= "team")
misc <- fb_big5_advanced_season_stats(season_end_year= 2023, stat_type= "misc", team_or_player= "team")

write.csv(standard, paste("C:/Users/admin/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/2023/team_standard.csv", sep=""), fileEncoding = "UTF-8")
write.csv(keeper, paste("C:/Users/admin/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/2023/team_keeper.csv", sep=""), fileEncoding = "UTF-8")
write.csv(keeper_adv, paste("C:/Users/admin/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/2023/team_keeper_adv.csv", sep=""), fileEncoding = "UTF-8")
write.csv(shooting, paste("C:/Users/admin/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/2023/team_shooting.csv", sep=""), fileEncoding = "UTF-8")
write.csv(passing, paste("C:/Users/admin/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/2023/team_passing.csv", sep=""), fileEncoding = "UTF-8")
write.csv(passing_types, paste("C:/Users/admin/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/2023/team_passing_types.csv", sep=""), fileEncoding = "UTF-8")
write.csv(goal_shot_creation, paste("C:/Users/admin/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/2023/team_goal_shot_creation.csv", sep=""), fileEncoding = "UTF-8")
write.csv(defense, paste("C:/Users/admin/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/2023/team_defense.csv", sep=""), fileEncoding = "UTF-8")
write.csv(possession, paste("C:/Users/admin/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/2023/team_possession.csv", sep=""), fileEncoding = "UTF-8")
write.csv(playing_time, paste("C:/Users/admin/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/2023/team_playing_time.csv", sep=""), fileEncoding = "UTF-8")
write.csv(misc, paste("C:/Users/admin/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/2023/team_misc.csv", sep=""), fileEncoding = "UTF-8")



mapped_players <- player_dictionary_mapping()
write.csv(mapped_players, paste("C:/Users/admin/OneDrive/Documentos antigo/Projetos_jl/Futebol/fbref data/Big 5/mapping.csv", sep=""), fileEncoding = "UTF-8")



url_fbref <- read.csv("dados/dados2023.csv")$Url
url_tm <- mapped_players$UrlTmarkt[mapped_players$UrlFBref %in% url_fbref]

injuries <- tm_player_injury_history(player_urls = url_tm)


write.csv(injuries, "dados/injuries.csv", fileEncoding = "UTF-8")
