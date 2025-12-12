import pandas as pd 
from neo4j import GraphDatabase
import numpy as np

config = {}

with open('config.txt', 'r') as file:
    for line in file:
        if "=" in line:
            key, value = line.split('=', 1)
            config[key.strip()] = value.strip()

uri = config.get('URI')
username = config.get('USERNAME')
password = config.get('PASSWORD')
driver = GraphDatabase.driver(uri, auth=(username, password))
print("Connected to Neo4j database")

def build_nodes(tx, season_names=None, gameweek_names=None, fixture_names=None, team_names=None, player_names=None, position_names=None):
    tx.run("MATCH (n) DETACH DELETE n") 
    if season_names: 
        tx.run( """ 
            UNWIND $names AS name 
            MERGE (s:Season {season_name: name}) 
            """, 
            names=season_names 
        )

    if gameweek_names: 
        tx.run( """ 
            UNWIND $names AS gw 
            MERGE (g:Gameweek {season: gw.season, GW_number: gw.GW}) 
            """,
            names=gameweek_names 
        )

    if fixture_names: 
        tx.run( """ 
            UNWIND $names AS fixture 
            MERGE (f:Fixture {season: fixture.season, fixture_number: fixture.fixture}) 
            SET f.kickoff_time = fixture.kickoff_time 
            """, 
            names=fixture_names 
        )

    if team_names:
        tx.run( """ 
            UNWIND $names AS name 
            MERGE (t:Team {name: name}) 
            """, 
            names=team_names 
        )
    if player_names:
        tx.run( """ 
            UNWIND $names AS player 
            MERGE (p:Player {player_name: player.name, player_element: player.element}) 
            """, 
            names=player_names 
        )
    if position_names:
        tx.run( """ 
            UNWIND $names AS pos 
            MERGE (p:Position {name: pos.name}) 
            """, 
            names=position_names 
        )

def build_relationships(tx, rel_season_gw=None, rel_gw_fixture=None, rel_fixture_home=None, rel_fixture_away=None, rel_player_position=None, rel_player_team=None, rel_player_fixture_stats=None):
    tx.run("MATCH ()-[r]->() DELETE r")

    if rel_season_gw:
        tx.run( """ 
            UNWIND $relations AS rel 
            MATCH (s:Season {season_name: rel.season}) 
            MATCH (g:Gameweek {season: rel.season, GW_number: rel.GW}) 
            MERGE (s)-[:HAS_GW]->(g) 
            """, 
            relations=rel_season_gw 
        )

    if rel_gw_fixture:
        tx.run( """ 
            UNWIND $relations AS rel 
            MATCH (g:Gameweek {season: rel.season, GW_number: rel.GW}) 
            MATCH (f:Fixture {season: rel.season, fixture_number: rel.fixture}) 
            MERGE (g)-[:HAS_FIXTURE]->(f) 
            """, 
            relations=rel_gw_fixture 
        )

    if rel_fixture_home:
        tx.run( """ 
            UNWIND $relations AS rel 
            MATCH (f:Fixture {season: rel.season, fixture_number: rel.fixture}) 
            MATCH (t:Team {name: rel.home_team}) 
            MERGE (f)-[:HAS_HOME_TEAM]->(t) 
            """, 
            relations=rel_fixture_home 
        )

    if rel_fixture_away:
        tx.run( """ 
            UNWIND $relations AS rel 
            MATCH (f:Fixture {season: rel.season, fixture_number: rel.fixture}) 
            MATCH (t:Team {name: rel.away_team}) 
            MERGE (f)-[:HAS_AWAY_TEAM]->(t) 
            """, 
            relations=rel_fixture_away 
        )

    if rel_player_position:
        tx.run( """ 
            UNWIND $relations AS rel 
            MATCH (p:Player {player_name: rel.name, player_element: rel.element}) 
            MATCH (pos:Position {name: rel.position}) 
            MERGE (p)-[r:PLAYS_AS]->(pos)
            """, 
            relations=rel_player_position 
        )

    if rel_player_team:
        tx.run( """ 
            UNWIND $relations AS rel 
            MATCH (p:Player {player_name: rel.name, player_element: rel.element}) 
            MATCH (t:Team {name: rel.team}) 
            MERGE (p)-[r:PLAYS_FOR]->(t) 
            """, 
            relations=rel_player_team 
        )

    if rel_player_fixture_stats:
        tx.run( """ 
            UNWIND $relations AS rel 
            MATCH (p:Player {player_name: rel.player_name, player_element: rel.player_element}) 
            MATCH (f:Fixture {season: rel.season, fixture_number: rel.fixture_id}) 
            MERGE (p)-[r:PLAYED_IN]->(f) 
            SET r.minutes = rel.minutes,
                r.goals_scored = rel.goals_scored,
                r.assists = rel.assists,
                r.total_points = rel.total_points,
                r.bonus = rel.bonus,
                r.clean_sheets = rel.clean_sheets,
                r.goals_conceded = rel.goals_conceded,
                r.own_goals = rel.own_goals,
                r.penalties_saved = rel.penalties_saved,
                r.penalties_missed = rel.penalties_missed,
                r.yellow_cards = rel.yellow_cards,
                r.red_cards = rel.red_cards,
                r.saves = rel.saves,
                r.bps = rel.bps,
                r.influence = rel.influence,
                r.creativity = rel.creativity,
                r.threat = rel.threat,
                r.ict_index = rel.ict_index,
                r.form = rel.form,
                r.total_points_sum = rel.total_points_sum,
                r.goals_scored_sum = rel.goals_scored_sum,
                r.assists_sum = rel.assists_sum,
                r.minutes_sum = rel.minutes_sum,
                r.bonus_sum = rel.bonus_sum,
                r.clean_sheets_sum = rel.clean_sheets_sum,
                r.feature_text = rel.feature_text
            """,
            relations=rel_player_fixture_stats
        )

try:
    df = pd.read_csv('../MS3/two_seasons_cleaned.csv')
except FileNotFoundError:
    print("CSV file not found. Please ensure the file path is correct.")
    exit()

seasons = df[['season']].drop_duplicates().to_dict('records')
gameweeks = df[['season', 'GW']].drop_duplicates().to_dict('records')
fixtures = df.groupby(['season', 'fixture'])['kickoff_time'].first().reset_index().to_dict('records')

home_teams = df['home_team'].unique()
away_teams = df['away_team'].unique()
unique_teams = set(home_teams) | set(away_teams)
teams = [{'name': team} for team in unique_teams]


players = df[['name', 'element']].drop_duplicates().to_dict('records')
positions = df[['position']].drop_duplicates().rename(columns={'position': 'name'}).to_dict('records')


rel_season_gw = df[['season', 'GW']].drop_duplicates().to_dict('records')
rel_gw_fixture = df[['season', 'GW', 'fixture']].drop_duplicates().to_dict('records')
rel_fixture_home = df[['season', 'fixture', 'home_team']].drop_duplicates().to_dict('records')
rel_fixture_away = df[['season', 'fixture', 'away_team']].drop_duplicates().to_dict('records')
rel_player_position = df[['name', 'element', 'position']].drop_duplicates().to_dict('records')
rel_player_team = df[['name', 'element', 'team']].drop_duplicates().to_dict('records')

stat_cols = [
    'minutes', 'goals_scored', 'assists', 'total_points', 'bonus', 
    'clean_sheets', 'goals_conceded', 'own_goals', 'penalties_saved', 
    'penalties_missed', 'yellow_cards', 'red_cards', 'saves', 'bps', 
    'influence', 'creativity', 'threat', 'ict_index', 'form', 'total_points_sum', 'goals_scored_sum', 
    'assists_sum', 'minutes_sum', 'bonus_sum', 'clean_sheets_sum', 'feature_text'
]

rel_player_fixture_stats = []

for row in df.itertuples(index=False):
    item = {
        'player_element': row.element,
        'player_name': row.name,
        'season': row.season,
        'fixture_id': row.fixture,
    }
    
    for col in stat_cols:
        item[col] = getattr(row, col)
        
    rel_player_fixture_stats.append(item)


with driver.session() as session:
    session.run("CREATE CONSTRAINT season_pk IF NOT EXISTS FOR (s:Season) REQUIRE s.season_name IS UNIQUE")
    session.run("CREATE CONSTRAINT gameweek_pk IF NOT EXISTS FOR (g:Gameweek) REQUIRE (g.season, g.GW_number) IS UNIQUE")
    session.run("CREATE CONSTRAINT fixture_pk IF NOT EXISTS FOR (f:Fixture) REQUIRE (f.season, f.fixture_number) IS UNIQUE")
    session.run("CREATE CONSTRAINT team_pk IF NOT EXISTS FOR (t:Team) REQUIRE t.name IS UNIQUE")
    session.run("CREATE CONSTRAINT player_pk IF NOT EXISTS FOR (p:Player) REQUIRE (p.player_name, p.player_element) IS UNIQUE")
    session.run("CREATE CONSTRAINT position_pk IF NOT EXISTS FOR (pos:Position) REQUIRE pos.name IS UNIQUE")

    session.execute_write( 
        build_nodes, 
        season_names= [s['season'] for s in seasons],
        gameweek_names=gameweeks, 
        fixture_names=fixtures, 
        team_names= [t['name'] for t in teams],
        player_names=players, 
        position_names=positions 
    )

    print("Nodes created successfully.")

    session.execute_write(
        build_relationships, 
        rel_season_gw=rel_season_gw, 
        rel_gw_fixture=rel_gw_fixture, 
        rel_fixture_home=rel_fixture_home, 
        rel_fixture_away=rel_fixture_away,
        rel_player_position=rel_player_position,
        rel_player_team=rel_player_team,
        rel_player_fixture_stats=rel_player_fixture_stats
    )

    print("Relationships created successfully.")

    

driver.close()