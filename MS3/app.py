from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase
import spacy
import re
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_community.vectorstores import Neo4jVector
from langchain_core.language_models import LLM
from typing import Optional, List, Any
from pydantic import Field
from huggingface_hub import InferenceClient
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class GemmaLangChainWrapper(LLM):
    client: Any = Field(...)
    max_tokens: int = 500
    
    @property
    def _llm_type(self) -> str:
        return "gemma_hf_api"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2 
        )
        return response.choices[0].message["content"]
    
class LlamaLangChainWrapper(LLM):
    client: Any = Field(...)
    max_tokens: int = 500
    
    @property
    def _llm_type(self) -> str:
        return "llama_hf_api"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2 
        )
        return response.choices[0].message["content"]
    
class MistralLangChainWrapper(LLM):
    client: Any = Field(...)
    max_tokens: int = 500
    
    @property
    def _llm_type(self) -> str:
        return "mistral_hf_api"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2 
        )
        return response.choices[0].message["content"]

# Load the kb from the graph database (optional enhancement)
def load_fpl_kb(graph: Neo4jGraph) -> dict:
    kb = {
        "players": [],
        "teams": [],
        "positions": ["gk","gkp", "def", "mid", "fwd", "goalkeeper","goalkeepers" "defender", "midfielder", "forward","defenders","midfielders","forwards"],
        "stats": {}
    }

    # Load players
    player_results = graph.query("MATCH (p:Player) RETURN p.player_name AS name")
    kb["players"] = [record["name"] for record in player_results]

    # Load teams
    team_results = graph.query("MATCH (t:Team) RETURN t.name AS name")
    kb["teams"] = [record["name"] for record in team_results]

    # Load stats mapping
    kb["stats"] = {
        "points": "total_points",
        "goals": "goals_scored",
        "assists": "assists",
        "minutes": "minutes",
        "bonus": "bonus",
        "influence": "influence",
        "creativity": "creativity",
        "threat": "threat",
        "ict": "ict_index",
        "clean sheets": "clean_sheets",
        "form": "form"
    }

    return kb

def classify_fpl_intents(classifier, query: str):

    INTENTS = {
        "fixture_details": [
            "fixture", "fixtures", "when do", "when does", "when is", "play next", 
            "next match", "kickoff", "schedule", "upcoming match", "future match"
        ],

        "best_players_by_metric": [
            "top", "best", "highest", "leader", "rank", "ranking", 
            "top scorer", "top assist", "highest points", "most points", "stat leaders", "top players","best forward",
            "best midfielder", "best defender", "best goalkeeper","top number","best number"
        ],
        "player_or_team_performance": [
            "how did", "performance", "stats", "statistics", "record", "scored", 
            "assists", "goals", "points", "clean sheets", "how many", 
            "results","compare", "vs", "versus", "better than", "head to head", "compare stats", "comparison","more than","compare player1 and player2"
        ]
    }

    candidate_labels = list(INTENTS.keys())
    result = classifier(query, candidate_labels, multi_label=True)

    return result["labels"][0]  # the top intent

def add_to_lookup(terms, category):
    for item in terms:
        # If it's a dict (like stats), the item is the key, canonical is the value
        if isinstance(terms, dict):
            value = terms[item]
            key = item
        else:
            value = item.title()
            key = item
        
        ENTITY_LOOKUP[key.lower()] = (category, value)

def extract_fpl_entities(query: str) -> dict:
    """
    Extract entities from FPL query with improved accuracy and validation
    """
    
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query)
    entities = {
        "stat_type": "total_points",  # Default fallback
        "season": "2022-23",  # Default season
        "limit": 10  # Default limit
    }
    
    query_lower = query.lower()
    
    # ============================================================================
    # STEP 1: Extract using spaCy + lookup
    # ============================================================================
    
    # Track which players/teams we've seen to handle comparisons
    seen_players = []
    seen_teams = []
    
    for token in doc:
        text = token.text.lower()
        lemma = token.lemma_.lower()
    

        match = ENTITY_LOOKUP.get(text) or ENTITY_LOOKUP.get(lemma)
        
        if match:
            category, value = match
            
            # Handle Players
            if category == "player":
                if value not in seen_players:
                    seen_players.append(value)
                    if "player1" not in entities:
                        entities["player1"] = value
                        entities["player_name"] = value
                    elif "player2" not in entities:
                        entities["player2"] = value
            
            # Handle Teams
            elif category == "team":
                if value not in seen_teams:
                    seen_teams.append(value)
                    if "team1" not in entities:
                        entities["team1"] = value
                        entities["team_name"] = value
                    elif "team2" not in entities:
                        entities["team2"] = value
            
            # Handle Positions
            elif category == "position":
                    # Normalize Aliases
                    norm = value.upper()
                    if "MID" in norm: norm = "MID"
                    elif "FWD" in norm or "FORWARD" in norm: norm = "FWD"
                    elif "DEF" in norm: norm = "DEF"
                    elif "GK" in norm or "GKP" in norm or "GOALKEEPER" in norm: norm = "GKP"
                    entities["position"] = norm
            # Handle Stats
            elif category == "stat":
                if entities.get("stat_type") != "bonus":
                    entities["stat_type"] = value
    
    # ============================================================================
    # STEP 2: Multi-word entity extraction (e.g., "Mohamed Salah")
    # ============================================================================
    
    # Extract multi-word player names
    for player in FPL_KB["players"]:
        if player.lower() in query_lower:
            if player not in seen_players:
                seen_players.append(player)
                if "player1" not in entities:
                    entities["player1"] = player
                    entities["player_name"] = player
                elif "player2" not in entities:
                    entities["player2"] = player
    
    # Extract multi-word team names (e.g., "Manchester City")
    for team in FPL_KB["teams"]:
        if team.lower() in query_lower:
            if team not in seen_teams:
                seen_teams.append(team)
                if "team1" not in entities:
                    entities["team1"] = team
                    entities["team_name"] = team
                elif "team2" not in entities:
                    entities["team2"] = team
    
    # Extract multi-word stats (e.g., "clean sheets")
    for stat_key, stat_value in FPL_KB["stats"].items():
        if stat_key in query_lower:
            if entities.get("stat_type") != "bonus":
                entities["stat_type"] = stat_value
                break
    
    # ============================================================================
    # STEP 3: Regex extraction for structured patterns
    # ============================================================================
    
    # Extract Gameweek
    gw_match = re.search(r"(?:gw|gameweek|game week)\s*(\d+)", query_lower)
    if gw_match:
        entities["gw_number"] = int(gw_match.group(1))
    
    # Extract Season (2022-23 format)
    season_match = re.search(r"(20\d{2}-\d{2})", query_lower)
    if season_match:
        entities["season"] = season_match.group(1)
    else:
        # Try alternative format: "season 2022" or "in 2022"
        year_match = re.search(r"(?:season|year|in)\s*(20\d{2})", query_lower)
        if year_match:
            year = year_match.group(1)
            next_year = str(int(year) + 1)[-2:]
            entities["season"] = f"{year}-{next_year}"
    
    # Extract "top N" or "best N"
    limit_match = re.search(r"(?:top|best|first)\s*(\d+)", query_lower)
    if limit_match:
        entities["limit"] = int(limit_match.group(1))
    
    # Extract current gameweek for recommendations
    if any(word in query_lower for word in ["recommend", "suggest", "current", "now", "right now"]):
        # If no specific GW mentioned, assume current GW context
        if "current_gw" not in entities and "gw_number" in entities:
            entities["current_gw"] = entities["gw_number"]
        elif "current_gw" not in entities:
            entities["current_gw"] = 20  # Default mid-season
    
    # Extract minimum value filters (e.g., "more than 10 points")
    filter_match = re.search(r"(?:more than|over|at least|minimum)\s*(\d+)", query_lower)
    if filter_match:
        entities["filter_value"] = int(filter_match.group(1))
    
    return entities

def get_fpl_cypher_query(intent: str, entities: dict) -> str:
    """
    Generate Cypher query based on FPL intents + extracted entities
    """

    player1 = entities.get("player1")
    player2 = entities.get("player2")
    team1 = entities.get("team1")
    team2 = entities.get("team2")
    stat = entities.get("stat_type", "total_points")
    gw = entities.get("gw_number")
    limit = entities.get("limit", 10)
    season = entities.get("season", "2022-23")
    

    # ----------------------------------------------------------------------
    # 1) PERFORMANCE: Single Player
    # ----------------------------------------------------------------------
    if intent == "player_or_team_performance" and player1 and not player2:
        if gw:
            return f"""
                MATCH (:Season {{season_name:'{season}'}})-[:HAS_GW]->(g:Gameweek {{GW_number:{gw}}})-[:HAS_FIXTURE]->(f:Fixture)
                MATCH (p:Player {{player_name:'{player1}'}})-[pi:PLAYED_IN]->(f)
                RETURN p.player_name AS player, pi.{stat} AS {stat}, g.GW_number AS gameweek
        
            """
        
        else:
            return f"""
                MATCH (:Season {{season_name:'{season}'}})-[:HAS_GW]->(g)-[:HAS_FIXTURE]->(f)
                MATCH (p:Player {{player_name:'{player1}'}})-[pi:PLAYED_IN]->(f)
                RETURN p.player_name AS player, SUM(pi.{stat}) AS total_{stat}, '{season}' AS season
            """

    # ----------------------------------------------------------------------
    # 2) PERFORMANCE: Single Team Performance summary
    # ----------------------------------------------------------------------
    if intent == "player_or_team_performance" and team1 and not team2:
        if gw:
            return f"""
                MATCH (:Season {{season_name:'{season}'}})-[:HAS_GW]->(g:Gameweek {{GW_number:{gw}}})-[:HAS_FIXTURE]->(f)
                MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team {{name:'{team1}'}})
                MATCH (p:Player)-[pi:PLAYED_IN]->(f)
                RETURN t.name AS team, SUM(pi.{stat}) AS total_{stat}, g.GW_number AS gameweek
            """
        else:
            return f"""
                MATCH (:Season {{season_name:'{season}'}})-[:HAS_GW]->(:Gameweek)-[:HAS_FIXTURE]->(f)
                MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team {{name:'{team1}'}})
                MATCH (p:Player)-[pi:PLAYED_IN]->(f)
                RETURN t.name AS team, SUM(pi.{stat}) AS total_{stat}
            """


    # ----------------------------------------------------------------------
    # 3) PERFORMANCE: Compare Two Players
    # ----------------------------------------------------------------------
    if intent == "player_or_team_performance" and player1 and player2:
        if gw:  
            return f"""
           UNWIND ['{player1}', '{player2}'] AS pname
            MATCH (p:Player {{player_name: pname}})
            OPTIONAL MATCH (p)-[pi:PLAYED_IN]->(f:Fixture)<-[:HAS_FIXTURE]-(g:Gameweek {{GW_number:{gw}}})<-[:HAS_GW]-(:Season {{season_name:'{season}'}})
            RETURN 
                p.player_name AS player, 
                COALESCE(SUM(pi.{stat}), 0) AS total_{stat},
                {gw} AS gameweek
            ORDER BY player


            """
        else:
            return f"""
                UNWIND ['{player1}', '{player2}'] AS pname
                MATCH (p:Player {{player_name: pname}})
                OPTIONAL MATCH (p)-[pi:PLAYED_IN]->(:Fixture)<-[:HAS_FIXTURE]-(g:Gameweek)<-[:HAS_GW]-(:Season {{season_name:'{season}'}})
                RETURN 
                    p.player_name AS player, 
                    COALESCE(SUM(pi.{stat}), 0) AS total_{stat}
                ORDER BY player

            """

    # ----------------------------------------------------------------------
    # 4) PERFORMANCE: Compare Two Teams
    # ----------------------------------------------------------------------
    if intent == "player_or_team_performance" and team1 and team2: 
        if gw: 
            return f""" MATCH (:Season {{season_name:'{season}'}}) -[:HAS_GW]->(g:Gameweek {{GW_number:{gw}}}) -[:HAS_FIXTURE]->(f) 
                        MATCH (f)-[:HAS_HOME_TEAM]->(home:Team) 
                        MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team) WHERE home.name IN ['{team1}', '{team2}'] OR away.name IN ['{team1}', '{team2}'] 
                        MATCH (p:Player)-[pi:PLAYEDIN]->(f) RETURN CASE WHEN home.name = '{team1}' OR away.name = '{team1}' 
                        THEN '{team1}' ELSE '{team2}' END 
                        AS team, SUM(pi.{stat}) AS total{stat}, g.GW_number AS gameweek """ 
        else: 
            return f""" MATCH (:Season {{season_name:'{season}'}})-[:HAS_GW]->(g)-[:HAS_FIXTURE]->(f) 
                        MATCH (f)-[:HAS_HOME_TEAM]->(home:Team) 
                        MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team) WHERE home.name IN ['{team1}', '{team2}'] OR away.name IN ['{team1}', '{team2}'] 
                        MATCH (p:Player)-[pi:PLAYEDIN]->(f) RETURN CASE WHEN home.name = '{team1}' OR away.name = '{team1}' 
                        THEN '{team1}' ELSE '{team2}' END AS team, SUM(pi.{stat}) AS total{stat} """
    # ----------------------------------------------------------------------
    # 5) FIXTURE: Next Fixture by Team
    # ----------------------------------------------------------------------
    if intent == "fixture_details" and team1 and gw and not team2:
        return f"""
             MATCH (:Season {{season_name:'{season}'}})-[:HAS_GW]->(g:Gameweek {{GW_number:{gw}}})-[:HAS_FIXTURE]->(f:Fixture)-[:HAS_HOME_TEAM]->(h:Team),
            (f)-[:HAS_AWAY_TEAM]->(a:Team)
            WHERE h.name = '{team1}' OR a.name = '{team1}'
            RETURN f.fixture_number AS fixture, f.kickoff_time, h, a
            ORDER BY f.kickoff_time ASC LIMIT 1
        """
    
    

    # ----------------------------------------------------------------------
    # 6) FIXTURE: Next Fixture involving Two Teams (Head-to-head)
    # ----------------------------------------------------------------------
    if intent == "fixture_details" and team1 and team2 and gw:
        return f"""
          MATCH (:Season {{season_name:'{season}'}})
        -[:HAS_GW]->(g:Gameweek {{GW_number:{gw}}})
        -[:HAS_FIXTURE]->(f:Fixture)
        MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
        MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
        WHERE (home.name = '{team1}' AND away.name = '{team2}')
        OR (home.name = '{team2}' AND away.name = '{team1}')
        RETURN 
            f.fixture_number AS fixture,
            f.kickoff_time AS kickoff_time,
            home.name AS home_team,
            away.name AS away_team
        ORDER BY f.kickoff_time ASC
        LIMIT 1

        """
    
    if intent == "fixture_details" and team1 and team2 and not gw:
        return f"""
            MATCH (:Season {{season_name:'{season}'}})
        -[:HAS_GW]->(g)
        -[:HAS_FIXTURE]->(f:Fixture)
        MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
        MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
        WHERE (home.name = '{team1}' AND away.name = '{team2}')
        OR (home.name = '{team2}' AND away.name = '{team1}')
        RETURN 
            f.fixture_number AS fixture,
            f.kickoff_time AS kickoff_time,
            g.GW_number AS gameweek,
            home.name AS home_team,
            away.name AS away_team
        ORDER BY f.kickoff_time ASC


            """

    # ----------------------------------------------------------------------
    # 7) BEST PLAYERS BY METRIC: Overall
    # ----------------------------------------------------------------------
    if intent == "best_players_by_metric" and not entities.get("position"):
        return f"""
            MATCH (:Season {{season_name:'{season}'}})-[:HAS_GW]->(g)-[:HAS_FIXTURE]->(f)
            MATCH (p:Player)-[pi:PLAYED_IN]->(f)
            RETURN p.player_name AS player, SUM(pi.{stat}) AS total_{stat}
            ORDER BY total_{stat} DESC LIMIT {limit}
        """

    # ----------------------------------------------------------------------
    # 8) BEST PLAYERS BY METRIC AND POSITION
    # ----------------------------------------------------------------------
    if intent == "best_players_by_metric" and entities.get("position"):
        position = entities["position"]
        return f"""
            MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {{name:'{position}'}})
            MATCH (:Season {{season_name:'{season}'}})-[:HAS_GW]->(g)-[:HAS_FIXTURE]->(f)
            MATCH (p)-[pi:PLAYED_IN]->(f)
            RETURN p.player_name AS player, SUM(pi.{stat}) AS total_{stat}, pos.name AS position
            ORDER BY total_{stat} DESC LIMIT {limit}
        """

    # ----------------------------------------------------------------------
    # 9) BEST PLAYERS — FILTER WHERE STAT ABOVE VALUE
    # ----------------------------------------------------------------------
    if intent == "best_players_by_metric" and entities.get("filter_value"):
        val = entities["filter_value"]
        return f"""
            MATCH (:Season {{season_name:'{season}'}})-[:HAS_GW]->(g)-[:HAS_FIXTURE]->(f)
            MATCH (p:Player)-[pi:PLAYED_IN]->(f)
            WITH p, SUM(pi.{stat}) AS total_stat
            WHERE total_stat > {val}
            RETURN p.player_name AS player, total_stat
            ORDER BY total_stat DESC LIMIT {limit}
        """

    # ----------------------------------------------------------------------
    # 10) FALLBACK
    # ----------------------------------------------------------------------
    return f"""MATCH (n:NonExistentLabel)
            RETURN n
            """

def format_query_result(intent: str, result: list, entities: dict = None) -> str:

    # Handle empty results
    if result is None or len(result) == 0:
        if intent == "fixture_details":
            team = entities.get("team1", "the team") if entities else "the team"
            return f"No fixtures found for {team}."
        elif intent == "best_players_by_metric":
            return "No players found matching the criteria."
        elif intent == "player_or_team_performance":
            entity_name = entities.get("player1") or entities.get("team1") if entities else None
            return f"No performance data found{' for ' + entity_name if entity_name else ''}."
        else:
            return "No results found."
    
    entities = entities or {}
    
    # ========================================================================
    # INTENT 1: FIXTURE_DETAILS
    # ========================================================================
    if intent == "fixture_details":
        if len(result) == 1:
            rec = result[0]
            home = rec.get('h', 'Unknown').get('name', rec.get('home_team', 'Unknown'))
            away = rec.get('a', 'Unknown').get('name', rec.get('away_team', 'Unknown'))
            kickoff = rec.get('f.kickoff_time', 'TBD')
            fixture_num = rec.get('fixture', rec.get('fixture_number', ''))
            
            return f"{home} will play against {away} at {kickoff if kickoff != 'TBD' else 'TBD'} (Fixture #{fixture_num})"
    
        else:
            response = "Upcoming Fixtures:\n\n"
            for i, rec in enumerate(result, 1):
                home = rec.get('home_team', 'Unknown')
                away = rec.get('away_team', 'Unknown')
                kickoff = rec.get('kickoff_time', 'TBD')
                gameweek = rec.get('gameweek', 'gw')
                response += f"{i}. {home} will play against {away} in Gameweek {gameweek}"
                if kickoff and kickoff != 'TBD':
                    response += f" at {kickoff} (Fixture #{rec.get('fixture', '')})"
                response += "\n"
            return response.strip()
    
    # ========================================================================
    # INTENT 2: BEST_PLAYERS_BY_METRIC
    # ========================================================================
    elif intent == "best_players_by_metric":
        stat_type = entities.get("stat_type", "total_points")
        position = entities.get("position")
        limit = entities.get("limit", len(result))
        
        # Convert stat field name to display name
        stat_names = {
            "goals_scored": "Goals", "assists": "Assists", "total_points": "Points",
            "bonus": "Bonus Points", "clean_sheets": "Clean Sheets", 
            "saves": "Saves", "minutes": "Minutes", "form": "Form"
        }
        stat_display = stat_names.get(stat_type, stat_type.replace('_', ' ').title())
        
        position_text = f" {position}" if position else ""
        response = f"**Top{position_text} Players by {stat_display}:**\n\n"
        
        for i, rec in enumerate(result[:limit], 1):
            player = rec.get('player', 'Unknown Player')
            
            # Try different possible field names for the stat value
            stat_value = (rec.get(f'total_{stat_type}') or 
                         rec.get('total_stat') or 
                         rec.get(stat_type) or 
                         rec.get('Total') or 0)
            
            pos = rec.get('position', '')
            pos_text = f" ({pos})" if pos else ""
            
            response += f"{i}. {player}{pos_text}: **{stat_value}** {stat_display}\n"
        
        return response.strip()
    
    # ========================================================================
    # INTENT 3: PLAYER_OR_TEAM_PERFORMANCE
    # ========================================================================
    elif intent == "player_or_team_performance":
        has_player1 = bool(entities.get("player1"))
        has_player2 = bool(entities.get("player2"))
        has_team1 = bool(entities.get("team1"))
        has_team2 = bool(entities.get("team2"))
        has_gw = bool(entities.get("gw_number"))
        
        stat_type = entities.get("stat_type", "total_points")
        stat_names = {
            "goals_scored": "Goals", "assists": "Assists", "total_points": "Points",
            "bonus": "Bonus Points", "clean_sheets": "Clean Sheets", 
            "saves": "Saves", "minutes": "Minutes", "form": "Form"
        }
        stat_display = stat_names.get(stat_type, stat_type.replace('_', ' ').title())
        
        # --------------------------------------------------------------------
        # SUB-CASE 1: Compare Two Players
        # --------------------------------------------------------------------
        if has_player1 and has_player2 and len(result) >= 2:
            player1 = entities["player1"]
            player2 = entities["player2"]
            
            p1_stat = 0
            p2_stat = 0
            
            for rec in result:
                player_name = rec.get('player', '')
                stat_value = rec.get(f'total_{stat_type}', rec.get(stat_type, 0))
                
                if player_name == player1:
                    p1_stat = stat_value
                elif player_name == player2:
                    p2_stat = stat_value
            
            response = f"**Player Comparison - {stat_display}:**\n\n"
            response += f"{player1}: **{p1_stat}** {stat_display}\n"
            response += f"{player2}: **{p2_stat}** {stat_display}\n\n"
            
            if p1_stat > p2_stat:
                diff = p1_stat - p2_stat
                response += f"{player1} has {diff} more {stat_display} than {player2}"
            elif p2_stat > p1_stat:
                diff = p2_stat - p1_stat
                response += f"{player2} has {diff} more {stat_display} than {player1}"
            else:
                response += f"Both players are equal in {stat_display}"
            
            return response
        
        # --------------------------------------------------------------------
        # SUB-CASE 2: Compare Two Teams
        # --------------------------------------------------------------------
        elif has_team1 and has_team2 and len(result) >= 2:
            team1 = entities["team1"]
            team2 = entities["team2"]
            
            t1_stat = 0
            t2_stat = 0
            
            for rec in result:
                team_name = rec.get('team', '')
                stat_value = rec.get(f'total_{stat_type}', rec.get(stat_type, 0))
                
                if team_name == team1:
                    t1_stat = stat_value
                elif team_name == team2:
                    t2_stat = stat_value
            
            response = f"**Team Comparison - {stat_display}:**\n\n"
            response += f"{team1}: **{t1_stat}** {stat_display}\n"
            response += f"{team2}: **{t2_stat}** {stat_display}\n\n"
            
            if t1_stat > t2_stat:
                diff = t1_stat - t2_stat
                response += f"{team1} has {diff} more {stat_display} than {team2}"
            elif t2_stat > t1_stat:
                diff = t2_stat - t1_stat
                response += f"{team2} has {diff} more {stat_display} than {team1}"
            else:
                response += f"Both teams are equal in {stat_display}"
            
            return response
        
        # --------------------------------------------------------------------
        # SUB-CASE 3: Single Player in Specific Gameweek
        # --------------------------------------------------------------------
        elif has_player1 and has_gw and len(result) >= 1:
            rec = result[0]
            player = rec.get('player', entities.get('player1', 'Unknown Player'))
            gw = rec.get('gameweek', entities.get('gw_number', '?'))
            
            stat_value = rec.get(stat_type, rec.get(f'total_{stat_type}', 0))
            
            response = (
                f"{player} played in Gameweek {gw}"
                + (f", scoring {rec.get('goals_scored', rec.get('goals', 0))} goals" if 'goals_scored' in rec or 'goals' in rec else "")
                + (f", providing {rec['assists']} assists" if 'assists' in rec else "")
                + (f", playing {rec['minutes']} minutes" if 'minutes' in rec else "")
                + f", with {stat_value} {stat_display}."
            )

            return response.strip()
        
        # --------------------------------------------------------------------
        # SUB-CASE 4: Single Team in Specific Gameweek
        # --------------------------------------------------------------------
        elif has_team1 and has_gw and len(result) >= 1:
            rec = result[0]
            team = rec.get('team', entities.get('team1', 'Unknown Team'))
            gw = rec.get('gameweek', entities.get('gw_number', '?'))
            
            stat_value = rec.get(f'total_{stat_type}', rec.get(stat_type, 0))
            
            response = f"{team} in Gameweek {gw} has a total of {stat_value} {stat_display}."

            
            return response.strip()
        
        # --------------------------------------------------------------------
        # SUB-CASE 5: Single Player Full Season
        # --------------------------------------------------------------------
        elif has_player1 and not has_gw:
            rec = result[0]
            player = rec.get('player', entities.get('player1', 'Unknown Player'))
            season = entities.get('season', '2022-23')
            
            stat_value = rec.get(f'total_{stat_type}', rec.get(stat_type, 0))
            
            response = (
            f"{player} in the {season} season has a total of {stat_value} {stat_display}"
            + (f", scoring {rec.get('total_goals_scored', rec.get('goals', 0))} goals" 
            if 'total_goals_scored' in rec or 'goals' in rec else "")
            + (f", providing {rec.get('total_assists', rec.get('assists', 0))} assists" 
            if 'total_assists' in rec or 'assists' in rec else "")
            + (f", playing {rec.get('total_minutes', rec.get('minutes', 0))} minutes" 
            if 'total_minutes' in rec or 'minutes' in rec else "")
            + "."
            )

            
            return response.strip()
        
        # --------------------------------------------------------------------
        # SUB-CASE 6: Single Team Full Season
        # --------------------------------------------------------------------
        elif has_team1 and not has_gw:
            rec = result[0]
            team = rec.get('team', entities.get('team1', 'Unknown Team'))
            season = entities.get('season', '2022-23')
            
            stat_value = rec.get(f'total_{stat_type}', rec.get(stat_type, 0))
            
            response = f"{team} in the {season} season has a total of {stat_value} {stat_display}."
            
            return response.strip()
        
        # --------------------------------------------------------------------
        # FALLBACK: Generic Performance
        # --------------------------------------------------------------------
        else:
            response = "**Performance Results:**\n\n"
            
            for i, rec in enumerate(result, 1):
                name = rec.get('player', rec.get('team', f'Entity {i}'))
                
                stat_value = None
                for key in [f'total_{stat_type}', stat_type, 'total_points', 'points']:
                    if key in rec:
                        stat_value = rec[key]
                        break
                
                if stat_value is not None:
                    response += f"{i}. {name}: {stat_value} {stat_display}\n"
                else:
                    response += f"{i}. {name}\n"
            
            return response.strip()
    
    # ========================================================================
    # FALLBACK for unknown intents
    # ========================================================================
    else:
        return "Query executed. Results retrieved."


def reset_vector_index(index_name, label, property_name, dimension):
    try:
        graph.query(f"DROP INDEX {index_name} IF EXISTS")
    except Exception as e:
        print(f"   - Warning dropping index: {e}")

    # 2. Create the new index
    create_query = f"""
    CREATE VECTOR INDEX {index_name}
    FOR (n:{label})
    ON (n.{property_name})
    OPTIONS {{indexConfig: {{
      `vector.dimensions`: {dimension},
      `vector.similarity_function`: 'cosine'
    }}}}
    """
    
    try:
        graph.query(create_query)
    except Exception as e:
        print(f"   - ❌ Error creating index: {e}")

def generate_player_feature_vector_embeddings(graph: Neo4jGraph, embedding_model, model_name: str):

    fetch_query = """
    MATCH (p:Player)
    WHERE p.fpl_features IS NOT NULL
    RETURN p.player_name AS name, p.fpl_features AS text
    """
    data = graph.query(fetch_query)

    update_query = f"""
    MATCH (p:Player {{player_name: $name}})
    SET p.feature_vector_embedding_{model_name} = $embedding
    """

    for row in data:
        vector = embedding_model.embed_query(row['text'])
        graph.query(update_query, {'name': row['name'], 'embedding': vector})

    reset_vector_index(
        index_name="player_feature_index_" + model_name, 
        label="Player", 
        property_name="feature_vector_embedding_" + model_name, 
        dimension=len(vector)
    )

    print("Feature Vector Embeddings generation complete!")

def generate_team_feature_vector_embeddings(graph: Neo4jGraph, embedding_model, model_name: str):

    fetch_query = """
    MATCH (t:Team)
    WHERE t.team_description IS NOT NULL
    RETURN t.name AS name, t.team_description AS text
    """
    data = graph.query(fetch_query)

    update_query = f"""
    MATCH (t:Team {{name: $name}})
    SET t.feature_vector_embedding_{model_name} = $embedding
    """

    for row in data:
        vector = embedding_model.embed_query(row['text'])
        graph.query(update_query, {'name': row['name'], 'embedding': vector})

    reset_vector_index(
        index_name="team_feature_index_" + model_name, 
        label="Team", 
        property_name="feature_vector_embedding_" + model_name, 
        dimension=len(vector)
    )

    print("Team Feature Vector Embeddings generation complete!")

def generate_all_embeddings(graph: Neo4jGraph):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    generate_player_feature_vector_embeddings(graph, embedding_model, "MiniLM")
    generate_team_feature_vector_embeddings(graph, embedding_model, "MiniLM")
    embedding_model2 = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    generate_player_feature_vector_embeddings(graph, embedding_model2, "MPNet")
    generate_team_feature_vector_embeddings(graph, embedding_model2, "MPNet")

def retrieve_embedding_search(query: str, embeddings_model, model_name: str):
    
    player_vector_store = Neo4jVector.from_existing_index(
        embedding=embeddings_model,  
        url=config.get('URI'),
        username=config.get('USERNAME'),
        password=config.get('PASSWORD'),
        index_name="player_feature_index_" + model_name,
        node_label="Player",
        embedding_node_property="feature_vector_embedding_" + model_name,
        text_node_property="fpl_features",
    )
    
    player_results = player_vector_store.similarity_search_with_score(query, k=5)
    
    team_vector_store = Neo4jVector.from_existing_index(
        embedding=embeddings_model,  
        url=config.get('URI'),
        username=config.get('USERNAME'),
        password=config.get('PASSWORD'),
        index_name="team_feature_index_" + model_name,
        node_label="Team",
        embedding_node_property="feature_vector_embedding_" + model_name,
        text_node_property="team_description",
    )
    
    team_results = team_vector_store.similarity_search_with_score(query, k=5)
    
    combined_results = []
    
    for doc, score in player_results:
        combined_results.append((doc, score, "Player"))
    
    for doc, score in team_results:
        combined_results.append((doc, score, "Team"))
    
    combined_results.sort(key=lambda x: x[1], reverse=True)
    
    top_3_results = combined_results[:3]

    formatted_context = "\n---\n".join([
        f"[{source}] {doc.page_content}" 
        for doc, score, source in top_3_results
    ])
    
    return formatted_context

def rag_pipline(llm, classifier, embedding_model, embedding_model_name, query):
    # 1. Retrieve from KG via Cypher
    intent = classify_fpl_intents(classifier, query)
    entities = extract_fpl_entities(query)
    cypher_query = get_fpl_cypher_query(intent, entities)
    cypher_result = graph.query(cypher_query)
    formatted_cypher = format_query_result(intent, cypher_result, entities)

    embedding_context = retrieve_embedding_search(query, embedding_model, embedding_model_name)

    # 3. Combine Contexts
    combined_context = f"Cypher Results:\n{formatted_cypher}\n\nEmbedding Results:\n{embedding_context}"
    print("Combined Context:\n", combined_context)

    # 4. Create Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are an expert Fantasy Premier League assistant.

    Use the context below to answer the user's question.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)

    class DummyRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str):
            return [Document(page_content=combined_context)]

    dummy_retriever = DummyRetriever()
    qa_chain = create_retrieval_chain(dummy_retriever, document_chain)

    return qa_chain

def generate_qa_chain(llm, combined_context):
   
    prompt = ChatPromptTemplate.from_template("""
    You are an expert Fantasy Premier League assistant.

    Use the context below to answer the user's question.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)

    class DummyRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str):
            return [Document(page_content=combined_context)]

    dummy_retriever = DummyRetriever()
    qa_chain = create_retrieval_chain(dummy_retriever, document_chain)

    return qa_chain

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

# Connect using the LangChain wrapper
graph = Neo4jGraph(
    url=uri,
    username=username,
    password=password,
    refresh_schema= False
)
# Ensure the connection is working by running a quick query (optional)
print(graph.query("MATCH (s:Season) RETURN s"))

FPL_KB = load_fpl_kb(graph)

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

ENTITY_LOOKUP = {}
add_to_lookup(FPL_KB["players"], "player")
add_to_lookup(FPL_KB["teams"], "team")
add_to_lookup(FPL_KB["positions"], "position")
add_to_lookup(FPL_KB["stats"], "stat")

model_minilm = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
model_mpnet = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

HF_TOKEN = config.get('HF_TOKEN')
print("Initializing models...")
# Gemma
gemma_client = InferenceClient(model="google/gemma-2-2b-it", token=HF_TOKEN)
gemma_llm = GemmaLangChainWrapper(client=gemma_client, max_tokens=500)

# Llama
llama_client = InferenceClient(model="meta-llama/Llama-3.2-3B-Instruct", token=HF_TOKEN)
llama_llm = LlamaLangChainWrapper(client=llama_client, max_tokens=500)

# Mistral
mistral_client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=HF_TOKEN)
mistral_llm = MistralLangChainWrapper(client=mistral_client, max_tokens=500)

models = {
        "Gemma-2-2B": gemma_llm,
        "Llama-3.2-3B": llama_llm,
        "Mistral-7B": mistral_llm
    }

# query = "When does Arsenal play against Liverpool in season 2022-23?"
# embedding_model_name = "MPNet"
# rag_chain = rag_pipline(
#     llm=gemma_llm,
#     classifier=classifier,
#     embedding_model=model_mpnet,
#     embedding_model_name=embedding_model_name,
#     query=query
# )
# response = rag_chain.invoke({"input": query})
# print(response["answer"])