import streamlit as st
import time
from datetime import datetime
import plotly.graph_objects as go
import networkx as nx
from typing import Dict, List, Tuple
import json
import re
from neo4j.graph import Node, Relationship, Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError


class GraphVisualizer:
    def __init__(self):
        self.color_map = {
            'Player': '#FF0000', 'Team': '#0000FF', 
            'Fixture': '#00FF00', 'Gameweek': '#FFFF00',
            'Season': '#800080', 'Position': '#FFA500',
            'Manager': '#FF9F43', 'Unknown': '#95A5A6'
        }

    def rewrite_query(self, query: str) -> str:
        """Rewrites query to return Graph Objects instead of scalars"""
        parts = re.split(r'(?i)\bRETURN\b', query, maxsplit=1)
        if len(parts) < 2: return query 
        
        query_body = parts[0].strip()
        node_vars = re.findall(r'\(\s*([a-zA-Z0-9_]+)(?::|[\s{])', query_body)
        rel_vars = re.findall(r'\[\s*([a-zA-Z0-9_]+)(?::|[\s{])', query_body)
        all_vars = sorted(list(set(node_vars + rel_vars)))
        
        if all_vars:
            return f"{query_body} RETURN {', '.join(all_vars)}"
        return query

    def visualize(self, driver, query: str) -> go.Figure:
        visual_query = self.rewrite_query(query)
        
        # Execute query using the driver to get native Neo4j objects
        results = []
        try:
            with driver.session() as session:
                result = session.run(visual_query)
                results = [record for record in result]
        except Exception as e:
            # Fallback if driver access fails
            print(f"Visualization Error: {e}")
            return go.Figure()

        G = nx.DiGraph()
        processed_ids = set()
        node_labels_map = {}
        node_colors = {}
        node_hover_text = {}

        def process_node(node: Node):
            uid = getattr(node, 'element_id', node.id)
            if uid not in processed_ids:
                labels = list(node.labels)
                primary_label = labels[0] if labels else 'Unknown'
                if 'Player' in labels: primary_label = 'Player'
                elif 'Team' in labels: primary_label = 'Team'
                elif 'Gameweek' in labels: primary_label = 'Gameweek'
                elif 'Season' in labels: primary_label = 'Season'
                elif 'Fixture' in labels: primary_label = 'Fixture'

                props = dict(node)
                # Better name extraction based on node type
                if 'player_name' in props:
                    name = props['player_name']
                elif 'name' in props:
                    name = props['name']
                elif 'team_name' in props:
                    name = props['team_name']
                elif 'season_name' in props:
                    name = props['season_name']
                elif 'GW_number' in props:
                    name = str(props['GW_number'])
                elif 'fixture_number' in props:
                    name = f"Fixture {props['fixture_number']}"
                else:
                    name = str(uid)
                
                processed_ids.add(uid)
                node_labels_map[uid] = str(name)
                node_colors[uid] = self.color_map.get(primary_label, '#95A5A6')
                prop_str = "<br>".join([f"{k}: {v}" for k,v in props.items()])
                node_hover_text[uid] = f"<b>{primary_label}</b>: {name}<br>---<br>{prop_str}"
                G.add_node(uid)
            return uid

        def process_rel(rel: Relationship):
            start_uid = process_node(rel.start_node)
            end_uid = process_node(rel.end_node)
            G.add_edge(start_uid, end_uid, label=rel.type)

        for record in results:
            for val in record.values():
                if isinstance(val, Node): process_node(val)
                elif isinstance(val, Relationship): process_rel(val)
                elif isinstance(val, Path):
                    for n in val.nodes: process_node(n)
                    for r in val.relationships: process_rel(r)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, Node): process_node(item)
                        elif isinstance(item, Relationship): process_rel(item)

        if len(G.nodes()) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No graph entities found in this query scope", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(xaxis={'visible':False}, yaxis={'visible':False}, plot_bgcolor='white')
            return fig

        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        # Create edge traces with arrows
        edge_x, edge_y = [], []
        edge_annotations = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Add arrow annotation
            edge_annotations.append(
                dict(
                    ax=x0, ay=y0,
                    x=x1, y=y1,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor='#888',
                    standoff=15
                )
            )
            
            # Add edge label at midpoint
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            edge_label = edge[2].get('label', '')
            
            edge_annotations.append(
                dict(
                    x=mid_x, y=mid_y,
                    xref='x', yref='y',
                    text=edge_label,
                    showarrow=False,
                    font=dict(size=10, color='#555', family='Arial Black'),
                    bgcolor='rgba(255,255,255,0.8)',
                    borderpad=2
                )
            )

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#888'), hoverinfo='none', mode='lines')

        node_x, node_y, texts, colors, hovers = [], [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            texts.append(node_labels_map[node])
            colors.append(node_colors[node])
            hovers.append(node_hover_text[node])

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', hoverinfo='text',
            hovertext=hovers, text=texts, textposition="top center",
            marker=dict(color=colors, size=30, line=dict(width=2, color='white'))
        )

        return go.Figure(data=[edge_trace, node_trace], 
                         layout=go.Layout(
                             title='Knowledge Graph Visualization', 
                             showlegend=False, 
                             hovermode='closest', 
                             margin=dict(b=20,l=5,r=5,t=40),
                             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                             plot_bgcolor='white', 
                             height=600,
                             annotations=edge_annotations
                         ))
    

# Page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="FPL Graph-RAG Assistant",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize app_loaded state before any imports
if 'app_loaded' not in st.session_state:
    st.session_state.app_loaded = False

# Show loading screen during initialization
if not st.session_state.app_loaded:
    # Inject CSS and HTML for loading screen
    loading_placeholder = st.empty()
    
    with loading_placeholder.container():
        st.markdown("""
        <style>
            /* Hide Streamlit default elements */
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            
            .block-container {
                padding: 0 !important;
                max-width: 100% !important;
            }
            
            .main .block-container {
                padding-top: 0 !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <style>
            .loading-container {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                z-index: 9999;
            }
            
            .football-loader {
                position: relative;
                width: 120px;
                height: 120px;
                margin-bottom: 40px;
            }
            
            .football {
                width: 80px;
                height: 80px;
                background: white;
                border-radius: 50%;
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                animation: bounce 1.5s ease-in-out infinite;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            
            .football::before {
                content: '⚽';
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-size: 60px;
                animation: spin 2s linear infinite;
            }
            
            @keyframes bounce {
                0%, 100% { transform: translate(-50%, -50%) translateY(0); }
                50% { transform: translate(-50%, -50%) translateY(-30px); }
            }
            
            @keyframes spin {
                0% { transform: translate(-50%, -50%) rotate(0deg); }
                100% { transform: translate(-50%, -50%) rotate(360deg); }
            }
            
            .loading-text {
                color: white;
                font-size: 28px;
                font-weight: 700;
                margin-bottom: 20px;
                animation: pulse 2s ease-in-out infinite;
            }
            
            .loading-subtext {
                color: rgba(255,255,255,0.9);
                font-size: 16px;
                margin-bottom: 30px;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.6; }
            }
            
            .progress-bar-container {
                width: 300px;
                height: 6px;
                background: rgba(255,255,255,0.3);
                border-radius: 10px;
                overflow: hidden;
                margin-bottom: 15px;
            }
            
            .progress-bar {
                height: 100%;
                background: white;
                border-radius: 10px;
                animation: progress 3s ease-in-out;
                box-shadow: 0 0 10px rgba(255,255,255,0.5);
            }
            
            @keyframes progress {
                0% { width: 0%; }
                100% { width: 100%; }
            }
            
            .loading-steps {
                color: rgba(255,255,255,0.8);
                font-size: 14px;
                font-family: 'Courier New', monospace;
            }
            
            .step {
                margin: 5px 0;
                opacity: 0;
                animation: fadeIn 0.5s ease-in forwards;
            }
            
            .step:nth-child(1) { animation-delay: 0.3s; }
            .step:nth-child(2) { animation-delay: 0.8s; }
            .step:nth-child(3) { animation-delay: 1.3s; }
            .step:nth-child(4) { animation-delay: 1.8s; }
            .step:nth-child(5) { animation-delay: 2.3s; }
            
            @keyframes fadeIn {
                to { opacity: 1; }
            }
            
            .checkmark {
                color: #4caf50;
                font-weight: bold;
            }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="loading-container">
        <div class="football-loader">
        <div class="football"></div>
        </div>
        <div class="loading-text">Fantasy Premier League</div>
        <div class="loading-subtext">Initializing Graph-RAG System...</div>
        <div class="progress-bar-container">
        <div class="progress-bar"></div>
        </div>
        <div class="loading-steps">
        <div class="step"><span class="checkmark">✓</span> Loading Knowledge Graph...</div>
        <div class="step"><span class="checkmark">✓</span> Initializing AI Models...</div>
        <div class="step"><span class="checkmark">✓</span> Setting up Embeddings...</div>
        <div class="step"><span class="checkmark">✓</span> Connecting to Neo4j...</div>
        <div class="step"><span class="checkmark">✓</span> Ready to assist you!</div>
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Simulate loading time and import heavy modules
    time.sleep(0.5)  # Small delay to show the animation
    
    # Import app modules (this is where the actual loading happens)
    from app import (
        graph, models, model_minilm, model_mpnet, model_bge,
        classify_fpl_intents, extract_fpl_entities, get_fpl_cypher_query,
        format_query_result, retrieve_embedding_search, generate_qa_chain
    )
    
    # Store in session state to avoid reimporting
    st.session_state.graph = graph
    st.session_state.models = models
    st.session_state.model_minilm = model_minilm
    st.session_state.model_mpnet = model_mpnet
    st.session_state.model_bge = model_bge
    st.session_state.classify_fpl_intents = classify_fpl_intents
    st.session_state.extract_fpl_entities = extract_fpl_entities
    st.session_state.get_fpl_cypher_query = get_fpl_cypher_query
    st.session_state.format_query_result = format_query_result
    st.session_state.retrieve_embedding_search = retrieve_embedding_search
    st.session_state.generate_qa_chain = generate_qa_chain
    
    st.session_state.app_loaded = True
    st.rerun()

# Retrieve from session state
graph = st.session_state.graph
models = st.session_state.models
model_minilm = st.session_state.model_minilm
model_mpnet = st.session_state.model_mpnet
model_bge = st.session_state.model_bge
classify_fpl_intents = st.session_state.classify_fpl_intents
extract_fpl_entities = st.session_state.extract_fpl_entities
get_fpl_cypher_query = st.session_state.get_fpl_cypher_query
format_query_result = st.session_state.format_query_result
retrieve_embedding_search = st.session_state.retrieve_embedding_search
generate_qa_chain = st.session_state.generate_qa_chain

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    
    .stApp {
        max-width: 100%;
    }
    
    .query-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .result-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .cypher-box {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 15px;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
        margin: 10px 0;
        white-space: pre;
    }
    
    .context-box {
        background-color: #f0f8ff;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .answer-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
        font-size: 16px;
    }
    
    h1, h2, h3 {
        color: #2c3e50;
    }
    
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .example-query {
        background-color: #f8f9fa;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 5px 0;
        cursor: pointer;
        border: 1px solid #e0e0e0;
        transition: all 0.2s;
    }
    
    .example-query:hover {
        background-color: #e9ecef;
        border-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None

def create_graph_visualization(cypher_query: str, cypher_result: List[Dict]) -> go.Figure:
    """Create a graph visualization using Plotly by executing a modified query to get graph structure"""
    
    G = nx.DiGraph()
    
    # Parse the Cypher result to extract nodes and relationships
    nodes = {}  # node_id: {label, type, properties}
    edges = []
    node_labels = {}
    node_colors = {}
    node_types = {}
    edge_labels = {}
    
    color_map = {
        'Player': '#FF6B6B',
        'Team': '#4ECDC4',
        'Fixture': '#95E1D3',
        'Gameweek': '#FFE66D',
        'Season': '#A8E6CF',
        'Position': '#C7CEEA',
        'Unknown': '#95A5A6'
    }
    
    # Use the GraphVisualizer class to get proper graph data
    try:
        from app import graph as graph_conn
        viz = GraphVisualizer()
        driver_instance = graph_conn._driver
        return viz.visualize(driver_instance, cypher_query)
    except Exception as e:
        # Fallback to query parsing if driver access fails
        print(f"Could not use GraphVisualizer, falling back to query parsing: {e}")
        pass
    
    # Parse Cypher query to extract the graph structure
    import re
    
    # Extract node patterns from MATCH clauses: (variable:Label {property:value})
    # This pattern now handles both (var:Label) and (:Label) patterns
    node_pattern = r'\((\w*)(?::(\w+))?\s*(?:\{([^}]+)\})?\)'
    # Extract relationship patterns: -\[(\w+)?:(\w+)\]->
    rel_pattern = r'-\[(\w*):(\w+)\]->'
    
    query_nodes = {}  # variable: {label, properties, display_name}
    query_rels = []   # [(from_var, rel_type, to_var)]
    node_counter = 0  # For generating unique IDs for anonymous nodes
    
    # Find all nodes in the query
    for match in re.finditer(node_pattern, cypher_query):
        var = match.group(1) if match.group(1) else f"_anon_{node_counter}"
        if not match.group(1):
            node_counter += 1
        
        label = match.group(2) if match.group(2) else 'Unknown'
        props = match.group(3) if match.group(3) else ''
        
        # Extract property for display name
        display_name = var
        if props:
            # Try to extract the property value
            # Handle various property patterns: property:'value', property:value, property:123
            name_match = re.search(r'(\w+)\s*:\s*[\'"]?([^\'",}]+)[\'"]?', props)
            if name_match:
                prop_name = name_match.group(1)
                prop_value = name_match.group(2).strip("'\"")
                # Use only the property value for display, not the label
                display_name = prop_value
            elif label:
                display_name = label
        elif label:
            display_name = label
        
        query_nodes[var] = {
            'label': label,
            'display_name': display_name,
            'variable': var
        }
    
    # Find all relationships in the query by parsing the pattern more carefully
    # Handle both forward (->) and backward (<-) relationships
    import re
    
    # Find all relationship patterns with their surrounding nodes
    # Pattern for forward: (anything)-[relationship]->(anything)
    # Pattern for backward: (anything)<-[relationship]-(anything)
    rel_pattern_forward = r'\(([^)]*)\)\s*-\[([^\]]*)\]->\s*\(([^)]*)\)'
    rel_pattern_backward = r'\(([^)]*)\)\s*<-\[([^\]]*)\]-\s*\(([^)]*)\)'
    
    def extract_node_var(node_str):
        """Extract variable name from node string"""
        var_match = re.match(r'(\w+)', node_str.strip())
        return var_match.group(1) if var_match else None
    
    def find_anon_node_by_label(label):
        """Find anonymous node by label"""
        for var, info in query_nodes.items():
            if info['label'] == label and var.startswith('_anon_'):
                return var
        return None
    
    def resolve_node_var(node_str):
        """Resolve node variable, handling anonymous nodes"""
        var = extract_node_var(node_str)
        if not var or var == '':
            label_match = re.search(r':(\w+)', node_str)
            if label_match:
                var = find_anon_node_by_label(label_match.group(1))
        return var
    
    # Process forward relationships (->)
    for match in re.finditer(rel_pattern_forward, cypher_query):
        from_node_str = match.group(1)
        rel_str = match.group(2)
        to_node_str = match.group(3)
        
        from_var = resolve_node_var(from_node_str)
        to_var = resolve_node_var(to_node_str)
        
        rel_type_match = re.search(r':(\w+)', rel_str)
        rel_type = rel_type_match.group(1) if rel_type_match else None
        
        if from_var and to_var and rel_type and from_var in query_nodes and to_var in query_nodes:
            query_rels.append((from_var, rel_type, to_var))
    
    # Process backward relationships (<-)
    for match in re.finditer(rel_pattern_backward, cypher_query):
        from_node_str = match.group(1)  # This is actually the "to" node in the relationship
        rel_str = match.group(2)
        to_node_str = match.group(3)    # This is actually the "from" node in the relationship
        
        # Reverse the direction for backward arrows
        from_var = resolve_node_var(to_node_str)
        to_var = resolve_node_var(from_node_str)
        
        rel_type_match = re.search(r':(\w+)', rel_str)
        rel_type = rel_type_match.group(1) if rel_type_match else None
        
        if from_var and to_var and rel_type and from_var in query_nodes and to_var in query_nodes:
            query_rels.append((from_var, rel_type, to_var))
    
    # Enhance node display names with actual data from query results if available
    if cypher_result and len(cypher_result) > 0:
        first_result = cypher_result[0]
        for var, info in query_nodes.items():
            # Check if this variable appears in the result
            if var in first_result:
                value = first_result[var]
                if isinstance(value, (str, int, float)):
                    info['display_name'] = str(value)
                elif isinstance(value, dict) and 'name' in value:
                    info['display_name'] = value['name']
    
    # Now create nodes and edges based on the query structure
    for var, info in query_nodes.items():
        node_id = info['display_name']
        nodes[node_id] = info
        node_labels[node_id] = info['display_name']
        node_types[node_id] = info['label']
        node_colors[node_id] = color_map.get(info['label'], color_map['Unknown'])
        G.add_node(node_id)
    
    # Create edges from query relationships
    for from_var, rel_type, to_var in query_rels:
        if from_var in query_nodes and to_var in query_nodes:
            from_node = query_nodes[from_var]['display_name']
            to_node = query_nodes[to_var]['display_name']
            if from_node in nodes and to_node in nodes:
                edges.append((from_node, to_node))
                edge_labels[(from_node, to_node)] = rel_type
                G.add_edge(from_node, to_node)
    
    # Handle empty graph case
    if len(G.nodes()) == 0:
        # Create a simple placeholder graph
        fig = go.Figure()
        fig.add_annotation(
            text="No graph entities found in results<br>Query may return scalar values only",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="#7f8c8d")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=500
        )
        return fig
    
    # Create positions using spring layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Create edge traces with labels
    edge_x = []
    edge_y = []
    edge_label_x = []
    edge_label_y = []
    edge_label_text = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Add edge label at midpoint
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        edge_label_x.append(mid_x)
        edge_label_y.append(mid_y)
        edge_label_text.append(edge_labels.get(edge, ''))
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    # Edge labels trace
    edge_label_trace = go.Scatter(
        x=edge_label_x, y=edge_label_y,
        mode='text',
        text=edge_label_text,
        textposition='middle center',
        textfont=dict(size=9, color='#555', family='Arial Black'),
        hoverinfo='none',
        showlegend=False
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    hover_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node_labels.get(node, node))
        node_color.append(node_colors.get(node, '#95A5A6'))
        node_type = node_types.get(node, 'Unknown')
        hover_text.append(f"{node}<br>Type: {node_type}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=hover_text,
        text=node_text,
        textposition="top center",
        textfont=dict(size=10, color='#2c3e50'),
        marker=dict(
            showscale=False,
            color=node_color,
            size=30,
            line=dict(width=2, color='white')
        ),
        showlegend=False
    )
    
    # Create figure with corrected layout
    fig = go.Figure(
        data=[edge_trace, edge_label_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text='Knowledge Graph Visualization',
                font=dict(size=16, color='#2c3e50')
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600
        )
    )
    
    return fig

def process_query(query: str, model_name: str, retrieval_method: str, embedding_model_name: str):
    """Process the query and return all relevant information"""
    
    start_time = time.time()
    
    # Initialize variables
    intent = None
    entities = None
    cypher_query = None
    cypher_result = []
    graph_figure = None
    formatted_cypher = ""
    
    # For embeddings-only retrieval, skip Cypher-based processing
    if retrieval_method == "embeddings":
        # Step 1: Embedding Search Only
        if embedding_model_name == "MPNet":
            embedding_model = model_mpnet
        elif embedding_model_name == "BGE":
            embedding_model = model_bge
        else:
            embedding_model = model_minilm
        embedding_context = retrieve_embedding_search(query, embedding_model, embedding_model_name, graph)
        combined_context = f"Embedding Results:\n{embedding_context}"
    else:
        # For baseline and hybrid, do full Cypher processing
        # Step 1: Intent Classification
        intent = classify_fpl_intents(query)
        
        # Step 2: Entity Extraction
        entities = extract_fpl_entities(query)
        
        # Step 3: Generate Cypher Query
        cypher_query = get_fpl_cypher_query(intent, entities)
        
        # Step 4: Execute Cypher Query
        cypher_result = graph.query(cypher_query)
                
        # Step 5: Format Cypher Result
        formatted_cypher = format_query_result(intent, cypher_result, entities)
        
        # Step 6: Embedding Search (if hybrid)
        embedding_context = ""
        if retrieval_method == "hybrid":
            if embedding_model_name == "MPNet":
                embedding_model = model_mpnet
            elif embedding_model_name == "BGE":
                embedding_model = model_bge
            else:
                embedding_model = model_minilm
            embedding_context = retrieve_embedding_search(query, embedding_model, embedding_model_name, graph)
        
        # Step 7: Combine contexts
        if retrieval_method == "baseline":
            combined_context = f"Cypher Results:\n{formatted_cypher}"
        else:  # hybrid
            combined_context = f"Cypher Results:\n{formatted_cypher}\n\nEmbedding Results:\n{embedding_context}"
    
    # Step 8: Generate LLM Response
    selected_llm = models[model_name]
    qa_chain = generate_qa_chain(selected_llm, combined_context)
    
    response = qa_chain.invoke({"input": query})
    
    # Clean up response from any stray HTML tags
    cleaned_answer = re.sub(r'</?div[^>]*>', '', response["answer"], flags=re.IGNORECASE)
    # Also remove any other common HTML tags that might appear
    cleaned_answer = re.sub(r'<[^>]+>', '', cleaned_answer)
    
    end_time = time.time()
    response_time = end_time - start_time
    
    if retrieval_method != "embeddings" and intent != "best_players_by_metric" and intent != "Worst_players_by_metric":
        graph_figure = create_graph_visualization(cypher_query, cypher_result)
    else:
        graph_figure = None

    return {
        'query': query,
        'intent': intent,
        'entities': entities,
        'cypher_query': '\n'.join(line.strip() for line in cypher_query.splitlines()) if cypher_query else None,
        'cypher_result': cypher_result,
        'formatted_cypher': formatted_cypher,
        'embedding_context': embedding_context if retrieval_method in ["embeddings", "hybrid"] else "",
        'combined_context': combined_context,
        'llm_answer': cleaned_answer,
        'response_time': response_time,
        'model_name': model_name,
        'retrieval_method': retrieval_method,
        'embedding_model': embedding_model_name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'graph_figure': graph_figure
    }

# Header
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 30px;'>
    <h1 style='margin: 0; color: white;'>⚽ Fantasy Premier League Graph-RAG Assistant</h1>
    <p style='margin: 10px 0 0 0; font-size: 18px;'>AI-Powered FPL Analysis with Knowledge Graphs</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Model Selection
    st.markdown("#### 🤖 LLM Model")
    model_name = st.selectbox(
        "Select Large Language Model",
        options=list(models.keys()),
        index=0,
        help="Choose the LLM to generate responses"
    )
    
    # Retrieval Method
    st.markdown("#### 🔍 Retrieval Method")
    retrieval_method = st.selectbox(
        "Select Retrieval Strategy",
        options=["baseline", "embeddings", "hybrid"],
        index=2,
        help="• Baseline: Cypher queries only\n• Embeddings: Vector search only\n• Hybrid: Both methods"
    )
    
    # Embedding Model (only if not baseline)
    if retrieval_method != "baseline":
        st.markdown("#### 📊 Embedding Model")
        embedding_model_name = st.selectbox(
            "Select Embedding Model",
            options=["MiniLM", "MPNet", "BGE"],
            index=0,
            help="• MiniLM: Fast, good quality\n• MPNet: Better quality, slowest\n• BGE: Best quality and speed"
        )
    else:
        embedding_model_name = "MPNet"
    
    st.markdown("---")
    
    # Example Queries
    st.markdown("### 💡 Example Queries")
    
    example_queries = [
        "How many goals did Harry Kane have in gameweek 36 season 2021-22?",
        "What is Liverpool's total bonus points for the 2022-23 season?",
        "Compare Mohamed Salah and Erling Haaland in gameweek 13 season 2022-23 for total points.",
        "Compare Liverpool and Chelsea in gameweek 12 for total points.",
        "When do Arsenal and Man city play each other?",
        "When does Harry Kane play against Liverpool?",
        "Who scored the most goals in the 2021-22 season?",
        "Who are the top 5 forwards by total points in the 2022-23 season?",
        "What position does Mohamed Salah play?",
        "Who are the bottom 3 midfielders by assists above 0?.",

    ]
    
    for i, example in enumerate(example_queries):
        if st.button(f"📝 {example[:40]}...", key=f"example_{i}", use_container_width=True):
            st.session_state.query_input = example
            st.rerun()
    
    st.markdown("---")
    
    # Statistics
    if st.session_state.query_history:
        st.markdown("### 📈 Session Statistics")
        st.metric("Queries Executed", len(st.session_state.query_history))
        avg_time = sum(q['response_time'] for q in st.session_state.query_history) / len(st.session_state.query_history)
        st.metric("Avg Response Time", f"{avg_time:.2f}s")

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🔎 Ask Your Question")
    
    # Query Input
    query = st.text_area(
        "Enter your FPL query:",
        height=100,
        placeholder="e.g., Who scored the most goals in 2022-23?",
        key="query_input"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        submit_button = st.button("🚀 Execute Query", type="primary", use_container_width=True)
    
    with col_btn2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)

with col2:
    st.markdown("### 📊 Current Settings")
    st.info(f"""
    **Model:** {model_name}  
    **Retrieval:** {retrieval_method.title()}  
    **Embeddings:** {embedding_model_name if retrieval_method != 'baseline' else 'N/A'}
    """)

# Clear button functionality
if clear_button:
    st.session_state.current_result = None
    st.rerun()

# Process Query
if submit_button and query:
    with st.spinner("🔄 Processing your query..."):
        try:
            result = process_query(query, model_name, retrieval_method, embedding_model_name)
            st.session_state.current_result = result
            st.session_state.query_history.append(result)
            st.success("✅ Query processed successfully!")
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.exception(e)

# Display Results
if st.session_state.current_result:
    result = st.session_state.current_result
    
    st.markdown("---")
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; color:white;">⏱️</h3>
            <h2 style="margin:10px 0; color:white;">{result['response_time']:.2f}s</h2>
            <p style="margin:0; color:white;">Response Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; color:white;">🎯</h3>
            <h2 style="margin:10px 0; color:white;">{result['intent'] if result['intent'] else 'N/A'}</h2>
            <p style="margin:0; color:white;">Intent</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; color:white;">📦</h3>
            <h2 style="margin:10px 0; color:white;">{len(result['cypher_result']) if result['cypher_result'] else 'N/A'}</h2>
            <p style="margin:0; color:white;">Cypher Results Found</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; color:white;">🤖</h3>
            <h2 style="margin:10px 0; color:white;">{result['model_name']}</h2>
            <p style="margin:0; color:white;">Model Used</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs for Different Views - conditionally show based on retrieval method
    if result['retrieval_method'] == "embeddings":
        # For embeddings only, show fewer tabs
        tab1, tab2 = st.tabs([
            "💬 Final Answer",
            "🔧 Debug Info"
        ])
    else:
        # For baseline and hybrid, show all tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💬 Final Answer",
            "🔍 Graph Context",
            "📝 Cypher Query",
            "📊 Graph Visualization",
            "🔧 Debug Info"
        ])
    
    with tab1:
        st.markdown("### 🎯 LLM Generated Answer")
        # Use a container with custom styling
        answer_container = st.container()
        with answer_container:
            st.markdown(f"""
            <div class="answer-box">
                <p style="margin: 0; white-space: pre-wrap;">{result['llm_answer']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("📋 Context Used for Answer"):
            st.markdown("**Combined Context:**")

            context_html = result['combined_context'].replace('\n', '<br>')

            st.markdown(
                f"""
                <div class="context-box">
                    {context_html}
                </div>
                """,
                unsafe_allow_html=True
            )

    
    # Only show these tabs for baseline and hybrid methods
    if result['retrieval_method'] != "embeddings":
        with tab2:
            st.markdown("### 🔍 Knowledge Graph Retrieved Context")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Extracted Entities")
                st.json(result['entities'])
            
            with col2:
                st.markdown("#### 📊 Formatted Results")

                formatted_html = result['formatted_cypher'].replace('\n', '<br>')
                st.markdown(
                    f"""
                    <div class="context-box">
                        {formatted_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("#### 📦 Raw Cypher Results")
        if result['cypher_result']:
            st.dataframe(result['cypher_result'], use_container_width=True)
        else:
            st.warning("No results returned from Cypher query")

        if result['embedding_context']:
            st.markdown("#### 🔎 Vector Search Context")

            embedding_html = result['embedding_context'].replace('\n', '<br>')
            st.markdown(
                f"""
                <div class="context-box">
                    {embedding_html}
                </div>
                """,
                unsafe_allow_html=True
            )

        
        with tab3:
            st.markdown("### 📝 Executed Cypher Query")
            st.code(result['cypher_query'], language="cypher")
        
        with tab4:
            st.markdown("### 📊 Knowledge Graph Visualization")
            
            if result.get('graph_figure'):
                st.plotly_chart(result['graph_figure'], use_container_width=True)
                
                st.markdown("#### 🎨 Legend")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("🔴 **Player**")
                    st.markdown("🔵 **Team**")
                with col2:
                    st.markdown("🟢 **Fixture**")
                    st.markdown("🟡 **Gameweek**")
                with col3:
                    st.markdown("🟣 **Season**")
                    st.markdown("🟠 **Position**")
            else:
                st.warning("No graph data available to visualize.")
        
        with tab5:
            st.markdown("### 🔧 Debug Information")
            
            with st.expander("🔍 Full Result Object", expanded=False):
                st.json({
                    'query': result['query'],
                    'intent': result['intent'],
                    'entities': result['entities'],
                    'model_name': result['model_name'],
                    'retrieval_method': result['retrieval_method'],
                    'embedding_model': result['embedding_model'],
                    'response_time': result['response_time'],
                    'timestamp': result['timestamp']
                })
            
            with st.expander("📊 Query Execution Details"):
                st.write(f"**Query:** {result['query']}")
                st.write(f"**Intent Classification:** {result['intent']}")
                st.write(f"**Entities Extracted:** {len(result['entities'])} entities")
                st.write(f"**Cypher Results:** {len(result['cypher_result'])} records")
    else:
        # For embeddings-only mode
        with tab2:
            st.markdown("### 🔧 Debug Information")
            
            with st.expander("🔍 Full Result Object", expanded=False):
                st.json({
                    'query': result['query'],
                    'model_name': result['model_name'],
                    'retrieval_method': result['retrieval_method'],
                    'embedding_model': result['embedding_model'],
                    'response_time': result['response_time'],
                    'timestamp': result['timestamp']
                })
            
            with st.expander("📊 Query Execution Details"):
                st.write(f"**Query:** {result['query']}")
                st.write(f"**Retrieval Method:** {result['retrieval_method']}")
                st.write(f"**Embedding Model:** {result['embedding_model']}")
                st.write(f"**Response Time:** {result['response_time']:.3f} seconds")

# Query History
if st.session_state.query_history:
    st.markdown("---")
    st.markdown("### 📜 Query History")
    
    for i, hist in enumerate(reversed(st.session_state.query_history[-5:])):
        with st.expander(f"🕐 {hist['timestamp']} - {hist['query'][:60]}..."):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Model:** {hist['model_name']}")
                st.write(f"**Intent:** {hist['intent']}")
            with col2:
                st.write(f"**Response Time:** {hist['response_time']:.2f}s")
                st.write(f"**Retrieval:** {hist['retrieval_method']}")
            
            if st.button(f"🔄 Reload Query", key=f"reload_{i}"):
                st.session_state.current_result = hist
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p>Built with ❤️ by Aby,Zeina,Habiba,Ehab</p>
    <p style='font-size: 12px;'>Fantasy Premier League Graph-RAG System | Version 1.0</p>
</div>
""", unsafe_allow_html=True)