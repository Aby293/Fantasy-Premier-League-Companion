import streamlit as st
import time
from datetime import datetime
import plotly.graph_objects as go
import networkx as nx
from typing import Dict, List, Tuple
import json
import re

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
        graph, classifier, models, model_minilm, model_mpnet,
        classify_fpl_intents, extract_fpl_entities, get_fpl_cypher_query,
        format_query_result, retrieve_embedding_search, generate_qa_chain
    )
    
    # Store in session state to avoid reimporting
    st.session_state.graph = graph
    st.session_state.classifier = classifier
    st.session_state.models = models
    st.session_state.model_minilm = model_minilm
    st.session_state.model_mpnet = model_mpnet
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
classifier = st.session_state.classifier
models = st.session_state.models
model_minilm = st.session_state.model_minilm
model_mpnet = st.session_state.model_mpnet
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
    """Create a graph visualization using Plotly with improved entity extraction"""
    
    G = nx.DiGraph()
    
    # Parse the Cypher result to extract nodes and relationships
    nodes = set()
    edges = []
    node_labels = {}
    node_colors = {}
    node_types = {}
    
    color_map = {
        'Player': '#FF6B6B',
        'Team': '#4ECDC4',
        'Fixture': '#95E1D3',
        'Gameweek': '#FFE66D',
        'Season': '#A8E6CF',
        'Position': '#C7CEEA',
        'Unknown': '#95A5A6'
    }
    
    # Extract entities from results with improved logic
    for record in cypher_result:
        for key, value in record.items():
            # Handle Neo4j node objects
            if isinstance(value, dict):
                if 'name' in value:
                    node_id = value['name']
                    nodes.add(node_id)
                    node_labels[node_id] = node_id
                    # Infer type from key name
                    if 'player' in key.lower():
                        node_types[node_id] = 'Player'
                    elif 'team' in key.lower() or key in ['h', 'a', 'home', 'away']:
                        node_types[node_id] = 'Team'
                    elif 'fixture' in key.lower():
                        node_types[node_id] = 'Fixture'
                    else:
                        node_types[node_id] = 'Unknown'
                    node_colors[node_id] = color_map.get(node_types[node_id], '#95A5A6')
                
                elif 'player_name' in value:
                    node_id = value['player_name']
                    nodes.add(node_id)
                    node_labels[node_id] = node_id
                    node_types[node_id] = 'Player'
                    node_colors[node_id] = color_map['Player']
            
            # Handle string values
            elif isinstance(value, str):
                if key in ['player', 'player_name']:
                    nodes.add(value)
                    node_labels[value] = value
                    node_types[value] = 'Player'
                    node_colors[value] = color_map['Player']
                elif key in ['team', 'team_name', 'home_team', 'away_team']:
                    nodes.add(value)
                    node_labels[value] = value
                    node_types[value] = 'Team'
                    node_colors[value] = color_map['Team']
    
    # If we have specific relationships, create edges
    for record in cypher_result:
        # Create edges for player-team relationships
        if 'player' in record and 'team' in record:
            player = record['player']
            team = record['team']
            if player in nodes and team in nodes:
                edges.append((player, team))
        
        # Create edges for home/away team relationships
        if 'home_team' in record and 'away_team' in record:
            home = record['home_team']
            away = record['away_team']
            if home in nodes and away in nodes:
                edges.append((home, away))
    
    # If no specific edges, create a simple chain
    if len(edges) == 0 and len(nodes) > 1:
        node_list = list(nodes)
        for i in range(len(node_list) - 1):
            edges.append((node_list[i], node_list[i + 1]))
    
    # Add nodes and edges to graph
    for node in nodes:
        G.add_node(node)
    for edge in edges:
        if edge[0] in nodes and edge[1] in nodes:
            G.add_edge(edge[0], edge[1])
    
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
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines',
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
        data=[edge_trace, node_trace],
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
            height=500
        )
    )
    
    return fig

def process_query(query: str, model_name: str, retrieval_method: str, embedding_model_name: str):
    """Process the query and return all relevant information"""
    
    start_time = time.time()
    
    # Step 1: Intent Classification
    intent = classify_fpl_intents(classifier, query)
    
    # Step 2: Entity Extraction
    entities = extract_fpl_entities(query)
    
    # Step 3: Generate Cypher Query
    cypher_query = get_fpl_cypher_query(intent, entities)
    
    # Step 4: Execute Cypher Query
    cypher_result = graph.query(cypher_query)
    
    # Step 5: Format Cypher Result
    formatted_cypher = format_query_result(intent, cypher_result, entities)
    
    # Step 6: Embedding Search (if needed)
    embedding_context = ""
    if retrieval_method in ["embeddings", "hybrid"]:
        embedding_model = model_mpnet if embedding_model_name == "MPNet" else model_minilm
        embedding_context = retrieve_embedding_search(query, embedding_model, embedding_model_name)
    
    # Step 7: Combine contexts
    if retrieval_method == "baseline":
        combined_context = f"Cypher Results:\n{formatted_cypher}"
    elif retrieval_method == "embeddings":
        combined_context = f"Embedding Results:\n{embedding_context}"
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
    
    return {
        'query': query,
        'intent': intent,
        'entities': entities,
        'cypher_query': cypher_query,
        'cypher_result': cypher_result,
        'formatted_cypher': formatted_cypher,
        'embedding_context': embedding_context,
        'combined_context': combined_context,
        'llm_answer': cleaned_answer,
        'response_time': response_time,
        'model_name': model_name,
        'retrieval_method': retrieval_method,
        'embedding_model': embedding_model_name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
            options=["MPNet", "MiniLM"],
            index=0,
            help="• MPNet: Higher quality, slower\n• MiniLM: Faster, good quality"
        )
    else:
        embedding_model_name = "MPNet"
    
    st.markdown("---")
    
    # Example Queries
    st.markdown("### 💡 Example Queries")
    
    example_queries = [
        "Show me how Mohamed Salah performed in gameweek 5, including total_points.",
        "What is Chelsea's total bonus points for the 2022-23 season?",
        "Compare Erling Haaland and Mohamed Salah for the 2022-23 season by goals.",
        "Compare Liverpool and Chelsea in gameweek 12 for total points.",
        "When do Arsenal and Man city play each other?",
        "Who are the top 5 forwards by total points in the 2022-23 season?"
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
            <h2 style="margin:10px 0; color:white;">{result['intent']}</h2>
            <p style="margin:0; color:white;">Intent</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; color:white;">📦</h3>
            <h2 style="margin:10px 0; color:white;">{len(result['cypher_result'])}</h2>
            <p style="margin:0; color:white;">Results Found</p>
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
    
    # Tabs for Different Views
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
        st.markdown(f"""
        <div class="cypher-box">
            {result['cypher_query']}
        </div>
        """, unsafe_allow_html=True)
        
        # Copy button
        if st.button("📋 Copy Query"):
            st.code(result['cypher_query'], language="cypher")
            st.success("Query copied! You can now paste it into Neo4j Browser")
    
    with tab4:
        st.markdown("### 📊 Knowledge Graph Visualization")
        
        if result['cypher_result']:
            try:
                fig = create_graph_visualization(result['cypher_query'], result['cypher_result'])
                st.plotly_chart(fig, use_container_width=True)
                
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
            except Exception as e:
                st.warning(f"Could not generate graph visualization: {str(e)}")
                st.info("This may happen if the query returns scalar values instead of graph entities.")
        else:
            st.warning("No graph data to visualize")
    
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
    <p>Built with ❤️ using Streamlit, Neo4j, and LangChain</p>
    <p style='font-size: 12px;'>Fantasy Premier League Graph-RAG System | Version 1.0</p>
</div>
""", unsafe_allow_html=True)