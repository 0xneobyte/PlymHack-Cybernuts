"""
🦉 OWL ENGINE - Intelligent Dashboard with Live Event Awareness
Real-time correlations, alerts, and predictive intelligence
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import OWL layers
try:
    from layer3_link import build_entity_graph
    from layer4_correlate import CorrelationEngine
    from layer6_7_infer_alert import BaselineModel, EventDetector, AlertSystem
except ImportError as e:
    st.error(f"Import error: {e}")

# Page config
st.set_page_config(
    page_title="🦉 Owl Engine - Intelligence Dashboard",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    .alert-critical {
        background-color: #dc3545;
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        margin: 10px 0;
    }
    .alert-high {
        background-color: #fd7e14;
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        margin: 10px 0;
    }
    .alert-medium {
        background-color: #ffc107;
        padding: 20px;
        border-radius: 10px;
        color: black;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    h1, h2, h3, h4, p {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def run_intelligence_pipeline():
    """Run OWL intelligence layers and cache results"""
    try:
        # Layer 3: Build graph
        graph = build_entity_graph()
        
        # Layer 4: Find correlations
        corr_engine = CorrelationEngine(graph)
        correlations = corr_engine.find_flood_traffic_correlations(
            min_spatial_threshold=0.05,
            min_temporal_threshold=0.2
        )
        
        # Layer 6: Learn baselines
        baseline = BaselineModel()
        baseline.learn_traffic_baselines(graph)
        
        # Layer 7: Detect events
        detector = EventDetector(graph, corr_engine, baseline)
        events = detector.detect_flood_impact_events(min_confidence=0.3)
        
        # Generate alerts
        alert_system = AlertSystem(detector)
        alerts = alert_system.generate_alerts(min_severity="LOW")
        
        return {
            'graph': graph,
            'correlations': correlations,
            'events': events,
            'alerts': alerts,
            'baseline': baseline
        }
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        return None


def display_header():
    """Display dashboard header"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1>🦉 OWL ENGINE</h1>
            <h3>Observant Wisdom Learning System</h3>
            <p style='font-size: 14px; opacity: 0.8;'>
                Real-time Intelligence | Flood-Traffic Correlation | Live Event Detection
            </p>
        </div>
        """, unsafe_allow_html=True)


def display_live_alerts(alerts):
    """Display active alerts banner"""
    if not alerts:
        st.success("✅ No active alerts - All systems normal")
        return
    
    st.markdown("### 🚨 ACTIVE ALERTS")
    
    # Group by severity
    critical = [a for a in alerts if a['severity'] == 'CRITICAL']
    high = [a for a in alerts if a['severity'] == 'HIGH']
    medium = [a for a in alerts if a['severity'] == 'MEDIUM']
    low = [a for a in alerts if a['severity'] == 'LOW']
    
    # Display critical alerts prominently
    for alert in critical:
        st.markdown(f"""
        <div class='alert-critical'>
            <h3>⚠️ {alert['title']}</h3>
            <p>{alert['message']}</p>
            <small>Confidence: {alert['confidence']:.0%} | Time: {alert['timestamp'][:19]}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Show high/medium in expandable sections
    if high or medium or low:
        with st.expander(f"📋 View All Alerts ({len(high)} HIGH, {len(medium)} MEDIUM, {len(low)} LOW)"):
            for alert in high + medium + low:
                severity_emoji = "🔴" if alert['severity'] == 'HIGH' else "🟡" if alert['severity'] == 'MEDIUM' else "🔵"
                st.markdown(f"""
                **{severity_emoji} {alert['severity']}** - {alert['title']}  
                *Confidence: {alert['confidence']:.0%}*  
                {alert['message'][:200]}...
                """)
                st.divider()


def display_metrics(pipeline_data):
    """Display key metrics"""
    st.markdown("### 📊 System Metrics")
    
    graph = pipeline_data['graph']
    correlations = pipeline_data['correlations']
    events = pipeline_data['events']
    alerts = pipeline_data['alerts']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        floods = len([e for e in graph.entities.values() if e['type'] == 'flood_warning'])
        st.metric("🌊 Flood Warnings", floods)
    
    with col2:
        traffic = len([e for e in graph.entities.values() if e['type'] == 'traffic_route'])
        st.metric("🚗 Traffic Routes", traffic)
    
    with col3:
        st.metric("🔗 Correlations", len(correlations))
    
    with col4:
        st.metric("🎯 Events Detected", len(events))
    
    with col5:
        st.metric("🚨 Active Alerts", len(alerts))


def display_correlations(correlations):
    """Display correlation analysis"""
    st.markdown("### 🔗 Flood-Traffic Correlations")
    
    if not correlations:
        st.info("No significant correlations detected")
        return
    
    # Top correlations table
    top_corr = correlations[:10]
    
    df = pd.DataFrame(top_corr)
    display_df = df[[
        'flood_description', 'traffic_route', 'combined_score', 
        'spatial_correlation', 'temporal_correlation', 'lag_hours', 'inference'
    ]].copy()
    
    display_df.columns = ['Flood Area', 'Traffic Route', 'Score', 'Spatial', 'Temporal', 'Lag (h)', 'Assessment']
    
    # Style the dataframe
    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        height=400
    )
    
    # Visualization: Correlation strength matrix
    st.markdown("#### Correlation Heatmap")
    
    # Create pivot table for heatmap
    corr_pivot = df.pivot_table(
        values='combined_score',
        index='flood_area',
        columns='traffic_route',
        aggfunc='max',
        fill_value=0
    )
    
    if not corr_pivot.empty:
        fig = px.imshow(
            corr_pivot,
            color_continuous_scale='RdYlGn_r',
            labels=dict(color="Correlation Score"),
            aspect='auto'
        )
        fig.update_layout(
            title="Flood Impact on Traffic Routes",
            xaxis_title="Traffic Route",
            yaxis_title="Flood Area",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


def display_events(events):
    """Display detected events"""
    st.markdown("### 🎯 Detected Events (Anomalies)")
    
    if not events:
        st.info("No anomalous events detected")
        return
    
    # Event timeline
    df = pd.DataFrame(events)
    
    # Group by severity
    severity_counts = df['severity'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Event Severity Breakdown")
        fig = px.pie(
            values=severity_counts.values,
            names=severity_counts.index,
            color=severity_counts.index,
            color_discrete_map={
                'CRITICAL': '#dc3545',
                'HIGH': '#fd7e14',
                'MEDIUM': '#ffc107',
                'LOW': '#6c757d'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Confidence Distribution")
        fig = px.histogram(
            df,
            x='confidence',
            nbins=20,
            title="Event Confidence Scores"
        )
        fig.update_layout(xaxis_title="Confidence", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed event list
    with st.expander(f"📋 View All {len(events)} Events"):
        for i, event in enumerate(events, 1):
            severity_color = {
                'CRITICAL': '🔴',
                'HIGH': '🟠',
                'MEDIUM': '🟡',
                'LOW': '🔵'
            }.get(event['severity'], '⚪')
            
            st.markdown(f"""
            **{i}. {severity_color} {event['severity']}** - {event['affected_route']}  
            *Confidence: {event['confidence']:.0%} | Delay Factor: {event['delay_factor']}σ*  
            
            **Flood:** {event['flood_area']}  
            **Impact:** {event['current_duration']} (expected: {event['expected_duration']})  
            **Correlation:** Spatial={event['spatial_correlation']:.2f}, Temporal={event['temporal_correlation']:.2f}, Lag={event['lag_hours']}h
            """)
            st.divider()


def display_traffic_analysis(pipeline_data):
    """Display traffic baseline analysis"""
    st.markdown("### 🚦 Traffic Analysis")
    
    baseline = pipeline_data['baseline']
    graph = pipeline_data['graph']
    
    # Get all traffic entities
    traffic_entities = [
        (eid, e) for eid, e in graph.entities.items() 
        if e['type'] == 'traffic_route'
    ]
    
    if not traffic_entities:
        st.info("No traffic data available")
        return
    
    # Create dataframe
    traffic_data = []
    for eid, entity in traffic_entities:
        route = entity['data'].get('route_name')
        duration = entity['data'].get('duration_minutes')
        timestamp = entity.get('timestamp', '')
        
        # Get baseline
        if route in baseline.route_baselines:
            bs = baseline.route_baselines[route]
            z_score = baseline.compute_anomaly_score(route, duration)
            
            traffic_data.append({
                'Route': route,
                'Duration (min)': duration,
                'Baseline Mean': bs['mean'],
                'Baseline Std': bs['std'],
                'Z-Score': z_score,
                'Status': 'DELAYED' if z_score > 2 else 'NORMAL',
                'Timestamp': timestamp[:19]
            })
    
    df = pd.DataFrame(traffic_data)
    
    # Traffic status chart
    fig = px.bar(
        df,
        x='Route',
        y='Duration (min)',
        color='Status',
        color_discrete_map={'NORMAL': '#28a745', 'DELAYED': '#dc3545'},
        title="Current Traffic Conditions vs Baseline"
    )
    
    # Add baseline line
    for route in df['Route'].unique():
        route_data = df[df['Route'] == route].iloc[0]
        fig.add_hline(
            y=route_data['Baseline Mean'],
            line_dash="dash",
            line_color="yellow",
            annotation_text=f"Baseline"
        )
    
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    with st.expander("📊 Detailed Traffic Data"):
        st.dataframe(df, hide_index=True, use_container_width=True)


def display_map(pipeline_data):
    """Display interactive map (placeholder for now)"""
    st.markdown("### 🗺️ Geographic View")
    st.info("🚧 Interactive map with flood zones and affected traffic routes coming soon!")


def main():
    """Main dashboard function"""
    
    # Header
    display_header()
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        
        if st.button("🔄 Refresh Intelligence", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Layers Active")
        st.success("✅ Layer 1: COLLECT")
        st.success("✅ Layer 2: EXTRACT")
        st.success("✅ Layer 3: LINK")
        st.success("✅ Layer 4: CORRELATE")
        st.success("✅ Layer 5: VECTORIZE")
        st.success("✅ Layer 6: INFER")
        st.success("✅ Layer 7: ALERT")
        
        st.markdown("---")
        st.markdown(f"""
        **Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """)
    
    # Run intelligence pipeline
    with st.spinner("🧠 Running intelligence analysis..."):
        pipeline_data = run_intelligence_pipeline()
    
    if not pipeline_data:
        st.error("Failed to load intelligence data")
        return
    
    # Display alerts at top
    display_live_alerts(pipeline_data['alerts'])
    
    st.markdown("---")
    
    # Metrics
    display_metrics(pipeline_data)
    
    st.markdown("---")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔗 Correlations",
        "🎯 Events",
        "🚦 Traffic Analysis",
        "🗺️ Map",
        "💾 Raw Data"
    ])
    
    with tab1:
        display_correlations(pipeline_data['correlations'])
    
    with tab2:
        display_events(pipeline_data['events'])
    
    with tab3:
        display_traffic_analysis(pipeline_data)
    
    with tab4:
        display_map(pipeline_data)
    
    with tab5:
        st.markdown("### 💾 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download Correlations"):
                corr_json = json.dumps(pipeline_data['correlations'], indent=2)
                st.download_button(
                    "Save correlations.json",
                    corr_json,
                    file_name="owl_correlations.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📥 Download Events"):
                events_json = json.dumps(pipeline_data['events'], indent=2)
                st.download_button(
                    "Save events.json",
                    events_json,
                    file_name="owl_events.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("📥 Download Alerts"):
                alerts_json = json.dumps(pipeline_data['alerts'], indent=2)
                st.download_button(
                    "Save alerts.json",
                    alerts_json,
                    file_name="owl_alerts.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()
