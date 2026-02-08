"""
🦉 OWL ENGINE - Palantir Gotham-Style Intelligence Dashboard
Multi-domain threat detection and predictive analytics with integrated timeline
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys
import folium
from streamlit_folium import folium_static
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# Import intelligence modules
try:
    from intelligence.threat_correlator import ThreatCorrelator
    from intelligence.predictive_engine import PredictiveEngine
    from layer3_link import build_entity_graph
    from layer4_correlate import CorrelationEngine
    from layer6_7_infer_alert import BaselineModel, EventDetector, AlertSystem
    from data_manager import DataManager
except ImportError as e:
    st.error(f"Import error: {e}")

# Page config
st.set_page_config(
    page_title="🦉 OWL Palantir Intelligence",
    page_icon="🦉",
    layout="wide",expand
    initial_sidebar_state="collapsed"
)

# Palantir-style dark theme
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        color: #e0e0e0;
    }
    .threat-critical {
        background: linear-gradient(135deg, #8b0000 0%, #dc143c 100%);
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #ff0000;
        margin: 10px 0;
        color: white;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(255, 0, 0, 0.3);
    }
    .threat-high {
        background: linear-gradient(135deg, #cc6600 0%, #ff8c00 100%);
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #ff6600;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 6px rgba(255, 102, 0, 0.3);
    }
    .threat-medium {
        background: linear-gradient(135deg, #cc9900 0%, #ffcc00 100%);
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ffaa00;
        margin: 10px 0;
        color: #1a1a1a;
        box-shadow: 0 4px 6px rgba(255, 204, 0, 0.3);
    }
    .metric-card {
        background: rgba(26, 31, 58, 0.8);
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #2a3f5f;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .domain-card {
        background: rgba(15, 20, 40, 0.9);
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #1a3f5f;
        margin: 10px 0;
    }
    h1, h2, h3, h4, p, span, div {
        color: #e0e0e0 !important;
    }
    .stMetric {
        background: rgba(26, 31, 58, 0.6);
        padding: 15px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(26, 31, 58, 0.8);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(42, 63, 95, 0.6);
        color: #e0e0e0;
        border-radius: 4px 4px 0 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=180)  # Cache for 3 minutes
def run_full_intelligence_pipeline():
    """Run complete Palantir-level intelligence analysis"""
    try:
        # Multi-domain threat correlation
        correlator = ThreatCorrelator()
        all_events = correlator.load_all_events()
        compound_threats = correlator.detect_compound_threats(min_domains=2, time_window_hours=24)
        
        # Predictive intelligence
        predictor = PredictiveEngine()
        predictions = predictor.generate_all_predictions(all_events)
        
        # Standard correlation (flood-traffic)
        try:
            graph = build_entity_graph()
            corr_engine = CorrelationEngine(graph)
            correlations = corr_engine.find_flood_traffic_correlations()
            baseline = BaselineModel()
            baseline.learn_traffic_baselines(graph)
            detector = EventDetector(graph, corr_engine, baseline)
            events = detector.detect_flood_impact_events(min_confidence=0.3)
            alert_system = AlertSystem(detector)
            alerts = alert_system.generate_alerts(min_severity="LOW")
        except:
            correlations = []
            events = []
            alerts = []
        
        return {
            'all_events': all_events,
            'compound_threats': compound_threats,
            'predictions': predictions,
            'correlations': correlations,
            'events': events,
            'alerts': alerts
        }
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        return None


def display_palantir_header():
    """Palantir Gotham-style header"""
    st.markdown("""
    <div style='text-align: center; padding: 30px 20px 10px 20px;'>
        <h1 style='font-size: 48px; font-weight: 300; letter-spacing: 8px; color: #00d4ff !important;'>
            🦉 OWL ENGINE
        </h1>
        <h3 style='font-weight: 200; color: #888 !important; margin-top: -10px;'>
            URBAN INTELLIGENCE PLATFORM
        </h3>
        <p style='font-size: 12px; color: #555 !important; letter-spacing: 2px;'>
            MULTI-DOMAIN THREAT DETECTION • PREDICTIVE ANALYTICS • REAL-TIME CORRELATION
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_threat_board(compound_threats, alerts):
    """Main threat board - Palantir style"""
    st.markdown("### 🚨 ACTIVE THREAT BOARD")
    
    # Combine compound threats and alerts
    all_threats = []
    
    # Add compound threats
    for threat in compound_threats:
        all_threats.append({
            'type': 'COMPOUND',
            'severity': threat['severity'],
            'score': threat['threat_score'],
            'title': f"{threat['threat_category'].replace('_', ' ').upper()}",
            'description': threat['description'],
            'location': threat['location_name'],
            'domains': threat['domains'],
            'confidence': threat['confidence'],
            'timestamp': threat['timestamp']
        })
    
    # Add alerts
    for alert in alerts:
        all_threats.append({
            'type': 'ALERT',
            'severity': alert['severity'],
            'score': alert['confidence'] * 10,
            'title': alert['title'],
            'description': alert['message'],
            'location': alert['event_data'].get('flood_area', 'Unknown'),
            'domains': ['flood', 'traffic'],
            'confidence': alert['confidence'],
            'timestamp': alert['timestamp']
        })
    
    # Sort by severity and score
    severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    all_threats.sort(key=lambda x: (severity_order.get(x['severity'], 3), -x['score']))
    
    if not all_threats:
        st.success("✅ NO ACTIVE THREATS - All systems nominal")
        return
    
    # Display threats
    for threat in all_threats[:10]:  # Top 10
        severity_class = f"threat-{threat['severity'].lower()}"
        
        st.markdown(f"""
        <div class='{severity_class}'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h3 style='margin: 0;'>⚠️ {threat['severity']} - {threat['title']}</h3>
                    <p style='margin: 5px 0; font-size: 14px;'>{threat['description'][:200]}</p>
                </div>
                <div style='text-align: right; min-width: 150px;'>
                    <div style='font-size: 24px; font-weight: bold;'>{threat['score']:.1f}</div>
                    <div style='font-size: 11px; opacity: 0.8;'>THREAT SCORE</div>
                </div>
            </div>
            <div style='margin-top: 10px; font-size: 12px; opacity: 0.9;'>
                📍 {threat['location']} | 🔗 Domains: {', '.join(threat['domains'])} | 
                📊 Confidence: {threat['confidence']:.0%} | 
                ⏱️ {threat['timestamp'][:19]}
            </div>
        </div>
        """, unsafe_allow_html=True)


def display_timeline_page():
    """Timeline Analysis Page"""
    st.title("⏱️ Unified Event Timeline")
    st.markdown("### Chronological view of ALL traffic, flood, weather, and intelligence events")
    
    # Initialize Data Manager
    try:
        dm = DataManager()
    except Exception as e:
        st.error(f"Failed to initialize Data Manager: {e}")
        return
    
    # Load unified timeline
    with st.spinner("📊 Loading timeline data from all sources..."):
        timeline = dm.create_unified_timeline()
    
    if timeline.empty:
        st.warning("⚠️ No timeline data available. Start data collection to populate the timeline.")
        st.info("💡 Run `python continuous_collector.py` to start collecting data")
        
        # Show what data exists
        st.markdown("### 📁 Available Data Sources")
        traffic = dm.load_all_traffic_history()
        floods = dm.load_all_flood_warnings()
        st.write(f"- Traffic records: {len(traffic)}")
        st.write(f"- Flood warnings: {len(floods)}")
        return
    
    # Filters in columns
    st.markdown("### 🔍 Filters")
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        # Date range filter
        min_date = timeline['timestamp'].min().date()
        max_date = timeline['timestamp'].max().date()
        date_range = st.date_input(
            "📅 Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="timeline_date_range"
        )
    
    with filter_col2:
        # Event type filter
        event_types = st.multiselect(
            "📊 Event Types",
            options=sorted(timeline['event_type'].unique()),
            default=list(timeline['event_type'].unique()),
            key="timeline_event_types"
        )
    
    with filter_col3:
        # Domain filter
        domains = st.multiselect(
            "🏗️ Domains",
            options=sorted(timeline['domain'].unique()),
            default=list(timeline['domain'].unique()),
            key="timeline_domains"
        )
    
    with filter_col4:
        # Severity filter
        severity_range = st.slider(
            "⚠️ Severity",
            min_value=1,
            max_value=5,
            value=(1, 5),
            key="timeline_severity"
        )
    
    # Apply filters
    filtered = timeline.copy()
    
    if len(date_range) == 2:
        filtered = filtered[
            (filtered['timestamp'].dt.date >= date_range[0]) &
            (filtered['timestamp'].dt.date <= date_range[1])
        ]
    
    filtered = filtered[
        (filtered['event_type'].isin(event_types)) &
        (filtered['domain'].isin(domains)) &
        (filtered['severity'] >= severity_range[0]) &
        (filtered['severity'] <= severity_range[1])
    ]
    
    # Metrics row
    st.markdown("---")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("📊 Total", len(filtered))
    with col2:
        st.metric("🚗 Traffic", len(filtered[filtered['event_type'] == 'traffic']))
    with col3:
        st.metric("🌊 Floods", len(filtered[filtered['event_type'] == 'flood']))
    with col4:
        st.metric("🌤️ Weather", len(filtered[filtered['event_type'] == 'weather']))
    with col5:
        st.metric("🌫️ Air", len(filtered[filtered['event_type'] == 'air_quality']))
    with col6:
        st.metric("⚠️ High Sev", len(filtered[filtered['severity'] >= 4]))
    
    st.markdown("---")
    
    # Main timeline scatter plot
    st.markdown("### 📈 Event Timeline")
    
    fig_timeline = px.scatter(
        filtered,
        x='timestamp',
        y='severity',
        color='event_type',
        size='severity',
        hover_data=['description', 'location', 'domain', 'source'],
        title='All Events Over Time (Size = Severity)',
        height=500,
        color_discrete_map={
            'traffic': '#FF6B6B',
            'flood': '#4ECDC4',
            'weather': '#FFD93D',
            'air_quality': '#95E1D3',
            'health': '#F38181'
        }
    )
    
    fig_timeline.update_layout(
        xaxis_title="Time",
        yaxis_title="Severity (1-5)",
        hovermode='closest'
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Two-column charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 📊 Hourly Distribution")
        filtered_copy = filtered.copy()
        filtered_copy['hour'] = filtered_copy['timestamp'].dt.hour
        hourly = filtered_copy.groupby(['hour', 'event_type']).size().reset_index(name='count')
        
        fig_hourly = px.bar(
            hourly,
            x='hour',
            y='count',
            color='event_type',
            title='Events by Hour',
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col_right:
        st.markdown("### 🎯 By Domain")
        domain_counts = filtered['domain'].value_counts().reset_index()
        domain_counts.columns = ['domain', 'count']
        
        fig_domain = px.pie(
            domain_counts,
            values='count',
            names='domain',
            title='Events by Domain',
            height=400,
            hole=0.4
        )
        st.plotly_chart(fig_domain, use_container_width=True)
    
    # Traffic trends if available
    traffic_data = filtered[filtered['event_type'] == 'traffic']
    if not traffic_data.empty:
        st.markdown("### 🚗 Traffic Duration Trends")
        
        traffic_data = traffic_data.copy()
        traffic_data['duration_minutes'] = traffic_data['value'].fillna(0)
        
        if traffic_data['duration_minutes'].sum() > 0:
            fig_traffic = px.line(
                traffic_data,
                x='timestamp',
                y='duration_minutes',
                color='location',
                title='Traffic Duration by Route',
                markers=True,
                height=400
            )
            st.plotly_chart(fig_traffic, use_container_width=True)
    
    # Correlation matrix
    st.markdown("### 🔗 Event Co-Occurrence")
    
    try:
        filtered_corr = filtered.copy()
        filtered_corr['time_window'] = filtered_corr['timestamp'].dt.floor('h')
        
        pivot = pd.crosstab(filtered_corr['time_window'], filtered_corr['event_type'])
        
        if not pivot.empty and len(pivot.columns) > 1:
            corr = pivot.corr()
            
            fig_corr = px.imshow(
                corr,
                labels=dict(color="Correlation"),
                x=corr.columns,
                y=corr.columns,
                color_continuous_scale='RdBu_r',
                title="Event Type Correlations (1-hour windows)",
                height=400
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.info("Correlation analysis requires more data points")
    
    # Event table
    st.markdown("---")
    st.markdown("### 📋 Event Details")
    
    display_cols = ['timestamp', 'event_type', 'domain', 'severity', 'description', 'location', 'source']
    available_cols = [col for col in display_cols if col in filtered.columns]
    
    display_df = filtered[available_cols].copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    st.dataframe(
        display_df.sort_values('timestamp', ascending=False),
        use_container_width=True,
        height=400
    )
    
    # Export buttons
    st.markdown("---")
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download CSV",
            csv,
            f"timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col_exp2:
        export_data = filtered.copy()
        export_data['timestamp'] = export_data['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        export_data['data'] = export_data['data'].apply(lambda x: str(x) if pd.notna(x) else '')
        
        json_str = json.dumps(export_data.to_dict('records'), indent=2)
        st.download_button(
            "📥 Download JSON",
            json_str,
            f"timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
      Continue with original threat intelligence view
    if page == "🔮 Predictions":
        display_predictive_intelligence(intel['predictions'])
        return
    
    if page == "🗺️ Geo-Intel":
        display_intelligence_map(intel['all_events'], intel['compound_threats'])
        return
    
    if page == "🔗 Correlations"
    # Load flood warnings
    flood_dir = Path(__file__).parent / "owl_data"
    if flood_dir.exists():
        for date_folder in flood_dir.iterdir():
            if date_folder.is_dir():
                flood_warning_dir = date_folder / "flood_warnings"
                if flood_warning_dir.exists():
                    for json_file in flood_warning_dir.glob("*.json"):
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                raw_data['flood_warnings'].append({
                                    'timestamp': data.get('metadata', {}).get('timestamp', 'N/A'),
                                    'area': data.get('raw_data', {}).get('floodArea', {}).get('county', 'N/A'),
                                    'severity': data.get('raw_data', {}).get('severity', 'N/A'),
                                    'description': data.get('raw_data', {}).get('description', 'N/A'),
                                    'message': data.get('raw_data', {}).get('message', 'N/A')[:150] + '...' if len(data.get('raw_data', {}).get('message', '')) > 150 else data.get('raw_data', {}).get('message', 'N/A')
                                })
                        except:
                            pass
    
    # Load traffic data
    traffic_dir = Path(__file__).parent / "traffic_data_collected"
    if traffic_dir.exists():
        for json_file in traffic_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for route in data:
                            raw_data['traffic'].append({
                                'timestamp': route.get('timestamp', 'N/A'),
                                'route': route.get('route_name', 'N/A'),
                                'origin': route.get('origin', 'N/A'),
                                'destination': route.get('destination', 'N/A'),
                                'duration': route.get('duration', 'N/A'),
                                'status': route.get('status', 'N/A')
                            })
            except:
                pass
    
    return raw_data


def display_domain_status(all_events):
    """Domain-by-domain status grid"""
    st.markdown("### 📊 DOMAIN STATUS")
    
    # Group events by domain
    domain_stats = {}
    for event in all_events:
        domain = event.get('domain', 'unknown')
        if domain not in domain_stats:
            domain_stats[domain] = {'count': 0, 'high_severity': 0}
        
        domain_stats[domain]['count'] += 1
        if event.get('severity', 3) <= 2:
            domain_stats[domain]['high_severity'] += 1
    
    # Display grid
    cols = st.columns(5)
    
    domain_config = {
        'flood': {'icon': '🌊', 'color': '#3498db'},
        'environmental': {'icon': '🌫️', 'color': '#9b59b6'},
        'infrastructure': {'icon': '🏗️', 'color': '#e67e22'},
        'social': {'icon': '📱', 'color': '#1abc9c'},
        'traffic': {'icon': '🚗', 'color': '#e74c3c'}
    }
    
    for i, (domain, stats) in enumerate(domain_stats.items()):
        config = domain_config.get(domain, {'icon': '❓', 'color': '#95a5a6'})
        
        with cols[i % 5]:
            status = "ALERT" if stats['high_severity'] > 0 else "NORMAL"
            status_color = "#ff6b6b" if status == "ALERT" else "#51cf66"
            
            st.markdown(f"""
            <div class='domain-card'>
                <div style='font-size: 32px; text-align: center;'>{config['icon']}</div>
                <div style='text-align: center; margin-top: 10px;'>
                    <div style='font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px;'>
                        {domain}
                    </div>
                    <div style='font-size: 24px; font-weight: bold; color: {config["color"]};'>
                        {stats['count']}
                    </div>
                    <div style='font-size: 10px; color: {status_color}; margin-top: 5px;'>
                        ● {status}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def display_predictive_intelligence(predictions):
    """Predictive threat forecasting"""
    st.markdown("### 🔮 PREDICTIVE INTELLIGENCE")
    
    if not predictions:
        st.info("No immediate predictions - monitoring for emerging patterns")
        return
    
    # Filter by prediction type
    prediction_types = {}
    for pred in predictions:
        ptype = pred.get('prediction_type', 'unknown')
        if ptype not in prediction_types:
            prediction_types[ptype] = []
        prediction_types[ptype].append(pred)
    
    # Display predictions
    for ptype, preds in prediction_types.items():
        with st.expander(f"🎯 {ptype.replace('_', ' ').title()} ({len(preds)} predictions)", expanded=True):
            for pred in preds[:5]:
                risk_score = pred.get('risk_score', 0)
                risk_color = "#ff4444" if risk_score > 0.7 else "#ffaa00" if risk_score > 0.4 else "#4444ff"
                
                st.markdown(f"""
                <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid {risk_color};'>
                    <div style='font-weight: bold; font-size: 16px; margin-bottom: 5px;'>
                        {pred.get('prediction', 'Unknown prediction')}
                    </div>
                    <div style='font-size: 13px; margin: 8px 0;'>
                        📍 {pred.get('location', 'Unknown')} | 
                        📊 Risk: {risk_score:.0%} | 
                        🎯 Confidence: {pred.get('confidence', 0):.0%}
                    </div>
                    <div style='font-size: 12px; color: #aaa; margin-top: 8px;'>
                        💡 Recommendation: {pred.get('recommendation', 'Monitor situation')}
                    </div>
                </div>
                """, unsafe_allow_html=True)


def display_intelligence_map(all_events, compound_threats):
    """Geographic threat visualization"""
    st.markdown("### 🗺️ GEO-INTELLIGENCE MAP")
    
    # Create map centered on London
    m = folium.Map(
        location=[51.5074, -0.1278],
        zoom_start=11,
        tiles='CartoDB dark_matter'
    )
    
    # Plot events by domain
    domain_colors = {
        'flood': 'blue',
        'environmental': 'purple',
        'infrastructure': 'orange',
        'social': 'green',
        'traffic': 'red'
    }
    
    for event in all_events:
        lat = event.get('lat')
        lon = event.get('lon')
        
        if lat and lon:
            domain = event.get('domain', 'unknown')
            color = domain_colors.get(domain, 'gray')
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                popup=f"{domain}: {event.get('location_name', 'Unknown')}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6
            ).add_to(m)
    
    # Plot compound threats as larger markers
    for threat in compound_threats:
        lat = threat.get('center_lat')
        lon = threat.get('center_lon')
        
        if lat and lon:
            severity_color = {
                'CRITICAL': 'darkred',
                'HIGH': 'red',
                'MEDIUM': 'orange',
                'LOW': 'yellow'
            }.get(threat['severity'], 'gray')
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=15,
                popup=f"<b>{threat['threat_category']}</b><br>{threat['description']}",
                color=severity_color,
                fill=True,
                fillColor=severity_color,
                fillOpacity=0.8
            ).add_to(m)
    
    folium_static(m, width=1200, height=500)


def main():
    """Main Palantir dashboard with timeline integration"""
    
    # Sidebar Navigation
    st.sidebar.title("🦉 OWL NAVIGATION")
    page = st.sidebar.radio(
        "Select View",
        ["🎯 Threat Intelligence", "⏱️ Timeline Analysis", "🔮 Predictions", "🗺️ Geo-Intel", "🔗 Correlations", "📊 Analytics", "📁 Raw Data"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ CONTROLS")
    
    if st.sidebar.button("♻️ REFRESH DATA", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 ACTIVE LAYERS")
    layers = ["COLLECT", "EXTRACT", "LINK", "CORRELATE", "VECTORIZE", "INFER", "ALERT", "PREDICT"]
    for layer in layers:
        st.sidebar.success(f"✅ {layer}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Last Update:**  \n{datetime.now().strftime('%H:%M:%S')}")
    
    # Header
    display_palantir_header()
    
    # ==================== TIMELINE PAGE ====================
    if page == "⏱️ Timeline Analysis":
        display_timeline_page()
        return
    
    # ==================== THREAT INTELLIGENCE PAGE (ORIGINAL) ====================
    if page != "🎯 Threat Intelligence":
        # For other pages, keep the original layout but show message
        st.info(f"📍 {page} view - Integration in progress")
    
    # Run intelligence pipeline
    with st.spinner("🧠 Running multi-domain intelligence analysis..."):
        intel = run_full_intelligence_pipeline()
    
    if not intel:
        st.error("Failed to load intelligence data")
        return
    
    # Main threat board
    display_threat_board(intel['compound_threats'], intel['alerts'])
    
    st.markdown("---")
    
    # Domain status grid
    display_domain_status(intel['all_events'])
    
    st.markdown("---")
    
    # Tabs for detailed views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 PREDICTIONS",
        "🗺️ GEO-INTEL",
        "🔗 CORRELATIONS",
        "📊 ANALYTICS",
        "📁 RAW DATA"
    ])
    
    with tab1:
        display_predictive_intelligence(intel['predictions'])
    
    with tab2:
        display_intelligence_map(intel['all_events'], intel['compound_threats'])
    
    with tab3:
        st.markdown("#### Flood-Traffic Correlations")
        if intel['correlations']:
            df = pd.DataFrame(intel['correlations'][:20])
            display_df = df[[
                'flood_description', 'traffic_route', 'combined_score',
                'spatial_correlation', 'temporal_correlation', 'inference'
            ]].copy()
            display_df.columns = ['Flood', 'Route', 'Score', 'Spatial', 'Temporal', 'Assessment']
            st.dataframe(display_df, use_container_width=True, height=400)
        else:
            st.info("No correlations detected")
    
    with tab4:
        st.markdown("#### System Analytics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Events", len(intel['all_events']))
        with col2:
            st.metric("Compound Threats", len(intel['compound_threats']))
        with col3:
            st.metric("Predictions", len(intel['predictions']))
        with col4:
            st.metric("Active Alerts", len(intel['alerts']))
        
        # Event timeline
        if intel['all_events']:
            st.markdown("#### Event Timeline (Last 24h)")
            
            events_df = pd.DataFrame([
                {
                    'timestamp': e['timestamp'][:19],
                    'domain': e['domain'],
                    'severity': e.get('severity', 3)
                } 
                for e in intel['all_events']
            ])
            
            fig = px.scatter(
                events_df,
                x='timestamp',
                y='domain',
                color='domain',
                size=[(4-s)*5 for s in events_df['severity']],
                title="Event Distribution by Domain and Time"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("#### 📁 Raw Collected Data")
        st.info("Showing all raw data collected from API sources")
        
        # Load raw data
        raw_data = load_raw_data()
        
        # Flood Warnings
        st.markdown("### 🌊 Flood Warnings")
        if raw_data['flood_warnings']:
            df_flood = pd.DataFrame(raw_data['flood_warnings'])
            st.dataframe(df_flood, use_container_width=True, height=300)
            st.markdown(f"**Total Records:** {len(raw_data['flood_warnings'])}")
        else:
            st.warning("No flood warning data available")
        
        st.markdown("---")
        
        # Traffic Data
        st.markdown("### 🚗 Traffic Conditions")
        if raw_data['traffic']:
            df_traffic = pd.DataFrame(raw_data['traffic'])
            st.dataframe(df_traffic, use_container_width=True, height=300)
            st.markdown(f"**Total Records:** {len(raw_data['traffic'])}")
            
            # Traffic duration chart
            try:
                if 'duration' in df_traffic.columns and len(df_traffic) > 0:
                    df_traffic_chart = df_traffic.copy()
                    df_traffic_chart['duration_min'] = df_traffic_chart['duration'].str.extract(r'(\d+)').astype(float)
                    
                    fig_traffic = px.bar(
                        df_traffic_chart,
                        x='route',
                        y='duration_min',
                        title="Current Traffic Durations by Route",
                        labels={'duration_min': 'Duration (minutes)', 'route': 'Route'},
                        color='duration_min',
                        color_continuous_scale='reds'
                    )
                    fig_traffic.update_layout(height=400)
                    st.plotly_chart(fig_traffic, use_container_width=True)
            except Exception as e:
                st.info(f"Chart unavailable: {str(e)}")
        else:
            st.warning("No traffic data available")
        
        st.markdown("---")
        
        # All Events (Processed)
        st.markdown("### 🧠 Processed Intelligence Events")
        if intel['all_events']:
            events_list = []
            for event in intel['all_events'][:100]:  # Show first 100
                events_list.append({
                    'Timestamp': event.get('timestamp', 'N/A')[:19],
                    'Domain': event.get('domain', 'N/A'),
                    'Severity': event.get('severity', 'N/A'),
                    'Location': event.get('location', 'N/A'),
                    'Description': str(event.get('description', 'N/A'))[:100]
                })
            
            df_events = pd.DataFrame(events_list)
            st.dataframe(df_events, use_container_width=True, height=400)
            st.markdown(f"**Total Events:** {len(intel['all_events'])} (showing first 100)")
        else:
            st.warning("No processed events available")
        
        # Download buttons
        st.markdown("---")
        st.markdown("### 📥 Export Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if raw_data['flood_warnings']:
                csv_flood = pd.DataFrame(raw_data['flood_warnings']).to_csv(index=False)
                st.download_button(
                    "⬇️ Export Flood Data (CSV)",
                    csv_flood,
                    "flood_warnings.csv",
                    "text/csv"
                )
        
        with col2:
            if raw_data['traffic']:
                csv_traffic = pd.DataFrame(raw_data['traffic']).to_csv(index=False)
                st.download_button(
                    "⬇️ Export Traffic Data (CSV)",
                    csv_traffic,
                    "traffic_data.csv",
                    "text/csv"
                )
        
        with col3:
            if intel['all_events']:
                csv_events = pd.DataFrame(events_list).to_csv(index=False)
                st.download_button(
                    "⬇️ Export Events (CSV)",
                    csv_events,
                    "intelligence_events.csv",
                    "text/csv"
                )


if __name__ == "__main__":
    main()
