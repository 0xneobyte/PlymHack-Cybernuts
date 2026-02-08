"""  
OWL ENGINE - Palantir Gotham-Style Intelligence Dashboard
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
    page_title="OWL Palantir Intelligence",
    page_icon="🦉",
    layout="wide",
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
        # Use DataManager for consistent data loading (same as timeline)
        dm = DataManager()
        
        # Load unified timeline with temporal weighting
        timeline_df = dm.create_unified_timeline()
        
        if timeline_df.empty:
            # Return empty but valid structure
            return {
                'all_events': [],
                'compound_threats': [],
                'predictions': [],
                'correlations': [],
                'events': [],
                'alerts': []
            }
        
        # Convert timeline DataFrame to event list for correlator
        all_events = []
        for _, row in timeline_df.iterrows():
            # Extract road_name from data dict if available
            road_name = ''
            if pd.notna(row.get('data')) and isinstance(row.get('data'), dict):
                road_name = row.get('data').get('road_name', '')
            
            event = {
                'event_id': f"{row['event_type']}_{row['timestamp']}",
                'domain': row['domain'],
                'threat_type': row['event_type'],
                'event_type': row['event_type'],  # Needed for cascade prediction
                'timestamp': row['timestamp'].isoformat(),
                'location_name': row.get('location', 'Unknown'),
                'location': row.get('location', 'Unknown'),  # Needed for road disruptions
                'lat': row.get('lat', 51.5074),
                'lon': row.get('lon', -0.1278),
                'severity': row.get('severity', 3),
                'confidence': 0.8,
                'source_type': 'sensor',
                'description': row.get('description', ''),
                'temporal_weight': row.get('temporal_weight', 1.0),
                'raw_data': row.get('data', {}),
                # Road disruption specific fields (from timeline columns)
                'category': row.get('category', ''),
                'has_closures': row.get('has_closures', False),
                'road_name': road_name
            }
            all_events.append(event)
        
        # Multi-domain threat correlation
        correlator = ThreatCorrelator()
        correlator.events = all_events  # Use loaded events
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
        import traceback
        st.error(traceback.format_exc())
        return {
            'all_events': [],
            'compound_threats': [],
            'predictions': [],
            'correlations': [],
            'events': [],
            'alerts': []
        }


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
    st.markdown("### ACTIVE THREAT BOARD")
    
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
        st.success("NO ACTIVE THREATS - All systems nominal")
        return
    
    # Display threats
    for threat in all_threats[:10]:  # Top 10
        severity_class = f"threat-{threat['severity'].lower()}"
        
        # Create expandable threat card
        with st.expander(f"[{threat['severity']}] {threat['title']}", expanded=(threat['severity'] in ['CRITICAL', 'HIGH'])):
            # Threat score and metadata
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Location:** {threat['location']}")
                st.markdown(f"**Domains:** {', '.join(threat['domains'])}")
                st.markdown(f"**Confidence:** {threat['confidence']:.0%}")
                st.markdown(f"**Time:** {threat['timestamp'][:19]}")
            
            with col2:
                st.metric("Threat Score", f"{threat['score']:.1f}", help="Higher score = more severe threat")
            
            # Full description with proper formatting
            st.markdown("---")
            st.markdown("### Threat Details")
            st.text(threat['description'])  # Use st.text to preserve line breaks
            
            # Add severity indicator
            if threat['severity'] == 'CRITICAL':
                st.error("CRITICAL - Immediate action required")
            elif threat['severity'] == 'HIGH':
                st.warning("HIGH - Monitor closely and prepare response")
            elif threat['severity'] == 'MEDIUM':
                st.info("MEDIUM - Continue monitoring situation")


def display_timeline_page():
    """Timeline Analysis Page"""
    st.title("Unified Event Timeline")
    st.markdown("### Chronological view of ALL traffic, flood, weather, and intelligence events")
    
    # Initialize Data Manager
    try:
        dm = DataManager()
    except Exception as e:
        st.error(f"Failed to initialize Data Manager: {e}")
        return
    
    # Load unified timeline
    with st.spinner("Loading timeline data from all sources..."):
        timeline = dm.create_unified_timeline()
    
    if timeline.empty:
        st.warning("No timeline data available. Start data collection to populate the timeline.")
        st.info("Run `python continuous_collector.py` to start collecting data")
        
        # Show what data exists
        st.markdown("### Available Data Sources")
        traffic = dm.load_all_traffic_history()
        floods = dm.load_all_flood_warnings()
        st.write(f"Traffic records: {len(traffic)}")
        st.write(f"Flood warnings: {len(floods)}")
        return
    
    # Filters in columns
    st.markdown("### Filters")
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        # Date range filter
        min_date = timeline['timestamp'].min().date()
        max_date = timeline['timestamp'].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="timeline_date_range"
        )
    
    with filter_col2:
        # Event type filter
        event_types = st.multiselect(
            "Event Types",
            options=sorted(timeline['event_type'].unique()),
            default=list(timeline['event_type'].unique()),
            key="timeline_event_types"
        )
    
    with filter_col3:
        # Domain filter
        domains = st.multiselect(
            "Domains",
            options=sorted(timeline['domain'].unique()),
            default=list(timeline['domain'].unique()),
            key="timeline_domains"
        )
    
    with filter_col4:
        # Severity filter
        severity_range = st.slider(
            "Severity",
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
        st.metric("Total Events", len(filtered))
    with col2:
        st.metric("Traffic", len(filtered[filtered['event_type'] == 'traffic']))
    with col3:
        st.metric("Floods", len(filtered[filtered['event_type'] == 'flood']))
    with col4:
        st.metric("Weather", len(filtered[filtered['event_type'] == 'weather']))
    with col5:
        st.metric("Air Quality", len(filtered[filtered['event_type'] == 'air_quality']))
    with col6:
        st.metric("High Severity", len(filtered[filtered['severity'] >= 4]))
    
    st.markdown("---")
    
    # Main timeline scatter plot
    st.markdown("### Event Timeline")
    
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
    
    st.plotly_chart(fig_timeline, width="stretch")
    
    # Two-column charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### Hourly Distribution")
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
        st.plotly_chart(fig_hourly, width="stretch")
    
    with col_right:
        st.markdown("### By Domain")
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
        st.plotly_chart(fig_domain, width="stretch")
    
    # Traffic trends if available
    traffic_data = filtered[filtered['event_type'] == 'traffic']
    if not traffic_data.empty:
        st.markdown("### Traffic Duration Trends")
        
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
            st.plotly_chart(fig_traffic, width="stretch")
    
    # Correlation matrix
    st.markdown("### Event Co-Occurrence")
    
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
            st.plotly_chart(fig_corr, width="stretch")
    except Exception as e:
        st.info("Correlation analysis requires more data points")
    
    # Event table
    st.markdown("---")
    st.markdown("### Event Details")
    
    display_cols = ['timestamp', 'event_type', 'domain', 'severity', 'description', 'location', 'source']
    available_cols = [col for col in display_cols if col in filtered.columns]
    
    display_df = filtered[available_cols].copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    st.dataframe(
        display_df.sort_values('timestamp', ascending=False),
        width="stretch",
        height=400
    )
    
    # Export buttons
    st.markdown("---")
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            csv,
            f"timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
    
    with col_exp2:
        export_data = filtered.copy()
        export_data['timestamp'] = export_data['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        export_data['data'] = export_data['data'].apply(lambda x: str(x) if pd.notna(x) else '')
        
        json_str = json.dumps(export_data.to_dict('records'), indent=2)
        st.download_button(
            "Download JSON",
            json_str,
            f"timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )


def display_domain_status(all_events):
    """Domain-by-domain status grid"""
    st.markdown("### DOMAIN STATUS")
    
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
        'flood': {'icon': '', 'color': '#3498db'},
        'environmental': {'icon': '', 'color': '#9b59b6'},
        'infrastructure': {'icon': '', 'color': '#e67e22'},
        'social': {'icon': '', 'color': '#1abc9c'},
        'traffic': {'icon': '', 'color': '#e74c3c'}
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


def load_raw_data():
    """Load raw data from all sources"""
    raw_data = {
        'flood_warnings': [],
        'traffic': []
    }
    
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


def display_predictive_intelligence(predictions):
    """Predictive threat forecasting with advanced cascade analysis"""
    st.markdown("### PREDICTIVE INTELLIGENCE")
    
    if not predictions:
        st.info("No immediate predictions - monitoring for emerging patterns")
        return
    
    # Separate cascade predictions from others
    cascade_preds = [p for p in predictions if p.get('prediction_type') == 'traffic_cascade']
    other_preds = [p for p in predictions if p.get('prediction_type') != 'traffic_cascade']
    
    # Display cascade predictions prominently
    if cascade_preds:
        st.markdown("#### TRAFFIC CASCADE PREDICTIONS")
        st.markdown("*Advanced network effect analysis: Predicting traffic redistribution from road disruptions*")
        
        for pred in cascade_preds[:10]:
            # Determine severity styling
            risk_score = pred.get('risk_score', 0)
            if risk_score > 0.7:
                severity_color = "#ff0000"
                severity_label = "SEVERE"
            elif risk_score > 0.5:
                severity_color = "#ff8800"
                severity_label = "HIGH"
            else:
                severity_color = "#ffaa00"
                severity_label = "MODERATE"
            
            # Create detailed cascade card
            with st.expander(
                f"[{severity_label}]: {pred.get('affected_route', 'Unknown Route')} - "
                f"Impact in {pred.get('time_to_impact', '15-30 min')}",
                expanded=(risk_score > 0.6)
            ):
                # Header metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Risk Score", f"{risk_score:.0%}", help="Probability of cascade occurring")
                with col2:
                    st.metric("Confidence", f"{pred.get('confidence', 0):.0%}", help="Model confidence in prediction")
                with col3:
                    st.metric("Delay Increase", pred.get('expected_delay_increase', 'N/A'), help="Expected additional delay")
                with col4:
                    st.metric("Capacity Overflow", pred.get('capacity_overflow', 'N/A'), help="Traffic volume increase")
                
                st.markdown("---")
                
                # Trigger and affected route
                st.markdown("**CASCADE TRIGGER:**")
                st.info(f"{pred.get('trigger_event', 'Unknown disruption')}")
                st.markdown(f"*Location: {pred.get('trigger_location', 'Unknown')}*")
                
                st.markdown("**AFFECTED ROUTE:**")
                st.warning(f"{pred.get('affected_route', 'Unknown')} ({pred.get('affected_location', 'Unknown')})")
                
                # Prediction details
                st.markdown("**PREDICTION:**")
                st.text(pred.get('prediction', 'Traffic cascade expected'))
                
                # Recommendation
                st.markdown("**RECOMMENDED ACTION:**")
                st.success(pred.get('recommendation', 'Monitor situation'))
                
                # Technical details
                if pred.get('factors'):
                    with st.expander("Technical Details"):
                        st.markdown("**Analysis Factors:**")
                        for factor in pred['factors']:
                            st.markdown(f"• {factor}")
    
    # Display other prediction types
    if other_preds:
        prediction_types = {}
        for pred in other_preds:
            ptype = pred.get('prediction_type', 'unknown')
            if ptype not in prediction_types:
                prediction_types[ptype] = []
            prediction_types[ptype].append(pred)
        
        st.markdown("---")
        st.markdown("#### OTHER PREDICTIONS")
        
        for ptype, preds in prediction_types.items():
            with st.expander(f"{ptype.replace('_', ' ').title()} ({len(preds)} predictions)", expanded=False):
                for pred in preds[:5]:
                    risk_score = pred.get('risk_score', 0)
                    risk_color = "#ff4444" if risk_score > 0.7 else "#ffaa00" if risk_score > 0.4 else "#4444ff"
                    
                    st.markdown(f"""
                    <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid {risk_color};'>
                        <div style='font-weight: bold; font-size: 16px; margin-bottom: 5px;'>
                            {pred.get('prediction', 'Unknown prediction')}
                        </div>
                        <div style='font-size: 13px; margin: 8px 0;'>
                            Location: {pred.get('location', 'Unknown')} | 
                            Risk: {risk_score:.0%} | 
                            Confidence: {pred.get('confidence', 0):.0%}
                        </div>
                        <div style='font-size: 12px; color: #aaa; margin-top: 8px;'>
                            Recommendation: {pred.get('recommendation', 'Monitor situation')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


def display_intelligence_map(all_events, compound_threats):
    """Geographic threat visualization with all traffic routes and flood zones"""
    st.markdown("### GEO-INTELLIGENCE MAP")
    
    # Statistics
    traffic_events = [e for e in all_events if e.get('domain') == 'transport' and e.get('event_type') != 'road_disruption']
    disruption_events = [e for e in all_events if e.get('event_type') == 'road_disruption']
    flood_events = [e for e in all_events if e.get('domain') == 'environmental' and e.get('threat_type') == 'flood']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Traffic Routes", len(traffic_events))
    with col2:
        st.metric("Road Disruptions", len(disruption_events))
    with col3:
        st.metric("Flood Warnings", len(flood_events))
    with col4:
        st.metric("Compound Threats", len(compound_threats))
    
    # Create map centered on London
    m = folium.Map(
        location=[51.5074, -0.1278],
        zoom_start=10,
        tiles='CartoDB dark_matter'
    )
    
    # Plot events by domain
    domain_colors = {
        'transport': '#ff4444',  # Red for traffic
        'environmental': '#4444ff',  # Blue for floods
        'infrastructure': '#ff8800',
        'social': '#44ff44',
        'health': '#ff44ff'
    }
    
    # Count events by location for clustering
    location_counts = {}
    
    for event in all_events:
        lat = event.get('lat')
        lon = event.get('lon')
        
        if lat and lon:
            domain = event.get('domain', 'unknown')
            color = domain_colors.get(domain, 'gray')
            event_type = event.get('threat_type', event.get('event_type', 'unknown'))
            description = event.get('description', 'Unknown')
            location = event.get('location_name', event.get('location', 'Unknown'))
            temporal_weight = event.get('temporal_weight', 1.0)
            severity = event.get('severity', 3)
            
            # Determine marker size by temporal weight and severity
            radius = 5 + (temporal_weight * 5) + (5 - severity)
            
            # Traffic routes - draw lines from origin to destination
            if domain == 'transport' and event.get('dest_lat') and event.get('dest_lon'):
                dest_lat = event.get('dest_lat')
                dest_lon = event.get('dest_lon')
                
                # Draw route line
                folium.PolyLine(
                    locations=[[lat, lon], [dest_lat, dest_lon]],
                    color=color,
                    weight=2,
                    opacity=0.6 * temporal_weight,
                    popup=f"<b>TRAFFIC: {location}</b><br>{description}"
                ).add_to(m)
                
                # Origin marker
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=radius,
                    popup=f"<b>START: {location}</b><br>{description}<br>Weight: {temporal_weight:.2f}<br>Severity: {severity}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7 * temporal_weight
                ).add_to(m)
                
                # Destination marker
                folium.CircleMarker(
                    location=[dest_lat, dest_lon],
                    radius=radius * 0.7,
                    popup=f"<b>END: {location}</b><br>{description}",
                    color=color,
                    fill=True,
                    fillColor='#ffcc00',
                    fillOpacity=0.5 * temporal_weight
                ).add_to(m)
            
            # Flood warnings - larger zones
            elif event_type == 'flood':
                # Flood area circle
                folium.Circle(
                    location=[lat, lon],
                    radius=1500,  # 1.5km radius flood zone
                    popup=f"<b>FLOOD: {location}</b><br>{description}<br>Severity: {severity}<br>Weight: {temporal_weight:.2f}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.2 * temporal_weight,
                    weight=2
                ).add_to(m)
                
                # Center marker
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    popup=f"<b>FLOOD: {location}</b><br>{description}",
                    color='#0044ff',
                    fill=True,
                    fillColor='#0088ff',
                    fillOpacity=0.8
                ).add_to(m)
            
            # Road disruptions - distinct markers
            elif event_type == 'road_disruption':
                category = event.get('category', 'Unknown')
                has_closures = event.get('has_closures', False)
                
                # Color by category
                disruption_color = {
                    'Collisions': '#ff0000',  # Red for accidents
                    'Works': '#ff8800',  # Orange for roadworks
                    'Restrictions': '#ffaa00',  # Yellow-orange
                    'Planned': '#ffcc00'  # Yellow
                }.get(category, '#ff6600')
                
                # Icon style
                icon_style = 'exclamation-sign' if has_closures else 'warning-sign'
                
                # Warning marker for disruptions
                folium.Marker(
                    location=[lat, lon],
                    popup=f"<b>DISRUPTION: {category.upper()}</b><br>{location}<br>{description}<br>Severity: {severity}/5<br>Closures: {'Yes' if has_closures else 'No'}<br>Weight: {temporal_weight:.2f}",
                    icon=folium.Icon(
                        color='red' if has_closures else 'orange',
                        icon=icon_style,
                        prefix='glyphicon'
                    )
                ).add_to(m)
                
                # Impact radius circle
                impact_radius = 500 if has_closures else 300
                folium.Circle(
                    location=[lat, lon],
                    radius=impact_radius,
                    popup=f"<b>{category} Impact Zone</b>",
                    color=disruption_color,
                    fill=True,
                    fillColor=disruption_color,
                    fillOpacity=0.15 * temporal_weight,
                    weight=1
                ).add_to(m)
            
            # Other events
            else:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=radius,
                    popup=f"<b>{event_type.upper()}</b><br>{location}<br>{description}<br>Weight: {temporal_weight:.2f}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6 * temporal_weight
                ).add_to(m)
    
    # Plot compound threats as large warning markers
    for threat in compound_threats:
        lat = threat.get('center_lat')
        lon = threat.get('center_lon')
        
        if lat and lon:
            severity_color = {
                'CRITICAL': '#8b0000',
                'HIGH': '#ff0000',
                'MEDIUM': '#ff8800',
                'LOW': '#ffff00'
            }.get(threat.get('severity', 'MEDIUM'), 'gray')
            
            # Large threat zone
            folium.Circle(
                location=[lat, lon],
                radius=2000,
                popup=f"<b>COMPOUND THREAT</b><br><b>{threat.get('threat_category', 'Unknown')}</b><br>{threat.get('description', '')}<br>Severity: {threat.get('severity', 'UNKNOWN')}",
                color=severity_color,
                fill=True,
                fillColor=severity_color,
                fillOpacity=0.3,
                weight=3
            ).add_to(m)
            
            # Center warning icon
            folium.Marker(
                location=[lat, lon],
                popup=f"<b>THREAT: {threat.get('threat_category', 'THREAT')}</b><br>{threat.get('description', '')}",
                icon=folium.Icon(color='red', icon='warning-sign', prefix='glyphicon')
            ).add_to(m)
    
    folium_static(m, width=1400, height=600)


def main():
    """Main Palantir dashboard with timeline integration"""
    
    # Sidebar Navigation
    st.sidebar.title("🦉 OWL NAVIGATION")
    page = st.sidebar.radio(
        "Select View",
        ["Threat Intelligence", "Timeline Analysis", "Predictions", "Geo-Intel", "Correlations", "Analytics", "Raw Data"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### CONTROLS")
    
    if st.sidebar.button("REFRESH DATA", ):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Last Update:**  \n{datetime.now().strftime('%H:%M:%S')}")
    
    # Header
    display_palantir_header()
    
    # ==================== TIMELINE PAGE ====================
    if page == "Timeline Analysis":
        display_timeline_page()
        return
    
    # ==================== RUN INTELLIGENCE PIPELINE (for all pages) ====================
    # Run intelligence pipeline
    with st.spinner("Running multi-domain intelligence analysis..."):
        intel = run_full_intelligence_pipeline()
    
    if not intel or intel is None:
        st.error("Failed to load intelligence data")
        return
    
    # Data availability check
    if not intel.get('all_events'):
        st.warning("No events loaded. Intelligence analysis requires data.")
        st.info("**To populate data:**")
        st.code("python owl_engine\\continuous_collector.py")
        st.markdown("This will collect traffic, flood, and environmental data every 10 minutes.")
        
        # Show what data exists
        try:
            dm = DataManager()
            timeline = dm.create_unified_timeline()
            st.markdown(f"**Current data available:** {len(timeline)} events")
            if not timeline.empty:
                st.dataframe(timeline[['timestamp', 'event_type', 'description']].head(10))
        except:
            pass
        return
    
    # ==================== PREDICTIONS PAGE ====================
    if page == "Predictions":
        display_predictive_intelligence(intel['predictions'])
        return
    
    # ==================== GEO-INTEL PAGE ====================
    if page == "Geo-Intel":
        display_intelligence_map(intel['all_events'], intel['compound_threats'])
        
        # Add street monitoring view
        st.markdown("---")
        st.markdown("### STREET MONITORING - LIVE CAMERA FEEDS")
        
        try:
            from abbey_road_dashboard import display_abbey_road_analytics
            dm_abbey = DataManager()
            display_abbey_road_analytics(dm_abbey)
        except ImportError as ie:
            st.error(f"Street monitoring module not found: {ie}")
            st.info("Make sure you're running from the owl_engine directory")
        except Exception as e:
            st.error(f"Error loading street monitoring: {e}")
        
        return
    
    # ==================== CORRELATIONS PAGE ====================
    if page == "Correlations":
        st.markdown("### Multi-Domain Correlation Engine")
        st.markdown("*Advanced geo-temporal correlation analysis across flood, traffic, environmental, and infrastructure domains*")
        
        # Metrics overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Flood-Traffic Correlations", len(intel['correlations']))
        with col2:
            st.metric("Compound Threats", len(intel['compound_threats']))
        with col3:
            total_events = len(intel['all_events'])
            st.metric("Total Events Analyzed", total_events)
        with col4:
            domains = set(e['domain'] for e in intel['all_events']) if intel['all_events'] else set()
            st.metric("Active Domains", len(domains))
        
        st.markdown("---")
        
        # Tabs for different correlation views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Correlation Engine", "Flood-Traffic Analysis", "Compound Threats", "Network Graph", "Event Causality & Scenarios"
        ])
        
        # ========== TAB 1: Correlation Engine Architecture ==========
        with tab1:
            st.markdown("#### Correlation Engine Architecture")
            
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.markdown("""
                **Layer 4: Geo-Temporal Correlation Engine**
                
                **Spatial Correlation (Jaccard Similarity):**
                ```
                Score = |locations_A ∩ locations_B| / |locations_A ∪ locations_B|
                ```
                Measures geographic overlap between events (0.0 to 1.0)
                
                **Temporal Correlation (Exponential Decay):**
                ```
                Score = exp(-λ × |Δt|)
                where Δt = time difference in hours
                      λ = decay rate (0.1 default)
                ```
                Events closer in time = higher correlation
                
                **Combined Score:**
                ```
                C = Spatial × Temporal × (Severity_A + Severity_B) / 2
                ```
                """)
            
            with col_right:
                st.markdown("""
                **Multi-Domain Threat Correlator**
                
                **Temporal Weighting:**
                - Recent events: Higher weight in analysis
                - Historical data: Preserved for pattern recognition
                - Decay half-life: 24 hours
                
                **DBSCAN Spatial Clustering:**
                - Groups events within 2km radius
                - Minimum 2 events per cluster
                - Identifies geographic hotspots
                
                **Compound Threat Detection:**
                - Requires ≥2 domains involved
                - Time window: 6 hours default
                - Calculates aggregate threat score
                """)
            
            st.markdown("---")
            
            # Processing statistics
            if intel['all_events']:
                st.markdown("#### Processing Statistics")
                
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                
                with stats_col1:
                    st.markdown("**Event Distribution by Domain**")
                    domain_counts = {}
                    for event in intel['all_events']:
                        domain = event.get('domain', 'unknown')
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1
                    
                    domain_df = pd.DataFrame([
                        {'Domain': k, 'Count': v} for k, v in domain_counts.items()
                    ])
                    fig = px.pie(domain_df, values='Count', names='Domain', 
                                title='Events by Domain')
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with stats_col2:
                    st.markdown("**Temporal Distribution (24h)**")
                    
                    # Parse timestamps and group by hour
                    hourly_counts = {}
                    for event in intel['all_events']:
                        try:
                            ts_str = event.get('timestamp', '')
                            if ts_str:
                                if isinstance(ts_str, str):
                                    ts = pd.to_datetime(ts_str)
                                else:
                                    ts = ts_str
                                hour = ts.hour
                                hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
                        except:
                            pass
                    
                    if hourly_counts:
                        hour_df = pd.DataFrame([
                            {'Hour': k, 'Events': v} for k, v in sorted(hourly_counts.items())
                        ])
                        fig = px.bar(hour_df, x='Hour', y='Events', 
                                    title='Events by Hour')
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                with stats_col3:
                    st.markdown("**Correlation Confidence**")
                    
                    if intel['correlations']:
                        confidence_data = {
                            'High (>0.7)': sum(1 for c in intel['correlations'] if c.get('combined_score', 0) > 0.7),
                            'Medium (0.4-0.7)': sum(1 for c in intel['correlations'] if 0.4 <= c.get('combined_score', 0) <= 0.7),
                            'Low (<0.4)': sum(1 for c in intel['correlations'] if c.get('combined_score', 0) < 0.4)
                        }
                        conf_df = pd.DataFrame([
                            {'Level': k, 'Count': v} for k, v in confidence_data.items()
                        ])
                        fig = px.bar(conf_df, x='Level', y='Count',
                                    title='Correlation Confidence Levels',
                                    color='Level',
                                    color_discrete_map={
                                        'High (>0.7)': '#2ecc71',
                                        'Medium (0.4-0.7)': '#f39c12',
                                        'Low (<0.4)': '#e74c3c'
                                    })
                        fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No correlations to analyze")
        
        # ========== TAB 2: Flood-Traffic Correlations ==========
        with tab2:
            st.markdown("#### Flood-Traffic Correlation Analysis")
            
            if intel['correlations']:
                # Top correlations
                st.markdown("**Top Correlations by Combined Score**")
                
                df = pd.DataFrame(intel['correlations'])
                df_sorted = df.sort_values('combined_score', ascending=False).head(20)
                
                # Visualization: Scatter plot
                fig = px.scatter(
                    df_sorted,
                    x='temporal_correlation',
                    y='spatial_correlation',
                    size='combined_score',
                    color='combined_score',
                    hover_data=['flood_description', 'traffic_route', 'lag_hours'],
                    title='Spatial vs Temporal Correlation',
                    labels={
                        'spatial_correlation': 'Spatial Overlap (Jaccard)',
                        'temporal_correlation': 'Temporal Proximity (Exp Decay)',
                        'combined_score': 'Combined Score'
                    },
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Time lag analysis
                st.markdown("**Time Lag Analysis**")
                fig2 = px.histogram(
                    df_sorted,
                    x='lag_hours',
                    title='Distribution of Time Lags (hours)',
                    nbins=20,
                    labels={'lag_hours': 'Time Lag (hours)', 'count': 'Frequency'}
                )
                fig2.add_vline(x=0, line_dash="dash", line_color="red", 
                              annotation_text="Simultaneous")
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown("---")
                
                # Detailed table
                st.markdown("**Detailed Correlation Data**")
                display_df = df_sorted[[
                    'flood_description', 'flood_area', 'traffic_route', 
                    'combined_score', 'spatial_correlation', 'temporal_correlation',
                    'lag_hours', 'inference'
                ]].copy()
                display_df.columns = [
                    'Flood Event', 'Area', 'Traffic Route', 
                    'Score', 'Spatial', 'Temporal', 'Lag (h)', 'Inference'
                ]
                
                # Color code by score
                def highlight_score(val):
                    if isinstance(val, (int, float)):
                        if val > 0.7:
                            return 'background-color: #27ae60; color: white'
                        elif val > 0.4:
                            return 'background-color: #f39c12; color: white'
                        else:
                            return 'background-color: #e74c3c; color: white'
                    return ''
                
                styled_df = display_df.style.applymap(highlight_score, subset=['Score'])
                st.dataframe(styled_df, use_container_width=True, height=400)
                
            else:
                st.info("""
                **No Flood-Traffic Correlations Detected**
                
                Possible reasons:
                - No active flood warnings in the system
                - No spatial overlap between floods and traffic delays
                - Events too far apart in time
                
                The correlation engine runs continuously and will detect patterns when conditions match.
                """)
        
        # ========== TAB 3: Compound Threats ==========
        with tab3:
            st.markdown("#### Multi-Domain Compound Threats")
            
            if intel['compound_threats']:
                for i, threat in enumerate(intel['compound_threats'][:10], 1):
                    severity = threat.get('severity', 'MEDIUM')
                    threat_score = threat.get('threat_score', 0)
                    domains = threat.get('domains_involved', [])
                    
                    # Color code by severity
                    if severity == 'CRITICAL':
                        bg_color = '#8b0000'
                    elif severity == 'HIGH':
                        bg_color = '#cc6600'
                    else:
                        bg_color = '#cc9900'
                    
                    st.markdown(f"""
                    <div style='background: {bg_color}; padding: 15px; border-radius: 8px; margin: 10px 0; color: white;'>
                        <h4 style='margin: 0; color: white;'>{i}. {threat.get('category', 'Unknown Threat')} - {severity}</h4>
                        <p style='margin: 5px 0;'><strong>Score:</strong> {threat_score:.2f} | <strong>Domains:</strong> {', '.join(domains)}</p>
                        <p style='margin: 5px 0;'><strong>Location:</strong> {threat.get('location', 'Multiple areas')}</p>
                        <p style='margin: 5px 0; font-style: italic;'>{threat.get('description', 'No description')}</p>
                        <p style='margin: 5px 0; font-size: 12px;'><strong>Events:</strong> {threat.get('event_count', 0)} correlated events</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("""
                **No Compound Threats Detected**
                
                The system is monitoring for multi-domain threat patterns.
                Compound threats require:
                - Multiple domains affected (≥2)
                - Geographic clustering (within 2km)
                - Temporal proximity (within 6 hours)
                """)
        
        # ========== TAB 4: Network Graph ==========
        with tab4:
            st.markdown("#### Correlation Network Graph")
            
            if intel['correlations'] and len(intel['correlations']) > 0:
                st.info("""
                **Network Visualization**
                
                This shows the relationship network between correlated events:
                - **Nodes**: Individual events (floods, traffic delays)
                - **Edges**: Correlations (thickness = strength)
                - **Colors**: Event types
                """)
                
                # Build network data for visualization
                import networkx as nx
                
                G = nx.Graph()
                
                # Add nodes and edges from correlations
                for corr in intel['correlations'][:30]:  # Limit to top 30
                    flood_id = corr.get('flood_id', 'flood')
                    traffic_id = corr.get('traffic_id', 'traffic')
                    score = corr.get('combined_score', 0)
                    
                    G.add_node(flood_id, type='flood', 
                              label=corr.get('flood_area', 'Flood')[:20])
                    G.add_node(traffic_id, type='traffic',
                              label=corr.get('traffic_route', 'Traffic')[:20])
                    G.add_edge(flood_id, traffic_id, weight=score)
                
                if len(G.nodes()) > 0:
                    # Create visualization
                    pos = nx.spring_layout(G, k=0.5, iterations=50)
                    
                    # Extract edge data
                    edge_x = []
                    edge_y = []
                    edge_weights = []
                    
                    for edge in G.edges(data=True):
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                        edge_weights.append(edge[2].get('weight', 0))
                    
                    # Extract node data
                    node_x = []
                    node_y = []
                    node_colors = []
                    node_labels = []
                    
                    for node in G.nodes(data=True):
                        x, y = pos[node[0]]
                        node_x.append(x)
                        node_y.append(y)
                        node_type = node[1].get('type', 'unknown')
                        node_colors.append('#3498db' if node_type == 'flood' else '#e74c3c')
                        node_labels.append(node[1].get('label', 'Unknown'))
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Add edges
                    fig.add_trace(go.Scatter(
                        x=edge_x, y=edge_y,
                        mode='lines',
                        line=dict(width=1, color='#888'),
                        hoverinfo='none',
                        showlegend=False
                    ))
                    
                    # Add nodes
                    fig.add_trace(go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        marker=dict(size=15, color=node_colors, line=dict(width=2, color='white')),
                        text=node_labels,
                        textposition="top center",
                        hoverinfo='text',
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        title='Event Correlation Network',
                        showlegend=False,
                        hovermode='closest',
                        height=600,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown(f"""
                    **Network Statistics:**
                    - Nodes: {len(G.nodes())}
                    - Correlations: {len(G.edges())}
                    - Average connections per event: {2*len(G.edges())/len(G.nodes()):.1f}
                    """)
                else:
                    st.warning("Not enough data to build network graph")
            else:
                st.info("No correlations available for network visualization")
        
        # ========== TAB 5: Event Causality & Scenarios ==========
        with tab5:
            st.markdown("#### Event Causality Analysis & Scenario Builder")
            st.markdown("*Understanding what events cause what - building causal chains and threat scenarios*")
            
            if intel['all_events'] or intel['correlations']:
                
                # Sub-tabs for different causality views
                causality_tab1, causality_tab2, causality_tab3, causality_tab4 = st.tabs([
                    "Causal Flow (Sankey)", "Timeline Cascade", "Scenario Stories", "Causality Matrix"
                ])
                
                # ========== CAUSALITY TAB 1: Sankey Diagram ==========
                with causality_tab1:
                    st.markdown("**Event Causality Flow Diagram**")
                    st.markdown("*Shows how events trigger other events - thickness indicates correlation strength*")
                    
                    if intel['correlations']:
                        # Build Sankey data from correlations
                        # Source = Cause, Target = Effect (based on time lag)
                        sankey_sources = []
                        sankey_targets = []
                        sankey_values = []
                        sankey_colors = []
                        
                        # Create node labels
                        all_nodes = []
                        node_labels = []
                        
                        for i, corr in enumerate(intel['correlations'][:15]):  # Top 15 correlations
                            lag_hours = corr.get('lag_hours', 0)
                            score = corr.get('combined_score', 0)
                            
                            # Determine causality based on time lag
                            if lag_hours > 0:
                                # Flood happened first -> caused traffic
                                cause = f"Flood: {corr.get('flood_area', 'Unknown')[:25]}"
                                effect = f"Traffic: {corr.get('traffic_route', 'Unknown')[:25]}"
                                inference = corr.get('inference', '')
                                
                                # Add to nodes if not exists
                                if cause not in all_nodes:
                                    all_nodes.append(cause)
                                    node_labels.append(cause)
                                if effect not in all_nodes:
                                    all_nodes.append(effect)
                                    node_labels.append(effect)
                                
                                cause_idx = all_nodes.index(cause)
                                effect_idx = all_nodes.index(effect)
                                
                                sankey_sources.append(cause_idx)
                                sankey_targets.append(effect_idx)
                                sankey_values.append(score * 100)  # Scale for visibility
                                
                                # Color based on inference strength
                                if 'HIGH' in inference:
                                    sankey_colors.append('rgba(231, 76, 60, 0.6)')  # Red for high causality
                                elif 'MODERATE' in inference:
                                    sankey_colors.append('rgba(243, 156, 18, 0.6)')  # Orange
                                else:
                                    sankey_colors.append('rgba(52, 152, 219, 0.4)')  # Blue for low
                        
                        if sankey_sources:
                            # Create Sankey diagram
                            fig = go.Figure(data=[go.Sankey(
                                node=dict(
                                    pad=15,
                                    thickness=20,
                                    line=dict(color='white', width=0.5),
                                    label=node_labels,
                                    color=['#3498db' if 'Flood' in label else '#e74c3c' for label in node_labels]
                                ),
                                link=dict(
                                    source=sankey_sources,
                                    target=sankey_targets,
                                    value=sankey_values,
                                    color=sankey_colors
                                )
                            )])
                            
                            fig.update_layout(
                                title="Causal Event Flow: What Causes What",
                                font=dict(size=10),
                                height=600
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("""
                            **How to Read:**
                            - **Left side**: Causative events (floods)
                            - **Right side**: Resulting events (traffic delays)
                            - **Flow width**: Strength of causal relationship
                            - **Color**: Red = HIGH causality, Orange = MODERATE, Blue = LOW
                            """)
                        else:
                            st.info("No clear causal relationships detected (all events are concurrent)")
                    else:
                        st.info("No correlations available for causal flow analysis")
                
                # ========== CAUSALITY TAB 2: Timeline Cascade ==========
                with causality_tab2:
                    st.markdown("**Timeline Cascade Visualization**")
                    st.markdown("*Chronological view showing when events happen and what they trigger*")
                    
                    if intel['correlations']:
                        # Build timeline cascade
                        cascade_events = []
                        
                        for corr in intel['correlations'][:20]:
                            lag_hours = corr.get('lag_hours', 0)
                            
                            if lag_hours != 0:  # Only show events with time separation
                                # Get timestamps (approximate)
                                flood_time = 0  # Base time
                                traffic_time = lag_hours
                                
                                cascade_events.append({
                                    'event': f"Flood in {corr.get('flood_area', 'Unknown')}",
                                    'time': flood_time,
                                    'type': 'Flood',
                                    'severity': corr.get('flood_severity', 0.5)
                                })
                                
                                cascade_events.append({
                                    'event': f"Traffic delay on {corr.get('traffic_route', 'Unknown')}",
                                    'time': traffic_time,
                                    'type': 'Traffic',
                                    'severity': corr.get('traffic_severity', 0.5),
                                    'caused_by': f"Flood in {corr.get('flood_area', 'Unknown')}"
                                })
                        
                        if cascade_events:
                            # Sort by time
                            cascade_events.sort(key=lambda x: x['time'])
                            
                            # Create cascade visualization
                            cascade_df = pd.DataFrame(cascade_events)
                            
                            fig = go.Figure()
                            
                            # Add events as scatter points
                            for event_type in ['Flood', 'Traffic']:
                                type_events = cascade_df[cascade_df['type'] == event_type]
                                
                                fig.add_trace(go.Scatter(
                                    x=type_events['time'],
                                    y=type_events['event'],
                                    mode='markers+text',
                                    name=event_type,
                                    marker=dict(
                                        size=15,
                                        color='#3498db' if event_type == 'Flood' else '#e74c3c',
                                        line=dict(width=2, color='white')
                                    ),
                                    text=[f"{t}h" for t in type_events['time']],
                                    textposition="top center"
                                ))
                            
                            # Add causality arrows
                            for _, event in cascade_df.iterrows():
                                if 'caused_by' in event:
                                    cause_event = cascade_df[cascade_df['event'] == event['caused_by']]
                                    if not cause_event.empty:
                                        fig.add_annotation(
                                            x=event['time'],
                                            y=event['event'],
                                            ax=cause_event.iloc[0]['time'],
                                            ay=cause_event.iloc[0]['event'],
                                            xref='x',
                                            yref='y',
                                            axref='x',
                                            ayref='y',
                                            showarrow=True,
                                            arrowhead=2,
                                            arrowsize=1,
                                            arrowwidth=2,
                                            arrowcolor='#95a5a6'
                                        )
                            
                            fig.update_layout(
                                title="Event Cascade Timeline (Time-Ordered Causality)",
                                xaxis_title="Time Offset (hours)",
                                yaxis_title="Events",
                                height=600,
                                showlegend=True,
                                hovermode='closest'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("""
                            **Reading the Cascade:**
                            - **Blue dots**: Flood events (causes)
                            - **Red dots**: Traffic delays (effects)
                            - **Gray arrows**: Causal relationships
                            - **X-axis**: Time progression (hours after initial event)
                            """)
                        else:
                            st.info("No time-separated events found for cascade visualization")
                    else:
                        st.info("No correlations available")
                
                # ========== CAUSALITY TAB 3: Scenario Stories ==========
                with causality_tab3:
                    st.markdown("**Threat Scenarios & Event Stories**")
                    st.markdown("*Narrative descriptions of how events unfolded and impacted each other*")
                    
                    # Build scenarios from correlations and compound threats
                    scenarios = []
                    
                    # Scenario 1: High-confidence causal chains
                    if intel['correlations']:
                        high_conf_corrs = [c for c in intel['correlations'] if c.get('combined_score', 0) > 0.6]
                        
                        if high_conf_corrs:
                            scenario_events = []
                            for corr in high_conf_corrs[:5]:
                                lag = corr.get('lag_hours', 0)
                                if lag > 0:
                                    scenario_events.append({
                                        'time': f"{abs(lag)} hours delay",
                                        'cause': f"Flood in {corr.get('flood_area', 'area')}",
                                        'effect': f"Traffic delays on {corr.get('traffic_route', 'route')}",
                                        'confidence': corr.get('combined_score', 0),
                                        'description': corr.get('inference', '')
                                    })
                            
                            if scenario_events:
                                scenarios.append({
                                    'title': 'Flood-Induced Traffic Disruption Chain',
                                    'category': 'Environmental → Infrastructure',
                                    'severity': 'HIGH',
                                    'events': scenario_events
                                })
                    
                    # Scenario 2: Compound threats as scenarios
                    if intel['compound_threats']:
                        for threat in intel['compound_threats'][:3]:
                            scenarios.append({
                                'title': threat.get('category', 'Unknown Threat'),
                                'category': f"Multi-Domain ({', '.join(threat.get('domains_involved', []))})",
                                'severity': threat.get('severity', 'MEDIUM'),
                                'description': threat.get('description', ''),
                                'location': threat.get('location', 'Unknown'),
                                'event_count': threat.get('event_count', 0)
                            })
                    
                    # Display scenarios
                    if scenarios:
                        for i, scenario in enumerate(scenarios, 1):
                            severity = scenario.get('severity', 'MEDIUM')
                            
                            if severity == 'CRITICAL' or severity == 'HIGH':
                                bg_color = '#8b0000'
                                icon = '🔴'
                            elif severity == 'MEDIUM':
                                bg_color = '#cc6600'
                                icon = '🟠'
                            else:
                                bg_color = '#2c3e50'
                                icon = '🔵'
                            
                            st.markdown(f"""
                            <div style='background: {bg_color}; padding: 20px; border-radius: 10px; margin: 15px 0; color: white; border-left: 5px solid white;'>
                                <h3 style='margin: 0; color: white;'>{icon} Scenario {i}: {scenario['title']}</h3>
                                <p style='margin: 5px 0; opacity: 0.9;'><strong>Category:</strong> {scenario['category']}</p>
                                <p style='margin: 10px 0; font-size: 16px; line-height: 1.6;'>{scenario.get('description', '')}</p>
                            """, unsafe_allow_html=True)
                            
                            # If scenario has event chain, display it
                            if 'events' in scenario:
                                st.markdown("<p style='margin: 10px 0; color: white;'><strong>Event Chain:</strong></p>", unsafe_allow_html=True)
                                for j, event in enumerate(scenario['events'], 1):
                                    st.markdown(f"""
                                    <div style='margin-left: 20px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 5px; margin-bottom: 5px;'>
                                        <p style='margin: 0; color: white;'><strong>Step {j}:</strong> {event['cause']}</p>
                                        <p style='margin: 5px 0 0 0; color: white;'>→ <em>{event['time']}</em> → {event['effect']}</p>
                                        <p style='margin: 5px 0 0 0; font-size: 12px; color: #ecf0f1;'>Confidence: {event['confidence']:.2f} | {event['description']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.info("No significant scenarios detected yet. Scenarios will appear when multiple correlated events are identified.")
                
                # ========== CAUSALITY TAB 4: Causality Matrix ==========
                with causality_tab4:
                    st.markdown("**Causality Matrix**")
                    st.markdown("*Shows which event types cause which other event types*")
                    
                    if intel['all_events'] and intel['correlations']:
                        # Build causality matrix
                        # Rows = Causes, Columns = Effects
                        
                        event_types = list(set([e.get('domain', 'unknown') for e in intel['all_events']]))
                        
                        # Initialize matrix
                        causality_matrix = pd.DataFrame(0, index=event_types, columns=event_types)
                        
                        # PopulateMatrix based on correlations with time lag
                        for corr in intel['correlations']:
                            lag = corr.get('lag_hours', 0)
                            
                            if lag > 0:  # Flood caused traffic
                                causality_matrix.loc['flood', 'traffic'] += corr.get('combined_score', 0)
                            elif lag < 0:  # Traffic happened before flood (rare but possible)
                                causality_matrix.loc['traffic', 'flood'] += corr.get('combined_score', 0)
                        
                        # Normalize by sum for percentages
                        total_causality = causality_matrix.sum().sum()
                        if total_causality > 0:
                            causality_matrix_pct = (causality_matrix / total_causality * 100).round(1)
                            
                            # Create heatmap
                            fig = px.imshow(
                                causality_matrix_pct,
                                labels=dict(x="Effect (Result)", y="Cause (Trigger)", color="Causality %"),
                                x=causality_matrix_pct.columns,
                                y=causality_matrix_pct.index,
                                color_continuous_scale='Reds',
                                title="Causality Matrix: What Causes What"
                            )
                            
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("""
                            **How to Read:**
                            - **Rows (Y-axis)**: Causative event types (what triggers)
                            - **Columns (X-axis)**: Effect event types (what results)
                            - **Color intensity**: Strength of causal relationship
                            - **Numbers**: Percentage of total causality
                            
                            **Example:** If "flood" row shows high value in "traffic" column, 
                            it means floods frequently cause traffic delays.
                            """)
                            
                            # Summary statistics
                            st.markdown("**Key Causal Relationships:**")
                            
                            # Find strongest relationships
                            causal_pairs = []
                            for cause in causality_matrix.index:
                                for effect in causality_matrix.columns:
                                    if cause != effect and causality_matrix.loc[cause, effect] > 0:
                                        causal_pairs.append({
                                            'Cause': cause.capitalize(),
                                            'Effect': effect.capitalize(),
                                            'Strength': causality_matrix.loc[cause, effect],
                                            'Percentage': causality_matrix_pct.loc[cause, effect]
                                        })
                            
                            if causal_pairs:
                                causal_df = pd.DataFrame(causal_pairs).sort_values('Strength', ascending=False)
                                st.dataframe(causal_df, use_container_width=True)
                            else:
                                st.info("No causal relationships detected yet")
                        else:
                            st.info("No quantified causal relationships available")
                    else:
                        st.info("Need both events and correlations to build causality matrix")
            
            else:
                st.info("""
                **Event Causality Analysis**
                
                This section will show:
                - Causal flow diagrams (what causes what)
                - Timeline cascades (chronological event triggering)
                - Threat scenarios (narrative event stories)
                - Causality matrix (systematic cause-effect relationships)
                
                Start collecting data to see causality analysis!
                """)
        
        return
    
    # ==================== ANALYTICS PAGE ====================
    if page == "Analytics":
        st.markdown("### System Analytics")
        
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
            st.plotly_chart(fig, width="stretch")
        return
    
    # ==================== RAW DATA PAGE ====================
    if page == "Raw Data":
        st.markdown("### Raw Collected Data")
        st.info("Showing all raw data collected from API sources")
        
        # Load raw data
        raw_data = load_raw_data()
        
        # Flood Warnings
        st.markdown("### Flood Warnings")
        if raw_data['flood_warnings']:
            df_flood = pd.DataFrame(raw_data['flood_warnings'])
            st.dataframe(df_flood, width="stretch", height=300)
            st.markdown(f"**Total Records:** {len(raw_data['flood_warnings'])}")
        else:
            st.warning("No flood warning data available")
        
        st.markdown("---")
        
        # Traffic Data
        st.markdown("### Traffic Conditions")
        if raw_data['traffic']:
            df_traffic = pd.DataFrame(raw_data['traffic'])
            st.dataframe(df_traffic, width="stretch", height=300)
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
                    st.plotly_chart(fig_traffic, width="stretch")
            except Exception as e:
                st.info(f"Chart unavailable: {str(e)}")
        else:
            st.warning("No traffic data available")
        
        st.markdown("---")
        
        # All Events (Processed)
        st.markdown("### Processed Intelligence Events")
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
            st.dataframe(df_events, width="stretch", height=400)
            st.markdown(f"**Total Events:** {len(intel['all_events'])} (showing first 100)")
        else:
            st.warning("No processed events available")
        
        # Download buttons
        st.markdown("---")
        st.markdown("### Export Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if raw_data['flood_warnings']:
                csv_flood = pd.DataFrame(raw_data['flood_warnings']).to_csv(index=False)
                st.download_button(
                    "Export Flood Data (CSV)",
                    csv_flood,
                    "flood_warnings.csv",
                    "text/csv"
                )
        
        with col2:
            if raw_data['traffic']:
                csv_traffic = pd.DataFrame(raw_data['traffic']).to_csv(index=False)
                st.download_button(
                    "Export Traffic Data (CSV)",
                    csv_traffic,
                    "traffic_data.csv",
                    "text/csv"
                )
        
        with col3:
            if intel['all_events']:
                csv_events = pd.DataFrame(events_list).to_csv(index=False)
                st.download_button(
                    "Export Events (CSV)",
                    csv_events,
                    "intelligence_events.csv",
                    "text/csv"
                )
        return
    
    # ==================== THREAT INTELLIGENCE PAGE (DEFAULT) ====================
    # Main threat board
    display_threat_board(intel['compound_threats'], intel['alerts'])
    
    st.markdown("---")
    
    # Domain status grid
    display_domain_status(intel['all_events'])


if __name__ == "__main__":
    main()
