"""
Timeline Dashboard - Unified view of ALL data sources over time
Traffic | Floods | Weather | Air Quality | Health
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from data_manager import DataManager

st.set_page_config(
    page_title="🦉 OWL Timeline",
    layout="wide",
    page_icon="⏱️",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    .metric-card {
        background: rgba(26, 31, 58, 0.8);
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #2a3f5f;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Data Manager
dm = DataManager()

# Header
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 style='font-size: 48px; color: #00d4ff;'>⏱️ OWL TIMELINE</h1>
    <p style='color: #888; letter-spacing: 2px;'>
        UNIFIED MULTI-SOURCE INTELLIGENCE TIMELINE
    </p>
</div>
""", unsafe_allow_html=True)

# Load data
with st.spinner("📊 Loading timeline data from all sources..."):
    timeline = dm.create_unified_timeline()
    stats = dm.get_summary_stats()

if timeline.empty:
    st.warning("⚠️ No timeline data available. Start data collection first!")
    st.info("💡 Run: `python continuous_collector.py` to start collecting data")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.markdown("### 🔍 TIMELINE FILTERS")
    
    # Date range
    min_date = timeline['timestamp'].min().date()
    max_date = timeline['timestamp'].max().date()
    
    date_range = st.date_input(
        "📅 Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Event type filter
    event_types = st.multiselect(
        "📊 Event Types",
        options=sorted(timeline['event_type'].unique()),
        default=sorted(timeline['event_type'].unique())
    )
    
    # Domain filter
    domains = st.multiselect(
        "🏗️ Domains",
        options=sorted(timeline['domain'].unique()),
        default=sorted(timeline['domain'].unique())
    )
    
    # Severity filter
    severity_range = st.slider(
        "⚠️ Severity Level",
        min_value=1,
        max_value=5,
        value=(1, 5)
    )
    
    # Source filter
    sources = st.multiselect(
        "📡 Data Sources",
        options=sorted(timeline['source'].unique()),
        default=sorted(timeline['source'].unique())
    )
    
    st.markdown("---")
    
    # Export options
    st.markdown("### 💾 EXPORT")
    if st.button("📥 Download Timeline JSON", use_container_width=True):
        export_path = dm.export_timeline()
        if export_path:
            with open(export_path, 'r') as f:
                st.download_button(
                    label="⬇️ Save JSON File",
                    data=f.read(),
                    file_name="owl_timeline.json",
                    mime="application/json",
                    use_container_width=True
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
    (filtered['severity'] <= severity_range[1]) &
    (filtered['source'].isin(sources))
]

# Metrics row
st.markdown("### 📊 OVERVIEW METRICS")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Total Events", f"{len(filtered):,}")

with col2:
    traffic_count = len(filtered[filtered['event_type'] == 'traffic'])
    st.metric("🚗 Traffic", traffic_count)

with col3:
    flood_count = len(filtered[filtered['event_type'] == 'flood'])
    st.metric("🌊 Floods", flood_count)

with col4:
    weather_count = len(filtered[filtered['event_type'] == 'weather'])
    st.metric("🌤️ Weather", weather_count)

with col5:
    air_count = len(filtered[filtered['event_type'] == 'air_quality'])
    st.metric("🌫️ Air Quality", air_count)

with col6:
    health_count = len(filtered[filtered['event_type'] == 'health'])
    st.metric("🏥 Health", health_count)

st.markdown("---")

# Main timeline visualization
st.markdown("### 📈 CHRONOLOGICAL EVENT TIMELINE")

# Create timeline scatter plot
fig_timeline = px.scatter(
    filtered,
    x='timestamp',
    y='severity',
    color='event_type',
    size='severity',
    hover_data={
        'timestamp': '|%Y-%m-%d %H:%M',
        'description': True,
        'location': True,
        'source': True,
        'severity': True,
        'event_type': False
    },
    title='All Events Over Time (Size = Severity)',
    height=500,
    color_discrete_map={
        'traffic': '#FF6B6B',
        'flood': '#4ECDC4',
        'weather': '#45B7D1',
        'air_quality': '#95E1D3',
        'health': '#F38181'
    }
)

fig_timeline.update_layout(
    xaxis_title="Time",
    yaxis_title="Severity Level (1-5)",
    hovermode='closest',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e0e0e0')
)

st.plotly_chart(fig_timeline, use_container_width=True)

# Tabs for detailed views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 BY SOURCE",
    "🕒 HOURLY PATTERNS", 
    "📉 TRENDS",
    "🔗 CORRELATIONS",
    "📋 RAW DATA"
])

with tab1:
    st.markdown("#### Event Distribution by Data Source")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Event type breakdown
        type_counts = filtered['event_type'].value_counts().reset_index()
        type_counts.columns = ['Event Type', 'Count']
        
        fig_types = px.pie(
            type_counts,
            values='Count',
            names='Event Type',
            title='Events by Type',
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        fig_types.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_types, use_container_width=True)
    
    with col2:
        # Source breakdown
        source_counts = filtered['source'].value_counts().reset_index()
        source_counts.columns = ['Source', 'Count']
        
        fig_sources = px.bar(
            source_counts,
            x='Source',
            y='Count',
            title='Events by Data Source',
            color='Count',
            color_continuous_scale='viridis'
        )
        fig_sources.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_sources, use_container_width=True)

with tab2:
    st.markdown("#### Hourly Event Patterns")
    
    # Add hour column
    filtered['hour'] = filtered['timestamp'].dt.hour
    filtered['day_of_week'] = filtered['timestamp'].dt.day_name()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Events by hour
        hourly_counts = filtered.groupby(['hour', 'event_type']).size().reset_index(name='count')
        
        fig_hourly = px.bar(
            hourly_counts,
            x='hour',
            y='count',
            color='event_type',
            title='Events by Hour of Day',
            barmode='stack',
            labels={'hour': 'Hour (24h)', 'count': 'Event Count'}
        )
        fig_hourly.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        # Events by day of week
        daily_counts = filtered['day_of_week'].value_counts().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ]).reset_index()
        daily_counts.columns = ['Day', 'Count']
        
        fig_daily = px.bar(
            daily_counts,
            x='Day',
            y='Count',
            title='Events by Day of Week',
            color='Count',
            color_continuous_scale='reds'
        )
        fig_daily.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_daily, use_container_width=True)

with tab3:
    st.markdown("#### Trends Over Time")
    
    # Traffic duration trend
    traffic_data = filtered[filtered['event_type'] == 'traffic'].copy()
    if not traffic_data.empty and 'value' in traffic_data.columns:
        st.markdown("##### 🚗 Traffic Duration Trends")
        
        fig_traffic = px.line(
            traffic_data.sort_values('timestamp'),
            x='timestamp',
            y='value',
            color='location',
            title='Traffic Duration by Route',
            markers=True,
            labels={'value': 'Duration (minutes)', 'timestamp': 'Time'}
        )
        fig_traffic.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_traffic, use_container_width=True)
    
    # Air quality trend
    air_data = filtered[filtered['event_type'] == 'air_quality'].copy()
    if not air_data.empty and 'value' in air_data.columns:
        st.markdown("##### 🌫️ Air Quality Index Trends")
        
        fig_air = px.line(
            air_data.sort_values('timestamp'),
            x='timestamp',
            y='value',
            title='Air Quality Index Over Time',
            markers=True,
            labels={'value': 'AQI', 'timestamp': 'Time'}
        )
        fig_air.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_air, use_container_width=True)
    
    # Severity trend
    st.markdown("##### ⚠️ Average Severity Over Time")
    
    # Resample by hour
    severity_trend = filtered.set_index('timestamp').resample('1H')['severity'].mean().reset_index()
    
    fig_severity = px.line(
        severity_trend,
        x='timestamp',
        y='severity',
        title='Average Event Severity (Hourly)',
        markers=True,
        labels={'severity': 'Avg Severity', 'timestamp': 'Time'}
    )
    fig_severity.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_severity, use_container_width=True)

with tab4:
    st.markdown("#### Cross-Domain Correlations")
    
    # Find events happening within same time window
    st.info("🔍 Identifying events that occurred within the same 1-hour window")
    
    # Create hourly buckets
    filtered['hour_bucket'] = filtered['timestamp'].dt.floor('H')
    
    # Group by hour and count event types
    correlation_matrix = filtered.groupby(['hour_bucket', 'event_type']).size().unstack(fill_value=0)
    
    if not correlation_matrix.empty:
        # Calculate correlation
        corr = correlation_matrix.corr()
        
        fig_corr = px.imshow(
            corr,
            title='Event Type Correlation Matrix',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            labels=dict(color="Correlation")
        )
        fig_corr.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("**Interpretation:**")
        st.markdown("- **Positive correlation (red)**: Events tend to occur together")
        st.markdown("- **Negative correlation (blue)**: Events rarely occur together")
        st.markdown("- **Zero (white)**: No relationship")

with tab5:
    st.markdown("#### Complete Event Data Table")
    
    # Display columns
    display_cols = [
        'timestamp', 'event_type', 'domain', 'severity', 
        'description', 'location', 'source', 'value', 'unit'
    ]
    
    # Available columns
    available_cols = [col for col in display_cols if col in filtered.columns]
    
    # Format timestamp
    display_df = filtered[available_cols].copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    st.dataframe(
        display_df.sort_values('timestamp', ascending=False),
        use_container_width=True,
        height=500
    )
    
    # CSV export
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"owl_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Footer with stats
st.markdown("---")
st.markdown("### 📈 SESSION STATISTICS")

if stats:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.json(stats.get('by_type', {}))
    
    with col2:
        st.json(stats.get('by_domain', {}))
    
    with col3:
        st.metric("Avg Severity", f"{stats.get('avg_severity', 0):.2f}")
        st.metric("High Severity Events", stats.get('high_severity_count', 0))

st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
