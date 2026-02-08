"""
Abbey Road Video Analytics Dashboard Helper
Visualization functions for traffic and crowd analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import cv2
import numpy as np
from pathlib import Path
import json
from PIL import Image


def load_realtime_feed(frame_index=0):
    """Load specific frame from 24/7 processor - called for each frame"""
    try:
        # Check today's realtime frames
        date = datetime.now().strftime("%Y-%m-%d")
        realtime_dir = Path("owl_data") / date / "video_analytics" / "realtime_frames"
        stats_file = Path("owl_data") / date / "video_analytics" / "realtime_stats.json"
        
        if not stats_file.exists():
            return None, None, 0
        
        # Count available frames
        available_count = 0
        for i in range(20):
            if (realtime_dir / f"frame_{i}.jpg").exists():
                available_count += 1
        
        if available_count == 0:
            # Fall back to latest.jpg
            latest_frame = realtime_dir / "latest.jpg"
            if latest_frame.exists():
                img = cv2.imread(str(latest_frame))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    with open(stats_file) as f:
                        stats = json.load(f)
                    return img, stats, 1
            return None, None, 0
        
        # Load the specific frame requested
        target_frame = frame_index % available_count
        frame_path = realtime_dir / f"frame_{target_frame}.jpg"
        
        if frame_path.exists():
            img = cv2.imread(str(frame_path))
            if img is not None:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Load stats
                with open(stats_file) as f:
                    stats = json.load(f)
                
                return img, stats, available_count
        
        return None, None, available_count
        
    except Exception as e:
        print(f"Error loading realtime feed: {e}")
        return None, None, 0


def load_latest_detection_frame():
    """Load the most recent detection frame with bounding boxes"""
    try:
        # Check last 3 days for detection frames
        for days_back in range(3):
            date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            data_dir = Path("owl_data") / date / "video_analytics" / "frames"
            
            if data_dir.exists():
                # Get all detection frames
                frames = sorted(data_dir.glob("detection_*.jpg"), reverse=True)
                
                if frames:
                    # Load most recent frame
                    latest_frame = frames[0]
                    img = cv2.imread(str(latest_frame))
                    
                    if img is not None:
                        # Convert BGR to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Get timestamp from filename
                        timestamp = latest_frame.stem.replace('detection_', '')
                        
                        return img, timestamp
        
        return None, None
        
    except Exception as e:
        print(f"Error loading detection frame: {e}")
        return None, None


def load_latest_detection_data():
    """Load detection statistics from most recent data file"""
    try:
        # Check last 3 days for detection data
        for days_back in range(3):
            date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            data_dir = Path("owl_data") / date / "video_analytics"
            
            if data_dir.exists():
                # Get all data files
                files = sorted(data_dir.glob("abbey_road_*.json"), reverse=True)
                
                if files:
                    # Load most recent
                    with open(files[0]) as f:
                        data = json.load(f)
                    
                    if 'detections' in data and data['detections']:
                        latest = data['detections'][0]
                        return {
                            'vehicle_count': latest.get('vehicle_count', 0),
                            'person_count': latest.get('person_count', 0),
                            'objects': latest.get('objects', []),
                            'timestamp': latest.get('timestamp', '')
                        }
        
        return None
        
    except Exception as e:
        print(f"Error loading detection data: {e}")
        return None




def display_abbey_road_analytics(dm):
    """
    Display Abbey Road camera analytics with peak time analysis
    """
    st.markdown("### ðŸŽ¥ Abbey Road Crossing Analytics")
    st.markdown("*Real-time AI-powered traffic and crowd monitoring from famous Abbey Road zebra crossing*")
    
    # Display live video feed
    st.markdown("#### ðŸ“¹ Live Camera Feed")
    
    # Create two columns - one for video, one for location info
    video_col, info_col = st.columns([2, 1])
    
    with video_col:
        # Embed YouTube live stream of Abbey Road
        st.markdown("""
        <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; background: #000; border-radius: 8px; border: 2px solid #3498db;">
            <iframe 
                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
                src="https://www.youtube.com/embed/Lxqcg1qt0XU?si=bVUOLTgiYGL5hZ5u&autoplay=1&mute=1" 
                title="Abbey Road Live Camera" 
                frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
                referrerpolicy="strict-origin-when-cross-origin" 
                allowfullscreen>
            </iframe>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption(" **The Beatles' Famous Zebra Crossing**  Abbey Road, London NW8 9AY   LIVE")
    
    with info_col:
        st.info("""
        **ðŸ“ Location:**
        Abbey Road Crossing
        51.5319Â°N, 0.1774Â°W
        
        **ðŸŽ¯ Monitoring:**
        â€¢ Vehicle counts by direction
        â€¢ Pedestrian crossings
        â€¢ Traffic flow patterns
        â€¢ Tourist activity
        
        **â±ï¸ Analysis:**
        Real-time AI detection with demo mode fallback
        """)
    
    st.markdown("---")
    
    # Real-time Detection Results - 24/7 Live Feed
    st.markdown("#### 🤖 Real-Time 24/7 Object Detection")
    
    @st.fragment(run_every=0.2)
    def render_live_feed_fragment():
        # Initialize session state for frame cycling
        if 'frame_index' not in st.session_state:
            st.session_state.frame_index = 0
        
        # Auto-increment frame index for animation effect
        st.session_state.frame_index += 1
        
        # Auto-refresh display
        st.markdown(f"""
        <div style='background: linear-gradient(90deg, #e74c3c, #c0392b); padding: 8px; border-radius: 5px; text-align: center; color: white; font-weight: bold; margin-bottom: 10px;'>
            🔴 LIVE • ~5 FPS Detection Stream
        </div>
        """, unsafe_allow_html=True)
        
        # Load current frame from disk
        current_frame, realtime_stats, frame_count = load_realtime_feed(st.session_state.frame_index)
        
        if current_frame is not None and realtime_stats is not None:
            # LIVE FEED STATUS
            if frame_count > 1:
                st.success("✅ **24/7 Real-Time Processor Active** - Video feed running continuously")
            else:
                st.warning("⚠️ **Processor Idle / Starting** - Showing static feed. Starting real-time processor...")
                
            detection_col1, detection_col2 = st.columns([3, 1])
            
            with detection_col1:
                # Display current frame with unique key to prevent caching
                current_time = datetime.now()
                frame_time = current_time.strftime("%H:%M:%S.%f")[:-3]
                display_index = (st.session_state.frame_index % frame_count) if frame_count > 0 else 0
                
                # Use PIL to add timestamp overlay to prevent caching
                from PIL import Image as PILImage, ImageDraw, ImageFont
                pil_img = PILImage.fromarray(current_frame)
                draw = ImageDraw.Draw(pil_img)
                
                # Add tiny timestamp in corner to force unique image
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                except:
                    font = ImageFont.load_default()
                
                overlay_text = f"{frame_time}"
                draw.text((10, 10), overlay_text, fill=(255, 255, 255), font=font)
                
                # Convert back to array
                unique_frame = np.array(pil_img)
                
                st.image(unique_frame, 
                        caption=f"🔴 LIVE Detection Feed - {frame_time} | Frame {display_index + 1}/{frame_count}", 
                        use_container_width=True)
            
            with detection_col2:
                st.markdown("**📊 Live Stats**")
                
                # Current frame stats
                st.metric("🚗 Vehicles (Now)", realtime_stats.get('vehicle_count', 0))
                st.metric("🚶 People (Now)", realtime_stats.get('person_count', 0))
                
                # Session stats
                st.markdown("---")
                st.markdown("**📈 This Hour**")
                st.metric("Vehicles", realtime_stats.get('hourly_vehicle_count', 0))
                st.metric("People", realtime_stats.get('hourly_people_count', 0))
                
                # Total today
                st.markdown("---")
                st.markdown("**📅 Today Total**")
                st.metric("Vehicles", realtime_stats.get('total_vehicles_today', 0))
                st.metric("People", realtime_stats.get('total_people_today', 0))
                
                # Processing info
                st.markdown("---")
                frames_processed = realtime_stats.get('frames_processed', 0)
                uptime = realtime_stats.get('session_uptime_minutes', 0)
                st.caption(f"⚙️ Frames: {frames_processed:,}")
                st.caption(f"⏱️ Uptime: {uptime:.1f} min")
                
                # Timestamp
                timestamp = realtime_stats.get('timestamp', '')
                if timestamp:
                    try:
                        det_time = datetime.fromisoformat(timestamp)
                        time_ago = (datetime.now() - det_time).total_seconds()
                        st.caption(f"🕐 Updated {time_ago:.0f}s ago")
                    except:
                        pass
            
        else:
            # Check for periodic detection (10-minute capture)
            detection_frame, frame_timestamp = load_latest_detection_frame()
            detection_data = load_latest_detection_data()
            
            if detection_frame is not None and detection_data is not None:
                detection_col1, detection_col2 = st.columns([3, 1])
                
                with detection_col1:
                    # Use empty container and timestamp to force refresh
                    img_container = st.empty()
                    img_container.image(detection_frame, caption=f"Latest Detection - {frame_timestamp}", width="stretch")
                
                with detection_col2:
                    st.markdown("**🔍 Detection Stats**")
                    
                    st.metric("🚗 Vehicles", detection_data['vehicle_count'])
                    st.metric("🚶 People", detection_data['person_count'])
                    
                    # Time since last detection
                    if detection_data['timestamp']:
                        try:
                            det_time = datetime.fromisoformat(detection_data['timestamp'])
                            time_ago = datetime.now() - det_time
                            minutes_ago = int(time_ago.total_seconds() / 60)
                            st.caption(f"⏱️ {minutes_ago} min ago")
                        except:
                            pass
                    
                    if st.button("🔄 Refresh Detection", width="stretch"):
                        st.rerun()
                    
                    with st.expander("📊 Detected Objects"):
                        objects = detection_data.get('objects', [])
                        if objects:
                            for obj in objects[:10]:  # Show first 10
                                st.text(f"{obj.get('class', 'unknown')} ({obj.get('confidence', 0):.2f})")
                        else:
                            st.caption("No detailed object data (demo mode)")
            elif detection_data is not None:
                # Data exists but no annotated frame yet
                st.warning("""
                **⚠️ Detection Data Found (Demo Mode Active)**
                
                The collector is running and saving data, but no annotated frames with bounding boxes yet.
                
                **This means:**
                - `continuous_collector.py` is running in **demo mode** (realistic simulated data)
                - YouTube 30-second video capture needs `yt-dlp` to work
                - Or YouTube capture is failing and falling back to demo mode
                
                **To enable real-time 30-second clip detection:**
                ```bash
                # Install yt-dlp first
                pip install yt-dlp
                
                # Then restart the collector
                python continuous_collector.py
                ```
                
                **What real mode does:**
                - 📹 Captures 30 seconds of live video every 10 minutes
                - 🔍 Processes ~30 frames in real-time
                - ⭐ Saves best frame with most vehicle/people activity
                - 🎨 Draws bounding boxes on detected objects
                
                **Current demo data:** {0} vehicles, {1} people detected
                
                *Once YouTube capture works, annotated frames will appear here!*
                """.format(detection_data.get('vehicle_count', 0), detection_data.get('person_count', 0)))
                
                if st.button("🔄 Refresh to Check for Frames", width="stretch"):
                    st.rerun()
            else:
                st.info("""
                **🎥 Real-Time Detection Not Yet Active**
                
                No detection data found in the database yet.
                
                **To start 24/7 real-time detection:**
                ```bash
                # Run the real-time video processor
                python realtime_video_processor.py
                ```
                
                **How it works (continuous detection):**
                - 📹 **Continuous 24/7 video stream** from YouTube
                - 🔍 Processes **2 frames per second** in real-time
                - 🤖 YOLO detects vehicles and people in **every frame**
                - 💾 **Auto-updates** latest detection every 2 seconds
                - 🎨 Draws bounding boxes (Green=vehicles, Blue=people)
                - 📊 **Live counting**: vehicles and people passing through
                - ✨ Dashboard **auto-refreshes** to show live feed
                
                **OR for periodic captures (every 10 minutes):**
                ```bash
                python continuous_collector.py
                ```
                
                *Install yt-dlp for real YouTube capture: `pip install yt-dlp`*
                """)
    
    # Run the fragment
    render_live_feed_fragment()
    
    st.markdown("---")
        # Load data  
    video_df = dm.load_all_video_analytics()
    
    if video_df.empty:
        st.info("ðŸ“¹ **Abbey Road Video Analytics - Demo Mode**")
        st.markdown("""
        The system is ready to collect Abbey Road camera data!
        
        **Currently running in DEMO MODE** with realistic traffic simulations.
        
        **To generate demo data right now:**
        ```bash
        cd owl_engine
        python data_collection/video_analytics.py
        ```
        
        **To auto-collect with continuous collector:**
        ```bash
        python continuous_collector.py
        ```
        
        This will generate realistic traffic patterns showing:
        - ðŸš— Vehicle counts by direction (N/S/E/W)
        - ðŸš¶ Pedestrian/crowd density
        - ðŸ“Š Rush hour vs. quiet periods
        - ðŸ“¸ Tourist peak times
        
        *Demo mode uses realistic time-based patterns without requiring camera access.*
        """)
        
        # Show sample data button
        if st.button("ðŸŽ¬ Generate Sample Data Now"):
            with st.spinner("Generating realistic Abbey Road traffic data..."):
                try:
                    from data_collection.video_analytics import AbbeyRoadCollector
                    collector = AbbeyRoadCollector(demo_mode=True)
                    
                    # Generate several samples
                    for i in range(5):
                        collector.collect()
                    
                    st.success("âœ… Generated 5 sample data points!")
                    st.info("ðŸ”„ Refresh the page to see the charts")
                except Exception as e:
                    st.error(f"Error generating data: {e}")
        
        return
    
    # Show data summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "ðŸ“Š Data Points",
            len(video_df),
            delta=f"{len(video_df[video_df['timestamp'] > datetime.now() - timedelta(hours=24)])} (24h)"
        )
    
    with col2:
        if 'vehicle_count' in video_df.columns:
            total_vehicles = video_df['vehicle_count'].sum()
            st.metric("ðŸš— Total Vehicles", f"{int(total_vehicles):,}")
    
    with col3:
        if 'person_count' in video_df.columns:
            total_people = video_df['person_count'].sum()
            st.metric("ðŸš¶ Total Pedestrians", f"{int(total_people):,}")
    
    with col4:
        latest_time = video_df['timestamp'].max()
        time_diff = datetime.now() - latest_time
        st.metric("â±ï¸ Last Update", f"{int(time_diff.total_seconds() / 60)} min ago")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "ðŸ“ˆ Peak Traffic Times",
        "ðŸ‘¥ Crowd Density",
        "ðŸ”€ Traffic Flow",
        "ðŸ“Š Statistics"
    ])
    
    with tab1:
        display_peak_traffic_times(video_df)
    
    with tab2:
        display_crowd_density_analysis(video_df)
    
    with tab3:
        display_traffic_flow_directions(video_df)
    
    with tab4:
        display_video_statistics(video_df)


def display_peak_traffic_times(df):
    """Show peak traffic times with hourly breakdown"""
    st.markdown("#### ðŸš¦ Peak Vehicle Traffic Times")
    
    if 'hour' in df.columns:
        # Hourly aggregation
        hourly_df = df.groupby('hour').agg({
            'cars_north': 'sum',
            'cars_south': 'sum',
            'cars_east': 'sum',
            'cars_west': 'sum',
            'buses_north': 'sum',
            'buses_south': 'sum'
        }).reset_index()
        
        # Total vehicles per hour
        hourly_df['total_vehicles'] = (
            hourly_df['cars_north'] + hourly_df['cars_south'] +
            hourly_df['cars_east'] + hourly_df['cars_west'] +
            hourly_df['buses_north'] + hourly_df['buses_south']
        )
        
        # Create stacked bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='North',
            x=hourly_df['hour'],
            y=hourly_df['cars_north'],
            marker_color='#3498db'
        ))
        
        fig.add_trace(go.Bar(
            name='South',
            x=hourly_df['hour'],
            y=hourly_df['cars_south'],
            marker_color='#e74c3c'
        ))
        
        fig.add_trace(go.Bar(
            name='East',
            x=hourly_df['hour'],
            y=hourly_df['cars_east'],
            marker_color='#2ecc71'
        ))
        
        fig.add_trace(go.Bar(
            name='West',
            x=hourly_df['hour'],
            y=hourly_df['cars_west'],
            marker_color='#f39c12'
        ))
        
        fig.update_layout(
            barmode='stack',
            title='Vehicle Count by Hour and Direction',
            xaxis_title='Hour',
            yaxis_title='Vehicle Count',
            height=400,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, width="stretch")
        
        # Find peak hour
        peak_hour = hourly_df.loc[hourly_df['total_vehicles'].idxmax()]
        st.success(f"ðŸ”¥ **Peak Traffic**: {peak_hour['hour']} with {int(peak_hour['total_vehicles'])} vehicles")
        
    elif 'vehicle_count' in df.columns and 'timestamp' in df.columns:
        # Snapshot data - aggregate by hour
        df['hour'] = df['timestamp'].dt.floor('h')  # Changed from 'H' to 'h'
        hourly_counts = df.groupby('hour')['vehicle_count'].sum().reset_index()
        
        fig = px.line(
            hourly_counts,
            x='hour',
            y='vehicle_count',
            title='Vehicle Count Over Time',
            markers=True
        )
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No vehicle count data available yet")


def display_crowd_density_analysis(df):
    """Show crowd density patterns"""
    st.markdown("#### ðŸ‘¥ Pedestrian Crowd Density")
    
    if 'person_count' in df.columns and 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.floor('h')  # Changed from 'H' to 'h'
        hourly_people = df.groupby('hour')['person_count'].agg(['mean', 'max', 'sum']).reset_index()
        
        # Create dual-axis chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hourly_people['hour'],
            y=hourly_people['mean'],
            name='Average',
            mode='lines+markers',
            line=dict(color='#3498db', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=hourly_people['hour'],
            y=hourly_people['max'],
            name='Peak',
            mode='lines+markers',
            line=dict(color='#e74c3c', width=2, dash='dash'),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Pedestrian Count by Hour',
            xaxis_title='Hour',
            yaxis_title='People Count',
            height=400,
            template='plotly_dark',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, width="stretch")
        
        # Peak crowd time
        peak_crowd = hourly_people.loc[hourly_people['max'].idxmax()]
        st.warning(f"ðŸ‘¥ **Busiest Crossing Time**: {peak_crowd['hour']} with {int(peak_crowd['max'])} people")
        
    elif 'avg_crowd_density' in df.columns:
        fig = px.line(
            df,
            x='timestamp',
            y='avg_crowd_density',
            title='Average Crowd Density Over Time',
            markers=True
        )
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No pedestrian count data available yet")


def display_traffic_flow_directions(df):
    """Show traffic flow in different directions"""
    st.markdown("#### ðŸ”€ Traffic Flow by Direction")
    
    if 'cars_north' in df.columns:
        # Calculate total by direction
        direction_totals = {
            'North': df['cars_north'].sum() + df['buses_north'].sum() + df.get('motorcycles_north', pd.Series([0])).sum(),
            'South': df['cars_south'].sum() + df['buses_south'].sum() + df.get('motorcycles_south', pd.Series([0])).sum(),
            'East': df['cars_east'].sum(),
            'West': df['cars_west'].sum()
        }
        
        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(direction_totals.keys()),
            values=list(direction_totals.values()),
            hole=0.4,
            marker=dict(colors=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        )])
        
        fig.update_layout(
            title='Traffic Distribution by Direction',
            height=400,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, width="stretch")
        
        # Show breakdown
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("â¬†ï¸ North", f"{int(direction_totals['North'])}")
        with col2:
            st.metric("â¬‡ï¸ South", f"{int(direction_totals['South'])}")
        with col3:
            st.metric("âž¡ï¸ East", f"{int(direction_totals['East'])}")
        with col4:
            st.metric("â¬…ï¸ West", f"{int(direction_totals['West'])}")
    else:
        st.info("Direction-specific data not available yet")


def display_video_statistics(df):
    """Show overall statistics and insights"""
    st.markdown("#### ðŸ“Š Analytics Statistics")
    
    # Data quality metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**ðŸ“ˆ Data Coverage**")
        if 'frame_count' in df.columns:
            total_frames = df['frame_count'].sum()
            st.write(f"Total frames analyzed: **{int(total_frames):,}**")
        
        if 'timestamp' in df.columns:
            time_range = df['timestamp'].max() - df['timestamp'].min()
            st.write(f"Time range: **{time_range}**")
            
            hours_covered = time_range.total_seconds() / 3600
            st.write(f"Hours of coverage: **{hours_covered:.1f}h**")
    
    with col2:
        st.markdown("**ðŸŽ¯ Detection Summary**")
        if 'vehicle_count' in df.columns:
            avg_vehicles = df['vehicle_count'].mean()
            st.write(f"Avg vehicles per snapshot: **{avg_vehicles:.1f}**")
        
        if 'person_count' in df.columns:
            avg_people = df['person_count'].mean()
            st.write(f"Avg people per snapshot: **{avg_people:.1f}**")
    
    # Recent data table
    if len(df) > 0:
        st.markdown("**ðŸ“‹ Recent Detections**")
        recent = df.tail(10).sort_values('timestamp', ascending=False)
        
        # Select relevant columns
        display_cols = [col for col in ['timestamp', 'vehicle_count', 'person_count', 'hour'] if col in recent.columns]
        if display_cols:
            st.dataframe(recent[display_cols], width="stretch")


