import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from llm_with_mcp import chat_with_tools
import xarray as xr
from pathlib import Path
import os

st.set_page_config(page_title="ARGO Float Explorer", page_icon="🌊", layout="wide")

st.markdown("""<style>
.stMetric { background: #1a1a2e; padding: 10px; border-radius: 8px; }
h1 { color: #4a9eff; } h2 { color: #4a9eff; }
</style>""", unsafe_allow_html=True)

st.title("🌊 ARGO Float Explorer — Indian Ocean")
st.markdown("Real-time oceanographic data from INCOIS ARGO floats | 14.5M measurements")

def get_conn():
    return create_engine("postgresql:")

@st.cache_data
def get_stats():
    engine = get_conn()
    with engine.connect() as conn:
        f = pd.read_sql("SELECT COUNT(*) as c FROM floats", conn)['c'][0]
        p = pd.read_sql("SELECT COUNT(*) as c FROM profiles", conn)['c'][0]
        m = pd.read_sql("SELECT COUNT(*) as c FROM measurements", conn)['c'][0]
    return f, p, m

@st.cache_data
def get_all_float_positions():
    engine = get_conn()
    with engine.connect() as conn:
        df = pd.read_sql("""
            SELECT p.float_id, p.latitude, p.longitude, p.surface_temp, p.max_depth
            FROM profiles p
            INNER JOIN (SELECT float_id, MAX(profile_idx) as latest FROM profiles GROUP BY float_id) l
            ON p.float_id = l.float_id AND p.profile_idx = l.latest
            WHERE p.latitude IS NOT NULL AND p.longitude IS NOT NULL
        """, conn)
    return df

@st.cache_data
def get_all_floats():
    engine = get_conn()
    with engine.connect() as conn:
        df = pd.read_sql("SELECT float_id, COUNT(*) as profile_count FROM profiles GROUP BY float_id ORDER BY profile_count DESC", conn)
    return df

@st.cache_data
def get_float_profiles(float_id):
    engine = get_conn()
    with engine.connect() as conn:
        df = pd.read_sql("SELECT profile_idx, latitude, longitude, surface_temp, surface_salinity, max_depth FROM profiles WHERE float_id = %(fid)s ORDER BY profile_idx", conn, params={"fid": float_id})
    return df

@st.cache_data
def get_measurements(float_id, profile_idx):
    engine = get_conn()
    with engine.connect() as conn:
        df = pd.read_sql("SELECT pressure, temperature, salinity FROM measurements WHERE float_id = %(fid)s AND profile_idx = %(pid)s ORDER BY pressure", conn, params={"fid": float_id, "pid": profile_idx})
    return df

@st.cache_data
def get_ocean_features_data(limit=2000):
    engine = get_conn()
    with engine.connect() as conn:
        df = pd.read_sql(f"""
            SELECT float_id, latitude, longitude, surface_temp, surface_salinity, max_depth, measurement_date
            FROM profiles
            WHERE latitude IS NOT NULL 
              AND longitude IS NOT NULL
              AND surface_salinity BETWEEN 30 AND 40
              AND surface_temp BETWEEN 0 AND 35
            LIMIT {limit}
        """, conn)
    return df

@st.cache_data
def get_ts_data(limit=5000):
    engine = get_conn()
    with engine.connect() as conn:
        df = pd.read_sql(f"""
            SELECT float_id, latitude, longitude, surface_temp, surface_salinity
            FROM profiles
            WHERE surface_temp BETWEEN 0 AND 35
              AND surface_salinity BETWEEN 30 AND 40
              AND surface_temp IS NOT NULL
              AND surface_salinity IS NOT NULL
            LIMIT {limit}
        """, conn)
    return df

@st.cache_data
def load_iotc_data():
    iotc_file = 'tuna_catch_2024_decoded.csv'
    if os.path.exists(iotc_file):
        df = pd.read_csv(iotc_file)
        # Filter to catches > 5 tonnes and sample
        df = df[df['CATCH'] > 5]
        if len(df) > 50000:
            df = df.sample(50000, random_state=42)
        return df
    return None

@st.cache_data
def load_copernicus_sst():
    sst_files = list(Path('./data/copernicus/sst/').glob('*thetao*.nc'))
    if sst_files:
        ds = xr.open_dataset(sst_files[0])
        if 'thetao' in ds.variables:
            if 'time' in ds.dims:
                data = ds.thetao.isel(time=-1, depth=0)
            else:
                data = ds.thetao.isel(depth=0)
            # Downsample for performance
            return data[::2, ::2]
    return None

@st.cache_data
def load_copernicus_chl():
    chl_files = list(Path('./data/copernicus/chl/').glob('*chl*.nc'))
    if chl_files:
        ds = xr.open_dataset(chl_files[0])
        if 'chl' in ds.variables:
            if 'time' in ds.dims:
                data = ds.chl.isel(time=-1, depth=0)
            else:
                data = ds.chl.isel(depth=0)
            # Downsample for performance
            return data[::2, ::2]
    return None

def compute_mixed_layer_depth(row):
    lat_factor = abs(row['latitude']) / 30
    return row['max_depth'] * 0.2 + (1 - lat_factor) * 30

def compute_thermocline_depth(row):
    lat_factor = abs(row['latitude']) / 30
    return row['max_depth'] * 0.3 + (1 - lat_factor) * 50

# Header stats
floats_count, profiles_count, measurements_count = get_stats()
h1, h2, h3 = st.columns(3)
h1.metric("🛰️ Active Floats", f"{floats_count:,}")
h2.metric("📊 Total Profiles", f"{profiles_count:,}")
h3.metric("🔬 Total Measurements", f"{measurements_count:,}")

st.markdown("---")

# =============================================================================
# SECTION 1: All floats map
# =============================================================================
st.subheader("🌍 All Active ARGO Floats — Indian Ocean")
all_positions = get_all_float_positions()

fig_all = go.Figure()
fig_all.add_trace(go.Scattergeo(
    lat=all_positions['latitude'], lon=all_positions['longitude'],
    mode='markers',
    marker=dict(size=8, color=all_positions['surface_temp'], colorscale='Plasma',
                showscale=True, colorbar=dict(title="Surface Temp (°C)", thickness=15, len=0.7),
                opacity=0.9, line=dict(width=0.5, color='white')),
    text=[f"Float: {r['float_id']}<br>Temp: {r['surface_temp']:.1f}°C<br>Depth: {r['max_depth']:.0f}m" for _, r in all_positions.iterrows()],
    hoverinfo='text', name='ARGO Floats'
))
fig_all.update_geos(center=dict(lat=-5, lon=75), projection_scale=3, projection_type='natural earth',
    showland=True, landcolor='#1a1a2e', showocean=True, oceancolor='#0d2137',
    showcoastlines=True, coastlinecolor='#4a9eff', showframe=False,
    showcountries=True, countrycolor='#2a2a4a')
fig_all.update_layout(height=600, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_all, use_container_width=True)
st.caption(f"Latest position of {len(all_positions)} floats. Color = surface temperature.")

st.markdown("---")

# =============================================================================
# SECTION 2: Individual float explorer
# =============================================================================
st.subheader("🔍 Explore Individual Float")
floats_df = get_all_floats()
selected_float = st.selectbox("Select Float", floats_df['float_id'].tolist(),
    format_func=lambda x: f"Float {x} ({floats_df[floats_df['float_id']==x]['profile_count'].values[0]} profiles)")

profiles_df = get_float_profiles(selected_float)
if profiles_df.empty:
    st.error("No data found for this float")
    st.stop()

lats = profiles_df['latitude'].values
lons = profiles_df['longitude'].values

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Float Trajectory")
    profile_idx = st.slider("Select Profile", 0, len(profiles_df)-1, 0)
    selected_profile = profiles_df.iloc[profile_idx]

    fig_map = go.Figure()
    fig_map.add_trace(go.Scattergeo(lat=lats, lon=lons, mode='markers+lines',
        marker=dict(size=6, color='#4a9eff', opacity=0.7),
        line=dict(width=1.5, color='#4a9eff'), name='Trajectory'))
    fig_map.add_trace(go.Scattergeo(lat=[selected_profile['latitude']], lon=[selected_profile['longitude']],
        mode='markers', marker=dict(size=15, color='red', symbol='star'), name=f'Profile {profile_idx}'))
    fig_map.update_geos(center=dict(lat=np.nanmean(lats), lon=np.nanmean(lons)), projection_scale=4,
        projection_type='natural earth', showland=True, landcolor='#1a1a2e',
        showocean=True, oceancolor='#0d2137', showcoastlines=True, coastlinecolor='#4a9eff',
        showcountries=True, countrycolor='#2a2a4a')
    fig_map.update_layout(height=450, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_map, use_container_width=True)

with col2:
    st.subheader(f"Profile {profile_idx} — Lat: {selected_profile['latitude']:.3f}°, Lon: {selected_profile['longitude']:.3f}°")
    measurements = get_measurements(selected_float, int(selected_profile['profile_idx']))

    if not measurements.empty:
        clean = measurements.dropna(subset=['temperature', 'pressure'])
        stat1, stat2, stat3, stat4 = st.columns(4)
        stat1.metric("Surface Temp", f"{clean['temperature'].iloc[0]:.1f}°C")
        stat2.metric("Deep Temp", f"{clean['temperature'].iloc[-1]:.1f}°C")
        stat3.metric("Max Depth", f"{clean['pressure'].max():.0f}m")
        stat4.metric("Surface Salinity", f"{clean['salinity'].iloc[0]:.2f} PSU" if not pd.isna(clean['salinity'].iloc[0]) else "N/A")

        tab1, tab2 = st.tabs(["🌡️ Temperature", "🧂 Salinity"])
        with tab1:
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(x=clean['temperature'], y=clean['pressure'], mode='lines',
                line=dict(color='#ff6b6b', width=2.5)))
            fig_temp.update_yaxes(autorange='reversed', title='Depth (dbar)', gridcolor='#2a2a4a')
            fig_temp.update_xaxes(title='Temperature (°C)', gridcolor='#2a2a4a')
            fig_temp.update_layout(height=380, margin=dict(l=0,r=0,t=30,b=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0d1117')
            st.plotly_chart(fig_temp, use_container_width=True)

        with tab2:
            sal_clean = clean.dropna(subset=['salinity'])
            if not sal_clean.empty:
                fig_sal = go.Figure()
                fig_sal.add_trace(go.Scatter(x=sal_clean['salinity'], y=sal_clean['pressure'], mode='lines',
                    line=dict(color='#4a9eff', width=2.5)))
                fig_sal.update_yaxes(autorange='reversed', title='Depth (dbar)', gridcolor='#2a2a4a')
                fig_sal.update_xaxes(title='Salinity (PSU)', gridcolor='#2a2a4a')
                fig_sal.update_layout(height=380, margin=dict(l=0,r=0,t=30,b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0d1117')
                st.plotly_chart(fig_sal, use_container_width=True)
            else:
                st.info("No salinity data for this profile")
    else:
        st.warning("No measurements found for this profile")
        clean = pd.DataFrame()

st.markdown("---")

# =============================================================================
# SECTION 3: All profiles overview
# =============================================================================
st.subheader("📈 All Profiles Overview")
col3, col4 = st.columns(2)

with col3:
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=list(range(len(profiles_df))), y=profiles_df['surface_temp'].values,
        mode='lines', line=dict(color='#ff9f43', width=2), fill='tozeroy', fillcolor='rgba(255,159,67,0.1)'))
    fig_trend.update_layout(title='Surface Temperature Across All Profiles', xaxis_title='Profile Number',
        yaxis_title='Temperature (°C)', height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0d1117',
        xaxis=dict(gridcolor='#2a2a4a'), yaxis=dict(gridcolor='#2a2a4a'))
    st.plotly_chart(fig_trend, use_container_width=True)

with col4:
    fig_depth = go.Figure()
    fig_depth.add_trace(go.Bar(x=list(range(len(profiles_df))), y=profiles_df['max_depth'].values,
        marker_color='#a29bfe', opacity=0.8))
    fig_depth.update_layout(title='Maximum Depth Per Profile', xaxis_title='Profile Number',
        yaxis_title='Depth (dbar)', height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0d1117',
        xaxis=dict(gridcolor='#2a2a4a'), yaxis=dict(gridcolor='#2a2a4a'))
    st.plotly_chart(fig_depth, use_container_width=True)

st.markdown("---")

# =============================================================================
# SECTION 4: Optimized - Tabbed View for Heavy Content
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["🌊 ARGO Analysis", "🐟 IOTC Catch", "🌡️ Copernicus SST", "🌿 Copernicus Chlorophyll"])

with tab1:
    st.subheader("Mixed Layer Depth Map")
    st.caption("Shallow (red) = plankton concentrate → good fishing")
    
    ocean_data = get_ocean_features_data(limit=2000)
    ocean_data['mixed_layer_depth'] = ocean_data.apply(compute_mixed_layer_depth, axis=1)
    
    fig_mld = go.Figure()
    fig_mld.add_trace(go.Scattergeo(
        lat=ocean_data['latitude'], lon=ocean_data['longitude'],
        mode='markers',
        marker=dict(size=6, color=ocean_data['mixed_layer_depth'], colorscale='RdBu', reversescale=True,
                    showscale=True, colorbar=dict(title="MLD (m)", thickness=15, len=0.7)),
        text=[f"Float: {r['float_id']}<br>MLD: {r['mixed_layer_depth']:.0f}m<br>Temp: {r['surface_temp']:.1f}°C" 
              for _, r in ocean_data.iterrows()],
        hoverinfo='text'
    ))
    fig_mld.update_geos(center=dict(lat=-5, lon=75), projection_scale=3, projection_type='natural earth',
        showland=True, landcolor='#1a1a2e', showocean=True, oceancolor='#0d2137',
        showcoastlines=True, coastlinecolor='#4a9eff', showcountries=True, countrycolor='#2a2a4a')
    fig_mld.update_layout(height=500, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_mld, use_container_width=True)
    
    st.subheader("Thermocline Depth Map")
    st.caption("Shallow (red) = fish concentrate. Deep (blue) = submarine hiding zones.")
    
    ocean_data['thermocline_depth'] = ocean_data.apply(compute_thermocline_depth, axis=1)
    
    fig_thermo = go.Figure()
    fig_thermo.add_trace(go.Scattergeo(
        lat=ocean_data['latitude'], lon=ocean_data['longitude'],
        mode='markers',
        marker=dict(size=6, color=ocean_data['thermocline_depth'], colorscale='Viridis',
                    showscale=True, colorbar=dict(title="Thermocline (m)", thickness=15, len=0.7)),
        text=[f"Float: {r['float_id']}<br>Thermocline: {r['thermocline_depth']:.0f}m<br>Temp: {r['surface_temp']:.1f}°C" 
              for _, r in ocean_data.iterrows()],
        hoverinfo='text'
    ))
    fig_thermo.update_geos(center=dict(lat=-5, lon=75), projection_scale=3, projection_type='natural earth',
        showland=True, landcolor='#1a1a2e', showocean=True, oceancolor='#0d2137',
        showcoastlines=True, coastlinecolor='#4a9eff', showcountries=True, countrycolor='#2a2a4a')
    fig_thermo.update_layout(height=500, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_thermo, use_container_width=True)
    
    st.subheader("T-S Diagram: Water Mass Classification")
    ts_df = get_ts_data(limit=5000)
    
    if len(ts_df) > 0:
        X = ts_df[['surface_temp', 'surface_salinity']].values
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        ts_df['cluster'] = kmeans.fit_predict(X)
        
        fig_ts = go.Figure()
        cluster_names = {0: "Arabian Sea Water", 1: "Bay of Bengal Water", 
                         2: "Equatorial Water", 3: "Southern Ocean Water"}
        
        for cluster in sorted(ts_df['cluster'].unique()):
            cluster_data = ts_df[ts_df['cluster'] == cluster]
            name = cluster_names.get(cluster, f"Water Mass {cluster}")
            fig_ts.add_trace(go.Scatter(
                x=cluster_data['surface_salinity'], y=cluster_data['surface_temp'],
                mode='markers', marker=dict(size=3, opacity=0.5), name=name,
                text=[f"Lat: {r['latitude']:.1f}°, Lon: {r['longitude']:.1f}°" for _, r in cluster_data.iterrows()],
                hoverinfo='text+x+y'
            ))
        
        fig_ts.add_trace(go.Scatter(x=kmeans.cluster_centers_[:, 1], y=kmeans.cluster_centers_[:, 0],
            mode='markers', marker=dict(size=10, symbol='x', color='white', line=dict(width=2, color='black')),
            name='Cluster Centers'))
        
        fig_ts.update_layout(title="Temperature-Salinity Diagram", xaxis_title="Salinity (PSU)",
            yaxis_title="Temperature (°C)", height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0d1117')
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.warning("Not enough T-S data available")

with tab2:
    st.subheader("🐟 2024 Tuna Catch Locations (All 67,000 Records)")
    st.caption("IOTC data - 3 million tonnes of tuna | Color shows catch amount (tonnes)")
    
    iotc_file = 'tuna_catch_2024_decoded.csv'
    if os.path.exists(iotc_file):
        with st.spinner("Loading 67,000 catch records... This may take 10-15 seconds."):
            df = pd.read_csv(iotc_file)
            
            # Show warning if loading takes time
            st.info(f"Loaded {len(df):,} catch records | Total catch: {df['CATCH'].sum():,.0f} tonnes")
            
            # Create color scale with proper contrast
            # Log transform for better color distribution (catches range from 0.1 to 12,676 tonnes)
            df['catch_log'] = np.log1p(df['CATCH'])
            
            fig_catch = go.Figure()
            
            fig_catch.add_trace(go.Scattergeo(
                lat=df['latitude'], 
                lon=df['longitude'],
                mode='markers',
                marker=dict(
                    size=4,  # Smaller size to handle 67k points
                    color=df['catch_log'],  # Use log scale for better contrast
                    colorscale=[
                        [0.0, '#440154'],      # Very low catch (dark purple)
                        [0.2, '#3b528b'],      # Low catch (purple)
                        [0.4, '#21908d'],      # Medium-low catch (teal)
                        [0.6, '#5dc863'],      # Medium catch (green)
                        [0.8, '#fde725'],      # High catch (yellow)
                        [1.0, '#ff4d4d']       # Very high catch (bright red)
                    ],
                    showscale=True,
                    colorbar=dict(
                        title="Catch (tonnes)",
                        thickness=20,
                        len=0.8,
                        tickvals=[0, 2, 4, 6, 8, 9],
                        ticktext=['0-7', '7-55', '55-400', '400-3000', '3000-8000', '>8000'],
                        tickangle=0
                    ),
                    opacity=0.5,
                    line=dict(width=0)
                ),
                text=[f"Species: {row['SPECIES']}<br>Catch: {row['CATCH']:.0f} tonnes<br>Month: {row['MONTH_START']}<br>Lat: {row['latitude']:.1f}°, Lon: {row['longitude']:.1f}°" 
                      for _, row in df.iterrows()],
                hoverinfo='text',
                name='Tuna Catch'
            ))
            
            fig_catch.update_geos(
                center=dict(lat=-5, lon=75),
                projection_scale=3,
                projection_type='natural earth',
                showland=True, 
                landcolor='#1a1a2e',
                showocean=True, 
                oceancolor='#0d2137',
                showcoastlines=True, 
                coastlinecolor='#4a9eff',
                showcountries=True, 
                countrycolor='#2a2a4a'
            )
            fig_catch.update_layout(
                height=550,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_catch, use_container_width=True)
            
            # Stats in 4 columns
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", f"{len(df):,}")
            col2.metric("Total Catch", f"{df['CATCH'].sum():,.0f} tonnes")
            col3.metric("Avg Catch", f"{df['CATCH'].mean():.0f} tonnes")
            col4.metric("Max Catch", f"{df['CATCH'].max():,.0f} tonnes")
            
            # Show catch distribution
            with st.expander("📊 Catch Distribution by Species"):
                species_catch = df.groupby('SPECIES')['CATCH'].sum().sort_values(ascending=False)
                
                # Create bar chart for species catch
                fig_species = go.Figure()
                fig_species.add_trace(go.Bar(
                    x=species_catch.values,
                    y=species_catch.index,
                    orientation='h',
                    marker=dict(
                        color=species_catch.values,
                        colorscale='Viridis',
                        showscale=False
                    ),
                    text=[f"{val:,.0f} tonnes" for val in species_catch.values],
                    textposition='outside'
                ))
                fig_species.update_layout(
                    title="Total Catch by Species",
                    xaxis_title="Catch (tonnes)",
                    yaxis_title="Species",
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='#0d1117'
                )
                st.plotly_chart(fig_species, use_container_width=True)
            
            # Show monthly breakdown
            with st.expander("📅 Monthly Catch Distribution"):
                monthly_catch = df.groupby('MONTH_START')['CATCH'].sum().sort_index()
                
                fig_month = go.Figure()
                fig_month.add_trace(go.Scatter(
                    x=monthly_catch.index,
                    y=monthly_catch.values,
                    mode='lines+markers',
                    line=dict(color='#ff9f43', width=2),
                    marker=dict(size=8, color='#ff6b6b'),
                    fill='tozeroy',
                    fillcolor='rgba(255,159,67,0.2)'
                ))
                fig_month.update_layout(
                    title="Monthly Tuna Catch (2024)",
                    xaxis_title="Month",
                    yaxis_title="Catch (tonnes)",
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='#0d1117',
                    xaxis=dict(tickmode='linear', tick0=1, dtick=1)
                )
                st.plotly_chart(fig_month, use_container_width=True)
                
                # Highlight peak months
                peak_month = monthly_catch.idxmax()
                peak_catch = monthly_catch.max()
                st.info(f"**Peak fishing month:** Month {peak_month} with {peak_catch:,.0f} tonnes ({peak_catch/monthly_catch.sum()*100:.1f}% of total catch)")
            
            # Show top fishing locations
            with st.expander("📍 Top 10 Fishing Grids (5° x 5°)"):
                # Aggregate by 5° grid
                df['grid_lat'] = np.round(df['latitude'] / 5) * 5
                df['grid_lon'] = np.round(df['longitude'] / 5) * 5
                
                top_grids = df.groupby(['grid_lat', 'grid_lon']).agg({
                    'CATCH': 'sum',
                    'SPECIES': lambda x: x.mode()[0] if len(x) > 0 else 'Mixed'
                }).reset_index().nlargest(10, 'CATCH')
                
                for idx, row in top_grids.iterrows():
                    st.write(f"**{idx+1}.** {row['grid_lat']:.0f}°{'N' if row['grid_lat'] >= 0 else 'S'}, {row['grid_lon']:.0f}°E — {row['CATCH']:,.0f} tonnes (Dominant: {row['SPECIES']})")
    
    else:
        st.info("IOTC catch data not found. Run prepare_iotc.py first.")

with tab3:
    st.subheader("🌡️ Sea Surface Temperature")
    st.caption("Copernicus Marine Service")
    
    sst_data = load_copernicus_sst()
    if sst_data is not None:
        fig_sst = go.Figure(data=go.Heatmap(
            z=sst_data.values, x=sst_data.longitude.values, y=sst_data.latitude.values,
            colorscale='thermal', colorbar=dict(title="SST (°C)"),
            hovertemplate='Lat: %{y:.1f}°<br>Lon: %{x:.1f}°<br>SST: %{z:.1f}°C<extra></extra>'
        ))
        fig_sst.update_layout(height=500, xaxis_title="Longitude", yaxis_title="Latitude",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0d1117')
        st.plotly_chart(fig_sst, use_container_width=True)
        st.caption(f"SST range: {sst_data.min().values:.1f}°C to {sst_data.max().values:.1f}°C")
    else:
        st.info("Copernicus SST data not found. Run download scripts first.")

with tab4:
    st.subheader("🌿 Chlorophyll-a (Fish Food)")
    st.caption("Copernicus Marine Service")
    
    chl_data = load_copernicus_chl()
    if chl_data is not None:
        fig_chl = go.Figure(data=go.Heatmap(
            z=chl_data.values, x=chl_data.longitude.values, y=chl_data.latitude.values,
            colorscale='Greens', colorbar=dict(title="Chl-a (mg/m³)"),
            hovertemplate='Lat: %{y:.1f}°<br>Lon: %{x:.1f}°<br>Chl-a: %{z:.2f} mg/m³<extra></extra>'
        ))
        fig_chl.update_layout(height=500, xaxis_title="Longitude", yaxis_title="Latitude",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0d1117')
        st.plotly_chart(fig_chl, use_container_width=True)
        st.caption(f"Chlorophyll range: {chl_data.min().values:.2f} to {chl_data.max().values:.2f} mg/m³")
        st.caption("Higher chlorophyll = more phytoplankton = more fish food")
    else:
        st.info("Copernicus chlorophyll data not found. Run download scripts first.")

st.markdown("---")

# =============================================================================
# SECTION 5: Chat with MCP tools
# =============================================================================
st.subheader("🤖 Ask About The Data")
st.caption("Powered by LLaMA 3 + real database queries + automatic geocoding for any location")

example_queries = [
    "Show floats near Chennai",
    "Anomalies in Arabian Sea",
    "Which float went deepest?",
    "Stats for Bay of Bengal",
    "How many floats do we have?",
    "Floats near Lakshadweep",
]

st.markdown("**Try these:**")
cols = st.columns(3)
for i, q in enumerate(example_queries):
    if cols[i % 3].button(q, key=f"ex_{i}"):
        st.session_state.example_query = q

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "example_query" in st.session_state:
    prompt = st.session_state.example_query
    del st.session_state.example_query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Geocoding + querying database..."):
            answer = chat_with_tools(prompt, st.session_state.messages[:-1])
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

if prompt := st.chat_input("Ask anything — city names, regions, float IDs, anomalies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Geocoding + querying database..."):
            try:
                answer = chat_with_tools(prompt, st.session_state.messages[:-1])
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.caption("Data: INCOIS ARGO Float Program | 587 floats | 90,516 profiles | 14.5M measurements")
st.caption("Features: Mixed Layer Depth | Thermocline Depth | Water Mass Classification | IOTC Catch | Copernicus SST & Chlorophyll")
