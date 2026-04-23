import json
import psycopg2
from psycopg2.extras import RealDictCursor
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from dotenv import load_dotenv
import os
import calendar
from groq import Groq
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

#ADD YOUR OWN POSTGRESQL CREDENTIALS
DB_CONFIG = {
    "host": "",
    "database": "",
    "user": "",
    "password": ""
}

REGIONS = {
    "arabian sea":    {"lat1": 5,   "lat2": 25,  "lon1": 50,  "lon2": 78},
    "bay of bengal":  {"lat1": 5,   "lat2": 22,  "lon1": 78,  "lon2": 100},
    "equatorial":     {"lat1": -10, "lat2": 10,  "lon1": 40,  "lon2": 110},
    "southern ocean": {"lat1": -70, "lat2": -30, "lon1": 0,   "lon2": 150},
    "indian ocean":   {"lat1": -70, "lat2": 30,  "lon1": 20,  "lon2": 120},
    "lakshadweep sea":{"lat1": 5,   "lat2": 15,  "lon1": 70,  "lon2": 78},
}

REGION_ALIASES = {
    "arabian": "arabian sea",
    "bengal": "bay of bengal",
    "south": "southern ocean",
    "southern": "southern ocean",
    "equator": "equatorial",
    "tropical": "equatorial",
    "lakshadweep": "lakshadweep sea",
    "indian": "indian ocean",
    "io": "indian ocean",
    "mozambique": "mozambique",
    "maldives": "maldives",
}

geolocator = Nominatim(user_agent="argo_float_explorer_v3")

def resolve_region(raw: str) -> str | None:
    if not raw:
        return None
    lower = raw.lower().strip()
    if lower in REGIONS:
        return lower
    for alias, canonical in REGION_ALIASES.items():
        if alias in lower:
            return canonical
    for key in REGIONS:
        if key in lower or lower in key:
            return key
    return None

def llm_call(messages: list, json_mode: bool = False) -> str:
    kwargs = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content

def geocode_place(place_name: str):
    try:
        location = geolocator.geocode(place_name, timeout=10)
        if location:
            return location.latitude, location.longitude, location.address
    except (GeocoderTimedOut, Exception):
        pass
    return None, None, None

def get_conn():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

# =============================================================================
# ARGO TOOLS
# =============================================================================

def _get_database_summary(args: dict) -> dict:
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) as c FROM floats")
        floats = cur.fetchone()['c']
        cur.execute("SELECT COUNT(*) as c FROM profiles")
        profiles = cur.fetchone()['c']
        cur.execute("SELECT COUNT(*) as c FROM measurements")
        measurements = cur.fetchone()['c']
        cur.execute("""
            SELECT ROUND(AVG(surface_temp)::numeric,2) as avg_temp,
                   ROUND(MIN(surface_temp)::numeric,2) as min_temp,
                   ROUND(MAX(surface_temp)::numeric,2) as max_temp
            FROM profiles
        """)
        temps = cur.fetchone()
        return {
            "total_floats": floats,
            "total_profiles": profiles,
            "total_measurements": measurements,
            "data_range": "2002-10-24 to 2026-03-20",
            "avg_surface_temp_celsius": float(temps['avg_temp']),
            "min_surface_temp_celsius": float(temps['min_temp']),
            "max_surface_temp_celsius": float(temps['max_temp']),
        }
    finally:
        cur.close()
        conn.close()

def _get_floats_near_location(args: dict) -> dict:
    lat = args['latitude']
    lon = args['longitude']
    radius = 2
    max_radius = 20
    rows = []
    actual_radius = radius
    conn = get_conn()
    cur = conn.cursor()
    try:
        while radius <= max_radius and not rows:
            cur.execute("""
                SELECT p.float_id, p.latitude, p.longitude,
                       ROUND(p.surface_temp::numeric, 2) as surface_temp,
                       ROUND(p.surface_salinity::numeric, 3) as surface_salinity,
                       ROUND(p.max_depth::numeric, 1) as max_depth,
                       p.measurement_date::date as last_measurement,
                       ROUND((
                           6371 * acos(LEAST(1.0,
                               cos(radians(%s)) * cos(radians(p.latitude)) *
                               cos(radians(p.longitude) - radians(%s)) +
                               sin(radians(%s)) * sin(radians(p.latitude))
                           ))
                       )::numeric, 1) as distance_km
                FROM profiles p
                INNER JOIN (
                    SELECT float_id, MAX(measurement_date) as latest_date
                    FROM profiles WHERE measurement_date IS NOT NULL
                    GROUP BY float_id
                ) l ON p.float_id = l.float_id AND p.measurement_date = l.latest_date
                WHERE p.latitude BETWEEN %s AND %s
                AND p.longitude BETWEEN %s AND %s
                AND p.surface_temp IS NOT NULL
                ORDER BY distance_km
                LIMIT 8
            """, (lat, lon, lat, lat-radius, lat+radius, lon-radius, lon+radius))
            rows = cur.fetchall()
            actual_radius = radius
            radius *= 2
        if rows:
            return {
                "floats_found": len(rows),
                "search_location": {"latitude": lat, "longitude": lon},
                "search_radius_km": round(actual_radius * 111, 0),
                "floats": [dict(r) for r in rows],
            }
        return {"floats_found": 0, "message": "No floats found within 2000km"}
    finally:
        cur.close()
        conn.close()

def _get_floats_near_location_filtered(args: dict) -> dict:
    lat = args['latitude']
    lon = args['longitude']
    days_back = args.get('days_back', 365)
    radius = 2
    max_radius = 20
    rows = []
    actual_radius = radius
    conn = get_conn()
    cur = conn.cursor()
    try:
        while radius <= max_radius and not rows:
            cur.execute("""
                SELECT p.float_id,
                       ROUND(p.latitude::numeric, 3) as latitude,
                       ROUND(p.longitude::numeric, 3) as longitude,
                       ROUND(p.surface_temp::numeric, 2) as surface_temp,
                       ROUND(p.surface_salinity::numeric, 3) as surface_salinity,
                       ROUND(p.max_depth::numeric, 1) as max_depth,
                       p.measurement_date::date as measurement_date,
                       ROUND((
                           6371 * acos(LEAST(1.0,
                               cos(radians(%s)) * cos(radians(p.latitude)) *
                               cos(radians(p.longitude) - radians(%s)) +
                               sin(radians(%s)) * sin(radians(p.latitude))
                           ))
                       )::numeric, 1) as distance_km
                FROM profiles p
                WHERE p.latitude BETWEEN %s AND %s
                AND p.longitude BETWEEN %s AND %s
                AND p.surface_temp IS NOT NULL
                AND p.measurement_date > NOW() - INTERVAL '1 day' * %s
                ORDER BY distance_km, p.measurement_date DESC
                LIMIT 8
            """, (lat, lon, lat, lat-radius, lat+radius, lon-radius, lon+radius, days_back))
            rows = cur.fetchall()
            actual_radius = radius
            radius *= 2
        if rows:
            return {
                "floats_found": len(rows),
                "search_location": {"latitude": lat, "longitude": lon},
                "search_radius_km": round(actual_radius * 111, 0),
                "time_filter": f"last {days_back} days",
                "floats": [dict(r) for r in rows],
            }
        return {"floats_found": 0, "message": f"No floats found near location in last {days_back} days"}
    finally:
        cur.close()
        conn.close()

def _get_region_statistics(args: dict) -> dict:
    region = resolve_region(args.get('region', ''))
    if not region:
        return {"error": f"Unknown region '{args.get('region')}'. Options: {list(REGIONS.keys())}"}
    r = REGIONS[region]
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT COUNT(DISTINCT float_id) as active_floats,
                   COUNT(*) as total_profiles,
                   ROUND(AVG(surface_temp)::numeric, 2) as avg_surface_temp,
                   ROUND(MIN(surface_temp)::numeric, 2) as min_surface_temp,
                   ROUND(MAX(surface_temp)::numeric, 2) as max_surface_temp,
                   ROUND(AVG(surface_salinity)::numeric, 3) as avg_salinity,
                   ROUND(AVG(max_depth)::numeric, 1) as avg_max_depth,
                   ROUND(MAX(max_depth)::numeric, 1) as deepest_dive,
                   MIN(measurement_date)::date as earliest_data,
                   MAX(measurement_date)::date as latest_data
            FROM profiles
            WHERE latitude BETWEEN %s AND %s
            AND longitude BETWEEN %s AND %s
        """, (r['lat1'], r['lat2'], r['lon1'], r['lon2']))
        row = dict(cur.fetchone())
        row['region'] = region
        return row
    finally:
        cur.close()
        conn.close()

def _get_region_statistics_by_year(args: dict) -> dict:
    region = resolve_region(args.get('region', ''))
    if not region:
        return {"error": f"Unknown region '{args.get('region')}'. Options: {list(REGIONS.keys())}"}
    year = args.get('year', 2023)
    r = REGIONS[region]
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT COUNT(DISTINCT float_id) as active_floats,
                   COUNT(*) as total_profiles,
                   ROUND(AVG(surface_temp)::numeric, 2) as avg_surface_temp,
                   ROUND(MIN(surface_temp)::numeric, 2) as min_surface_temp,
                   ROUND(MAX(surface_temp)::numeric, 2) as max_surface_temp,
                   ROUND(AVG(surface_salinity)::numeric, 3) as avg_salinity,
                   ROUND(AVG(max_depth)::numeric, 1) as avg_depth
            FROM profiles
            WHERE latitude BETWEEN %s AND %s
            AND longitude BETWEEN %s AND %s
            AND EXTRACT(YEAR FROM measurement_date) = %s
        """, (r['lat1'], r['lat2'], r['lon1'], r['lon2'], year))
        row = dict(cur.fetchone())
        row['region'] = region
        row['year'] = year
        return row
    finally:
        cur.close()
        conn.close()

def _get_seasonal_statistics(args: dict) -> dict:
    region = resolve_region(args.get('region', ''))
    if not region:
        return {"error": f"Unknown region '{args.get('region')}'. Options: {list(REGIONS.keys())}"}
    month = args.get('month', 6)
    years_back = args.get('years_back', 10)
    r = REGIONS[region]
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT COUNT(*) as total_profiles,
                   COUNT(DISTINCT float_id) as active_floats,
                   ROUND(AVG(surface_temp)::numeric, 2) as avg_surface_temp,
                   ROUND(MIN(surface_temp)::numeric, 2) as min_surface_temp,
                   ROUND(MAX(surface_temp)::numeric, 2) as max_surface_temp,
                   ROUND(AVG(surface_salinity)::numeric, 3) as avg_salinity,
                   ROUND(AVG(max_depth)::numeric, 1) as avg_depth,
                   MIN(measurement_date)::date as earliest,
                   MAX(measurement_date)::date as latest
            FROM profiles
            WHERE latitude BETWEEN %s AND %s
            AND longitude BETWEEN %s AND %s
            AND EXTRACT(MONTH FROM measurement_date) = %s
            AND measurement_date > NOW() - INTERVAL '1 year' * %s
            AND surface_temp BETWEEN 0 AND 35
        """, (r['lat1'], r['lat2'], r['lon1'], r['lon2'], month, years_back))
        row = dict(cur.fetchone())
        row['region'] = region
        row['month'] = calendar.month_name[int(month)]
        row['years_back'] = years_back
        return row
    finally:
        cur.close()
        conn.close()

def _detect_anomalies(args: dict) -> dict:
    region = resolve_region(args.get('region', 'equatorial')) or 'equatorial'
    threshold = args.get('threshold', 2.0)
    r = REGIONS[region]
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            WITH region_avg AS (
                SELECT AVG(surface_temp) as mean_temp
                FROM profiles
                WHERE latitude BETWEEN %s AND %s
                AND longitude BETWEEN %s AND %s
                AND surface_temp BETWEEN 0 AND 35
            )
            SELECT p.float_id,
                   ROUND(p.latitude::numeric, 3) as latitude,
                   ROUND(p.longitude::numeric, 3) as longitude,
                   ROUND(p.surface_temp::numeric, 2) as surface_temp,
                   ROUND((p.surface_temp - ra.mean_temp)::numeric, 2) as anomaly_celsius,
                   ROUND(ra.mean_temp::numeric, 2) as region_mean_temp,
                   p.measurement_date::date as measurement_date
            FROM profiles p, region_avg ra
            WHERE p.latitude BETWEEN %s AND %s
            AND p.longitude BETWEEN %s AND %s
            AND p.surface_temp BETWEEN 0 AND 35
            AND ABS(p.surface_temp - ra.mean_temp) > %s
            ORDER BY ABS(p.surface_temp - ra.mean_temp) DESC
            LIMIT 10
        """, (r['lat1'], r['lat2'], r['lon1'], r['lon2'],
              r['lat1'], r['lat2'], r['lon1'], r['lon2'], threshold))
        rows = cur.fetchall()
        return {
            "region": region,
            "threshold_celsius": threshold,
            "anomalies_found": len(rows),
            "anomalies": [dict(row) for row in rows],
        }
    finally:
        cur.close()
        conn.close()

def _get_float_details(args: dict) -> dict:
    float_id = args['float_id']
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT COUNT(*) as total_profiles,
                   ROUND(AVG(surface_temp)::numeric, 2) as avg_surface_temp,
                   ROUND(MIN(surface_temp)::numeric, 2) as coldest_surface,
                   ROUND(MAX(surface_temp)::numeric, 2) as warmest_surface,
                   ROUND(MAX(max_depth)::numeric, 1) as deepest_dive,
                   ROUND(AVG(max_depth)::numeric, 1) as avg_depth,
                   ROUND(MIN(latitude)::numeric, 3) as min_lat,
                   ROUND(MAX(latitude)::numeric, 3) as max_lat,
                   ROUND(MIN(longitude)::numeric, 3) as min_lon,
                   ROUND(MAX(longitude)::numeric, 3) as max_lon,
                   MIN(measurement_date)::date as first_measurement,
                   MAX(measurement_date)::date as last_measurement
            FROM profiles WHERE float_id = %s
        """, (float_id,))
        row = dict(cur.fetchone())
        row['float_id'] = float_id
        return row
    finally:
        cur.close()
        conn.close()

def _find_deepest_profiles(args: dict) -> dict:
    limit = args.get('limit', 5)
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT float_id,
                   ROUND(latitude::numeric, 3) as latitude,
                   ROUND(longitude::numeric, 3) as longitude,
                   ROUND(max_depth::numeric, 1) as max_depth_meters,
                   ROUND(surface_temp::numeric, 2) as surface_temp,
                   measurement_date::date as date
            FROM profiles
            WHERE max_depth IS NOT NULL
            ORDER BY max_depth DESC LIMIT %s
        """, (limit,))
        return {"deepest_profiles": [dict(r) for r in cur.fetchall()]}
    finally:
        cur.close()
        conn.close()

def _compare_floats(args: dict) -> dict:
    results = []
    conn = get_conn()
    cur = conn.cursor()
    try:
        for fid in [args['float_id_1'], args['float_id_2']]:
            cur.execute("""
                SELECT float_id, COUNT(*) as profiles,
                       ROUND(AVG(surface_temp)::numeric, 2) as avg_temp,
                       ROUND(MAX(max_depth)::numeric, 1) as max_depth,
                       ROUND(AVG(surface_salinity)::numeric, 3) as avg_salinity,
                       MIN(measurement_date)::date as first_measurement,
                       MAX(measurement_date)::date as last_measurement
                FROM profiles WHERE float_id = %s GROUP BY float_id
            """, (fid,))
            row = cur.fetchone()
            if row:
                results.append(dict(row))
        if len(results) == 2:
            return {
                "float_1": results[0],
                "float_2": results[1],
                "temp_difference": round(
                    float(results[0]['avg_temp'] or 0) - float(results[1]['avg_temp'] or 0), 2
                ),
            }
        return {"error": "One or both floats not found"}
    finally:
        cur.close()
        conn.close()

def _get_temporal_statistics(args: dict) -> dict:
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
                EXTRACT(YEAR FROM measurement_date) as year,
                COUNT(*) as profiles,
                ROUND(AVG(surface_temp)::numeric, 2) as avg_temp
            FROM profiles
            WHERE measurement_date IS NOT NULL
            AND surface_temp BETWEEN 0 AND 35
            GROUP BY year
            ORDER BY year DESC
            LIMIT 10
        """)
        rows = cur.fetchall()
        return {
            "recent_years": [dict(r) for r in rows],
            "data_range": "2002-10-24 to 2026-03-20",
            "total_years": 23,
        }
    finally:
        cur.close()
        conn.close()

def _get_profiles_by_date(args: dict) -> dict:
    start_date = args.get('start_date', '2025-01-01')
    end_date = args.get('end_date', '2026-03-20')
    region = resolve_region(args.get('region', ''))
    query = """
        SELECT float_id,
               ROUND(latitude::numeric, 3) as latitude,
               ROUND(longitude::numeric, 3) as longitude,
               ROUND(surface_temp::numeric, 2) as surface_temp,
               ROUND(max_depth::numeric, 1) as max_depth,
               measurement_date::date as date
        FROM profiles
        WHERE measurement_date BETWEEN %s AND %s
        AND surface_temp BETWEEN 0 AND 35
    """
    params = [start_date, end_date]
    if region:
        r = REGIONS[region]
        query += " AND latitude BETWEEN %s AND %s AND longitude BETWEEN %s AND %s"
        params += [r['lat1'], r['lat2'], r['lon1'], r['lon2']]
    query += " ORDER BY measurement_date DESC LIMIT 10"
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(query, params)
        rows = cur.fetchall()
        return {
            "profiles_found": len(rows),
            "date_range": f"{start_date} to {end_date}",
            "region_filter": region or "all regions",
            "profiles": [dict(r) for r in rows],
        }
    finally:
        cur.close()
        conn.close()

def _get_active_floats_today(args: dict) -> dict:
    days = args.get('days', 30)
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT p.float_id,
                   ROUND(p.latitude::numeric, 3) as latitude,
                   ROUND(p.longitude::numeric, 3) as longitude,
                   ROUND(p.surface_temp::numeric, 2) as surface_temp,
                   p.measurement_date::date as last_seen
            FROM profiles p
            INNER JOIN (
                SELECT float_id, MAX(measurement_date) as latest
                FROM profiles
                WHERE measurement_date IS NOT NULL
                GROUP BY float_id
            ) l ON p.float_id = l.float_id AND p.measurement_date = l.latest
            WHERE p.measurement_date > NOW() - INTERVAL '1 day' * %s
            AND p.surface_temp IS NOT NULL
            ORDER BY p.measurement_date DESC
            LIMIT 20
        """, (days,))
        rows = cur.fetchall()
        return {
            "active_floats": len(rows),
            "time_window": f"last {days} days",
            "floats": [dict(r) for r in rows],
        }
    finally:
        cur.close()
        conn.close()

def _get_warming_trend(args: dict) -> dict:
    region = resolve_region(args.get('region', ''))
    if not region:
        return {"error": f"Unknown region '{args.get('region')}'. Options: {list(REGIONS.keys())}"}
    r = REGIONS[region]
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
                EXTRACT(YEAR FROM measurement_date) as year,
                ROUND(AVG(surface_temp)::numeric, 3) as avg_temp,
                COUNT(*) as profiles
            FROM profiles
            WHERE latitude BETWEEN %s AND %s
            AND longitude BETWEEN %s AND %s
            AND surface_temp BETWEEN 0 AND 35
            AND measurement_date IS NOT NULL
            GROUP BY year
            HAVING COUNT(*) > 10
            ORDER BY year
        """, (r['lat1'], r['lat2'], r['lon1'], r['lon2']))
        rows = [dict(row) for row in cur.fetchall()]
        if len(rows) < 2:
            return {"error": "Not enough data for trend analysis"}
        temps = [float(row['avg_temp']) for row in rows]
        years = [float(row['year']) for row in rows]
        n = len(temps)
        mean_y = sum(years) / n
        mean_t = sum(temps) / n
        slope = sum((years[i]-mean_y)*(temps[i]-mean_t) for i in range(n)) / \
                sum((years[i]-mean_y)**2 for i in range(n))
        return {
            "region": region,
            "trend_celsius_per_year": round(slope, 4),
            "warming": slope > 0,
            "total_change_over_period": round(slope*(years[-1]-years[0]), 3),
            "first_year": int(years[0]),
            "last_year": int(years[-1]),
            "first_year_avg_temp": temps[0],
            "last_year_avg_temp": temps[-1],
            "yearly_data": rows,
        }
    finally:
        cur.close()
        conn.close()

def _get_monsoon_analysis(args: dict) -> dict:
    year = args.get('year', 2023)
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
                CASE
                    WHEN EXTRACT(MONTH FROM measurement_date) BETWEEN 6 AND 9
                    THEN 'monsoon'
                    ELSE 'non_monsoon'
                END as season,
                ROUND(AVG(surface_temp)::numeric, 2) as avg_temp,
                ROUND(AVG(surface_salinity)::numeric, 3) as avg_salinity,
                COUNT(*) as profiles
            FROM profiles
            WHERE EXTRACT(YEAR FROM measurement_date) = %s
            AND latitude BETWEEN 5 AND 25
            AND longitude BETWEEN 60 AND 100
            AND surface_temp BETWEEN 0 AND 35
            AND surface_salinity IS NOT NULL
            GROUP BY season
            ORDER BY season
        """, (year,))
        rows = [dict(r) for r in cur.fetchall()]
        result = {"year": year, "seasons": rows}
        if len(rows) == 2:
            monsoon = next((r for r in rows if r['season'] == 'monsoon'), None)
            non_monsoon = next((r for r in rows if r['season'] == 'non_monsoon'), None)
            if monsoon and non_monsoon:
                result['salinity_drop_during_monsoon'] = round(
                    float(non_monsoon['avg_salinity']) - float(monsoon['avg_salinity']), 3
                )
                result['temp_change_during_monsoon'] = round(
                    float(monsoon['avg_temp']) - float(non_monsoon['avg_temp']), 3
                )
        return result
    finally:
        cur.close()
        conn.close()

# =============================================================================
# COPERNICUS TOOLS
# =============================================================================

def _get_sst_at_location(args: dict) -> dict:
    lat = args.get('latitude')
    lon = args.get('longitude')
    month = args.get('month', 12)
    year = args.get('year', 2024)
    
    if lat is None or lon is None:
        return {"error": "Please provide latitude and longitude"}
    
    sst_files = list(Path('./data/copernicus/sst/').glob('*thetao*.nc'))
    if not sst_files:
        return {"error": "SST data not found"}
    
    try:
        ds = xr.open_dataset(sst_files[0])
        date = f'{year}-{month:02d}-15'
        
        if 'time' in ds.dims:
            sst = ds.thetao.sel(
                time=date,
                latitude=lat,
                longitude=lon,
                depth=0.5058,
                method='nearest'
            ).values
        else:
            sst = ds.thetao.sel(
                latitude=lat,
                longitude=lon,
                depth=0.5058,
                method='nearest'
            ).values
        
        return {
            "location": {"latitude": lat, "longitude": lon},
            "sst_celsius": float(sst),
            "month": month,
            "year": year,
            "source": "Copernicus Marine Service"
        }
    except Exception as e:
        return {"error": f"Could not retrieve SST: {str(e)}"}

def _get_chlorophyll_at_location(args: dict) -> dict:
    lat = args.get('latitude')
    lon = args.get('longitude')
    month = args.get('month', 12)
    year = args.get('year', 2024)
    
    if lat is None or lon is None:
        return {"error": "Please provide latitude and longitude"}
    
    chl_files = list(Path('./data/copernicus/chl/').glob('*chl*.nc'))
    if not chl_files:
        return {"error": "Chlorophyll data not found"}
    
    try:
        ds = xr.open_dataset(chl_files[0])
        date = f'{year}-{month:02d}-15'
        
        if 'time' in ds.dims:
            chl = ds.chl.sel(
                time=date,
                latitude=lat,
                longitude=lon,
                depth=0.5058,
                method='nearest'
            ).values
        else:
            chl = ds.chl.sel(
                latitude=lat,
                longitude=lon,
                depth=0.5058,
                method='nearest'
            ).values
        
        return {
            "location": {"latitude": lat, "longitude": lon},
            "chlorophyll_mgm3": float(chl),
            "month": month,
            "year": year,
            "source": "Copernicus Marine Service"
        }
    except Exception as e:
        return {"error": f"Could not retrieve chlorophyll: {str(e)}"}

def _get_ssh_at_location(args: dict) -> dict:
    lat = args.get('latitude')
    lon = args.get('longitude')
    month = args.get('month', 12)
    year = args.get('year', 2024)
    
    if lat is None or lon is None:
        return {"error": "Please provide latitude and longitude"}
    
    ssh_files = list(Path('./data/copernicus/ssh/').glob('*zos*.nc'))
    if not ssh_files:
        return {"error": "SSH data not found"}
    
    try:
        ds = xr.open_dataset(ssh_files[0])
        date = f'{year}-{month:02d}-15'
        
        if 'time' in ds.dims:
            ssh = ds.zos.sel(
                time=date,
                latitude=lat,
                longitude=lon,
                method='nearest'
            ).values
        else:
            ssh = ds.zos.sel(
                latitude=lat,
                longitude=lon,
                method='nearest'
            ).values
        
        interpretation = "Normal SSH"
        if float(ssh) > 0.1:
            interpretation = "Positive SSH anomaly → warm eddy (good for fish)"
        elif float(ssh) < -0.1:
            interpretation = "Negative SSH anomaly → cold eddy or upwelling (nutrient-rich)"
        
        return {
            "location": {"latitude": lat, "longitude": lon},
            "ssh_meters": float(ssh),
            "month": month,
            "year": year,
            "interpretation": interpretation,
            "source": "Copernicus Marine Service"
        }
    except Exception as e:
        return {"error": f"Could not retrieve SSH: {str(e)}"}

def _get_region_ocean_conditions(args: dict) -> dict:
    region = args.get('region', '').lower()
    month = args.get('month', 12)
    year = args.get('year', 2024)
    
    region_bounds = {
        'arabian sea': {'lat_min': 5, 'lat_max': 25, 'lon_min': 50, 'lon_max': 78},
        'bay of bengal': {'lat_min': 5, 'lat_max': 22, 'lon_min': 78, 'lon_max': 100},
        'mozambique': {'lat_min': -25, 'lat_max': -10, 'lon_min': 35, 'lon_max': 45},
        'maldives': {'lat_min': -5, 'lat_max': 5, 'lon_min': 70, 'lon_max': 80},
        'equatorial': {'lat_min': -10, 'lat_max': 10, 'lon_min': 50, 'lon_max': 90},
    }
    
    if region not in region_bounds:
        return {"error": f"Unknown region. Options: {list(region_bounds.keys())}"}
    
    bounds = region_bounds[region]
    result = {"region": region, "month": month, "year": year}
    
    sst_files = list(Path('./data/copernicus/sst/').glob('*thetao*.nc'))
    if sst_files:
        try:
            ds = xr.open_dataset(sst_files[0])
            date = f'{year}-{month:02d}-15'
            if 'time' in ds.dims:
                sst_data = ds.thetao.sel(
                    time=date,
                    latitude=slice(bounds['lat_max'], bounds['lat_min']),
                    longitude=slice(bounds['lon_min'], bounds['lon_max']),
                    depth=0.5058,
                    method='nearest'
                )
            else:
                sst_data = ds.thetao.sel(
                    latitude=slice(bounds['lat_max'], bounds['lat_min']),
                    longitude=slice(bounds['lon_min'], bounds['lon_max']),
                    depth=0.5058,
                    method='nearest'
                )
            result["avg_sst_celsius"] = float(sst_data.mean().values)
        except:
            pass
    
    chl_files = list(Path('./data/copernicus/chl/').glob('*chl*.nc'))
    if chl_files:
        try:
            ds = xr.open_dataset(chl_files[0])
            date = f'{year}-{month:02d}-15'
            if 'time' in ds.dims:
                chl_data = ds.chl.sel(
                    time=date,
                    latitude=slice(bounds['lat_max'], bounds['lat_min']),
                    longitude=slice(bounds['lon_min'], bounds['lon_max']),
                    depth=0.5058,
                    method='nearest'
                )
            else:
                chl_data = ds.chl.sel(
                    latitude=slice(bounds['lat_max'], bounds['lat_min']),
                    longitude=slice(bounds['lon_min'], bounds['lon_max']),
                    depth=0.5058,
                    method='nearest'
                )
            result["avg_chlorophyll_mgm3"] = float(chl_data.mean().values)
        except:
            pass
    
    return result

# =============================================================================
# IOTC TOOLS
# =============================================================================

def _get_tuna_catch_summary(args: dict) -> dict:
    iotc_file = 'tuna_catch_2024_decoded.csv'
    if not os.path.exists(iotc_file):
        return {"error": "IOTC catch data not found. Run prepare_iotc.py first."}
    
    df = pd.read_csv(iotc_file)
    
    species = args.get('species', None)
    month = args.get('month', None)
    region = args.get('region', None)
    
    if species:
        df = df[df['SPECIES'].str.contains(species, case=False)]
    if month:
        df = df[df['MONTH_START'] == month]
    
    region_bounds = {
        'arabian sea': {'lat_min': 5, 'lat_max': 25, 'lon_min': 50, 'lon_max': 78},
        'bay of bengal': {'lat_min': 5, 'lat_max': 22, 'lon_min': 78, 'lon_max': 100},
        'mozambique': {'lat_min': -25, 'lat_max': -10, 'lon_min': 35, 'lon_max': 45},
        'maldives': {'lat_min': -5, 'lat_max': 5, 'lon_min': 70, 'lon_max': 80},
    }
    
    if region and region.lower() in region_bounds:
        bounds = region_bounds[region.lower()]
        df = df[(df['latitude'] >= bounds['lat_min']) & (df['latitude'] <= bounds['lat_max']) &
                (df['longitude'] >= bounds['lon_min']) & (df['longitude'] <= bounds['lon_max'])]
    elif region:
        return {"error": f"Unknown region. Options: {list(region_bounds.keys())}"}
    
    if len(df) == 0:
        return {"message": "No catch data found for the specified criteria"}
    
    total_catch = df['CATCH'].sum()
    species_breakdown = df.groupby('SPECIES')['CATCH'].sum().to_dict()
    monthly_breakdown = df.groupby('MONTH_START')['CATCH'].sum().to_dict()
    
    return {
        "total_catch_tonnes": round(total_catch, 0),
        "total_records": len(df),
        "species_breakdown": {k: round(v, 0) for k, v in species_breakdown.items()},
        "monthly_breakdown": {int(k): round(v, 0) for k, v in monthly_breakdown.items()},
        "peak_month": max(monthly_breakdown, key=monthly_breakdown.get) if monthly_breakdown else None,
        "filters": {"species": species or "all", "month": month or "all", "region": region or "all"}
    }

def _get_tuna_catch_by_location(args: dict) -> dict:
    iotc_file = 'tuna_catch_2024_decoded.csv'
    if not os.path.exists(iotc_file):
        return {"error": "IOTC catch data not found"}
    
    lat = args.get('latitude')
    lon = args.get('longitude')
    radius_km = args.get('radius_km', 100)
    
    if lat is None or lon is None:
        return {"error": "Please provide latitude and longitude"}
    
    df = pd.read_csv(iotc_file)
    
    lat_rad = np.radians(lat)
    lon_km_per_deg = 111 * np.cos(lat_rad)
    
    df['distance_km'] = np.sqrt(
        ((df['latitude'] - lat) * 111) ** 2 +
        ((df['longitude'] - lon) * lon_km_per_deg) ** 2
    )
    
    nearby = df[df['distance_km'] <= radius_km]
    
    if len(nearby) == 0:
        return {"message": f"No catch records within {radius_km} km of ({lat}, {lon})"}
    
    closest = nearby.nsmallest(1, 'distance_km').iloc[0]
    
    return {
        "location": {"latitude": lat, "longitude": lon},
        "search_radius_km": radius_km,
        "total_catch_tonnes": round(nearby['CATCH'].sum(), 0),
        "total_records": len(nearby),
        "species_breakdown": nearby.groupby('SPECIES')['CATCH'].sum().to_dict(),
        "closest_catch": {
            "distance_km": round(closest['distance_km'], 1),
            "catch_tonnes": round(closest['CATCH'], 0),
            "species": closest['SPECIES'],
            "month": int(closest['MONTH_START'])
        }
    }

def _compare_ocean_and_fish(args: dict) -> dict:
    lat = args.get('latitude')
    lon = args.get('longitude')
    month = args.get('month', 11)
    
    if lat is None or lon is None:
        return {"error": "Please provide latitude and longitude"}
    
    result = {"location": {"latitude": lat, "longitude": lon}, "month": month}
    
    sst_result = _get_sst_at_location({"latitude": lat, "longitude": lon, "month": month})
    if "sst_celsius" in sst_result:
        result["sst_celsius"] = sst_result["sst_celsius"]
    
    chl_result = _get_chlorophyll_at_location({"latitude": lat, "longitude": lon, "month": month})
    if "chlorophyll_mgm3" in chl_result:
        result["chlorophyll_mgm3"] = chl_result["chlorophyll_mgm3"]
    
    catch_result = _get_tuna_catch_by_location({"latitude": lat, "longitude": lon, "radius_km": 100})
    if "total_catch_tonnes" in catch_result:
        result["nearby_catch_tonnes"] = catch_result["total_catch_tonnes"]
    
    if "sst_celsius" in result and "chlorophyll_mgm3" in result:
        if result["sst_celsius"] > 28 and result["chlorophyll_mgm3"] < 0.3:
            result["interpretation"] = "Excellent tuna conditions: warm, clear water"
        elif result["sst_celsius"] > 26 and result["chlorophyll_mgm3"] < 0.5:
            result["interpretation"] = "Good tuna conditions"
        else:
            result["interpretation"] = "Moderate conditions"
    
    return result

# =============================================================================
# TOOL REGISTRY
# =============================================================================

TOOL_REGISTRY = {
    "get_database_summary": _get_database_summary,
    "get_floats_near_location": _get_floats_near_location,
    "get_floats_near_location_filtered": _get_floats_near_location_filtered,
    "get_region_statistics": _get_region_statistics,
    "get_region_statistics_by_year": _get_region_statistics_by_year,
    "get_seasonal_statistics": _get_seasonal_statistics,
    "detect_anomalies": _detect_anomalies,
    "get_float_details": _get_float_details,
    "find_deepest_profiles": _find_deepest_profiles,
    "compare_floats": _compare_floats,
    "get_temporal_statistics": _get_temporal_statistics,
    "get_profiles_by_date": _get_profiles_by_date,
    "get_active_floats_today": _get_active_floats_today,
    "get_warming_trend": _get_warming_trend,
    "get_monsoon_analysis": _get_monsoon_analysis,
    "get_sst_at_location": _get_sst_at_location,
    "get_chlorophyll_at_location": _get_chlorophyll_at_location,
    "get_ssh_at_location": _get_ssh_at_location,
    "get_region_ocean_conditions": _get_region_ocean_conditions,
    "get_tuna_catch_summary": _get_tuna_catch_summary,
    "get_tuna_catch_by_location": _get_tuna_catch_by_location,
    "compare_ocean_and_fish": _compare_ocean_and_fish,
}

def execute_tool(tool_name: str, arguments: dict) -> dict:
    fn = TOOL_REGISTRY.get(tool_name)
    if not fn:
        return {"error": f"Unknown tool '{tool_name}'. Available: {list(TOOL_REGISTRY.keys())}"}
    try:
        return fn(arguments)
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are an expert oceanographic AI assistant with access to:
1. ARGO floats (587 floats, 90,516 profiles, 14.5M measurements)
2. Copernicus Marine data (SST, Chlorophyll, SSH for 2024)
3. IOTC tuna catch data (67,000 records, 3M tonnes for 2024)

## Available Tools

### ARGO Tools
- get_database_summary: overall statistics
- get_floats_near_location(latitude, longitude): find floats near a location
- get_region_statistics(region): stats for Arabian Sea, Bay of Bengal, etc.
- detect_anomalies(region, threshold): temperature anomalies
- get_float_details(float_id): details for one float
- get_warming_trend(region): temperature trends over time
- get_monsoon_analysis(year): monsoon vs non-monsoon comparison

### Copernicus Tools
- get_sst_at_location(latitude, longitude, month, year): Sea Surface Temperature
- get_chlorophyll_at_location(latitude, longitude, month, year): Chlorophyll-a (fish food)
- get_ssh_at_location(latitude, longitude, month, year): Sea Surface Height (eddies)
- get_region_ocean_conditions(region, month, year): Complete ocean conditions for a region

### Fisheries Tools
- get_tuna_catch_summary(species, month, region): Tuna catch statistics for 2024
- get_tuna_catch_by_location(latitude, longitude, radius_km): Catch near a location
- compare_ocean_and_fish(latitude, longitude, month): Compare ocean conditions with catch

## Valid Regions
arabian sea, bay of bengal, equatorial, indian ocean, lakshadweep sea, mozambique, maldives

## Valid Species
Skipjack tuna, Yellowfin tuna, Bigeye tuna

## Response Format
Return JSON with a "tools" key containing a list of tool calls.

Example: {"tools": [{"tool": "get_region_ocean_conditions", "arguments": {"region": "mozambique", "month": 11}}]}

ONLY return valid JSON. No prose. No markdown fences."""

# =============================================================================
# MAIN CHAT FUNCTION
# =============================================================================

def chat_with_tools(user_message: str, history: list) -> str:
    # Step 1: Extract and geocode any place name
    place_resp = llm_call([
        {"role": "user", "content":
         f"Extract any place name, city, town, island, or geographic location from this text.\n"
         f"If found respond with ONLY: PLACE: <exact place name>\n"
         f"If no place found respond with ONLY: NO_PLACE\n"
         f"Text: {user_message}"}
    ]).strip()

    geocode_context = ""
    if place_resp.startswith("PLACE:"):
        place_name = place_resp.replace("PLACE:", "").strip()
        lat, lon, full_address = geocode_place(place_name)
        if lat and lon:
            geocode_context = (
                f"\n[GEOCODED: '{place_name}' → lat={lat:.4f}, lon={lon:.4f}, "
                f"address={full_address}]"
            )
        else:
            geocode_context = f"\n[GEOCODE FAILED: Could not find '{place_name}']"

    # Step 2: Ask LLM to plan which tools to call (JSON mode)
    planning_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{user_message}{geocode_context}"},
    ]
    plan_raw = llm_call(planning_messages, json_mode=True)

    # Step 3: Parse and execute all planned tool calls
    all_results = []
    try:
        start = plan_raw.find('{')
        end = plan_raw.rfind('}') + 1
        plan = json.loads(plan_raw[start:end])
        tool_calls = plan.get("tools", [])

        if not tool_calls and "tool" in plan:
            tool_calls = [plan]

        for call in tool_calls[:3]:
            tool_name = call.get("tool")
            arguments = call.get("arguments", {})
            result = execute_tool(tool_name, arguments)
            all_results.append({
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
            })

    except Exception as e:
        all_results = [{"tool": "parse_error", "error": str(e), "raw": plan_raw}]

    # Step 4: Synthesise a final answer with all tool results
    if all_results and not all(r.get("result", {}).get("error") for r in all_results):
        results_text = json.dumps(all_results, indent=2, default=str)
        final_messages = [
            {"role": "system", "content":
             "You are an expert oceanographic data assistant. "
             "Answer clearly and accurately using ONLY the real data provided. "
             "Include specific numbers, distances, and dates where available. "
             "Never invent values not present in the data. "
             "If multiple tool results are provided, synthesise them into one coherent answer."},
            {"role": "user", "content":
             f"User question: {user_message}{geocode_context}\n\n"
             f"Database results:\n{results_text}"},
        ]
    else:
        final_messages = [
            {"role": "system", "content":
             "You are an expert oceanographic assistant for Indian Ocean ARGO float data."},
            {"role": "user", "content":
             f"User question: {user_message}\n\n"
             f"The database query returned errors or no results: "
             f"{json.dumps(all_results, default=str)}\n\n"
             f"Tell the user what went wrong and suggest how to rephrase, "
             f"listing valid region names and example questions they can ask."},
        ]

    return llm_call(final_messages)
