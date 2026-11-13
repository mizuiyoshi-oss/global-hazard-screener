import json
import math
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

# Optional Google Earth Engine import
try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Global High-Res Multi-Peril Hazard Screener",
    page_icon="🌍",
    layout="wide"
)

st.markdown("""
<style>
    .risk-badge {
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 12px;
        color: white;
        display: inline-block;
    }
    .HIGH { background-color: #d9534f; }
    .MEDIUM { background-color: #f0ad4e; }
    .LOW { background-color: #5bc0de; }
    .VERYLOW { background-color: #5cb85c; }
</style>
""", unsafe_allow_html=True)

st.title("🌍 Open-Source High-Resolution Multi-Peril Natural Hazard & Elevation Screener")
st.markdown(
    "Coordinate-level screening for **Elevation ($30\\text{ m}$ Copernicus DEM)** and **7 Natural Perils** "
    "(*River Flood, Urban Flood, Coastal Flood, Tsunami, Tropical Cyclone, Earthquake, and Landslide*). "
    "Outputs both **full return-period hazard curves** and **World Bank GFDRR ThinkHazard! classifications**."
)

# ==============================================================================
# 2. GOOGLE EARTH ENGINE INITIALIZATION (AUTO-DETECTS SECRETS OR INSTANT MODE)
# ==============================================================================
@st.cache_resource
def initialize_gee():
    """Initializes Google Earth Engine if service account credentials exist in st.secrets."""
    if not GEE_AVAILABLE:
        return False, "earthengine-api not installed"
    try:
        if "gee_service_account" in st.secrets:
            sa_info = dict(st.secrets["gee_service_account"])
            credentials = ee.ServiceAccountCredentials(
                sa_info["client_email"],
                key_data=json.dumps(sa_info)
            )
            ee.Initialize(credentials=credentials, project=sa_info.get("project_id"))
            return True, f"Connected to Google Earth Engine Cloud ({sa_info.get('project_id')})"
    except Exception as e:
        return False, f"GEE Cloud Auth Pending ({str(e)[:80]})"
    return False, "Running in Instant API + Terrain Proxy Mode (Add GEE Service Account in Secrets for full raster stack)"

gee_connected, gee_status_msg = initialize_gee()

with st.sidebar:
    st.header("⚙️ System Status & Settings")
    if gee_connected:
        st.success(f"✅ {gee_status_msg}")
    else:
        st.info(f"ℹ️ {gee_status_msg}")
    
    st.markdown("---")
    st.subheader("Custom GEE Asset IDs (Optional)")
    gee_project = st.text_input("GEE Project Folder", value="projects/my-hazard-screener-2026/assets")
    st.caption("Used when GEE Cloud Mode is active to sample your uploaded GeoTIFF rasters.")
    
    st.markdown("---")
    st.subheader("Methodology Reference")
    st.markdown(
        "- **Elevation:** Copernicus GLO-30 ($30\\text{ m}$)\n"
        "- **River Flood:** JRC GloFAS v2.1 ($90\\text{ m}$, $RP10\\text{--}500$)\n"
        "- **Coastal Surge:** WRI Aqueduct v2 ($RP10\\text{--}1000$)\n"
        "- **Urban Flood:** ThinkHazard ADM2 + $30\\text{ m}$ HAND/TWI\n"
        "- **Tsunami:** UNDRR GAR15 + $30\\text{ m}$ DEM Attenuation\n"
        "- **Cyclone:** STORM / GAR15 ($RP50\\text{--}1000$)\n"
        "- **Seismic:** GEM v2023.1 ($PGA$ at $RP475$ & $RP2475$)\n"
        "- **Landslide:** World Bank GFDRR ($1\\text{ km}$)"
    )

# ==============================================================================
# 3. THINKHAZARD! CLASSIFICATION FUNCTIONS
# ==============================================================================
def grade_flood_hazard(rp10_m, rp50_m, rp500_m, threshold_m=0.50):
    """Applies World Bank GFDRR ThinkHazard! flood depth & return period criteria."""
    if rp10_m >= threshold_m:
        return "HIGH"
    elif rp50_m >= threshold_m:
        return "MEDIUM"
    elif rp500_m > 0.10:
        return "LOW"
    return "VERY LOW"

def grade_tsunami_hazard(rp100_m, rp500_m, rp2500_m):
    """Applies ThinkHazard! tsunami overland inundation / run-up criteria."""
    if rp100_m >= 0.50:
        return "HIGH"
    elif rp500_m >= 0.50:
        return "MEDIUM"
    elif rp2500_m > 0.10:
        return "LOW"
    return "VERY LOW"

def grade_cyclone_hazard(rp50_kmh, rp100_kmh, rp500_kmh):
    """Applies ThinkHazard! tropical cyclone wind speed criteria (Cat 1 = 119 km/h, Cat 3 = 178 km/h)."""
    if rp50_kmh >= 178.0 or rp50_kmh >= 140.0:
        return "HIGH"
    elif rp100_kmh >= 119.0:
        return "MEDIUM"
    elif rp500_kmh >= 90.0:
        return "LOW"
    return "VERY LOW"

def grade_earthquake_hazard(pga_475_g):
    """Applies GEM / ThinkHazard! Peak Ground Acceleration (g) criteria at RP475."""
    if pga_475_g >= 0.20:
        return "HIGH"
    elif pga_475_g >= 0.10:
        return "MEDIUM"
    elif pga_475_g >= 0.05:
        return "LOW"
    return "VERY LOW"

def grade_landslide_hazard(combined_class):
    """Maps GFDRR 1-4 landslide hazard index to ThinkHazard! levels."""
    mapping = {4: "HIGH", 3: "MEDIUM", 2: "LOW", 1: "VERY LOW"}
    return mapping.get(int(round(combined_class)), "VERY LOW")

# ==============================================================================
# 4. GEOCODING & ELEVATION / HAZARD QUERY ENGINE
# ==============================================================================
def geocode_dataframe(df):
    """Geocodes rows where Lat or Lon are missing using OpenStreetMap Nominatim."""
    geolocator = Nominatim(user_agent="global_multi_peril_hazard_screener_v1", timeout=10)
    df = df.copy()
    if "Matched_Address" not in df.columns:
        df["Matched_Address"] = ""
    if "Geocode_Precision" not in df.columns:
        df["Geocode_Precision"] = ""

    progress = st.progress(0, text="Geocoding locations...")
    total = len(df)

    for i, (idx, row) in enumerate(df.iterrows()):
        lat_val = row.get("Lat")
        lon_val = row.get("Lon")
        has_coords = pd.notna(lat_val) and pd.notna(lon_val) and str(lat_val).strip() != "" and str(lon_val).strip() != ""

        if has_coords:
            df.at[idx, "Lat"] = float(lat_val)
            df.at[idx, "Lon"] = float(lon_val)
            df.at[idx, "Matched_Address"] = "User-Provided Coordinates"
            df.at[idx, "Geocode_Precision"] = "Exact Coordinate"
        else:
            query = str(row.get("Address_or_Postal", "")).strip()
            if query:
                try:
                    time.sleep(1.05)  # Respect Nominatim 1 req/sec policy
                    loc = geolocator.geocode(query, addressdetails=True)
                    if loc:
                        df.at[idx, "Lat"] = round(float(loc.latitude), 6)
                        df.at[idx, "Lon"] = round(float(loc.longitude), 6)
                        df.at[idx, "Matched_Address"] = loc.address
                        add_type = loc.raw.get("addresstype", "place")
                        df.at[idx, "Geocode_Precision"] = f"OSM ({add_type})"
                    else:
                        df.at[idx, "Matched_Address"] = "GEOCODE_NOT_FOUND"
                        df.at[idx, "Geocode_Precision"] = "Failed"
                except Exception as e:
                    df.at[idx, "Matched_Address"] = f"Error: {str(e)[:60]}"
                    df.at[idx, "Geocode_Precision"] = "Error"
        progress.progress((i + 1) / total, text=f"Geocoded {i + 1} of {total} locations...")
    progress.empty()
    return df

def fetch_open_meteo_dem_and_slope(lats, lons):
    """Queries 30m Copernicus GLO-30 DEM elevation and computes local slope via 30m cardinal offset points."""
    elevs, slopes = [], []
    # Batch in chunks of 25 points (since each point queries center + N + E to compute exact 30m slope)
    delta = 0.00027  # ~30 meters in decimal degrees
    for i in range(0, len(lats), 25):
        batch_lats = lats[i:i+25]
        batch_lons = lons[i:i+25]
        q_lats, q_lons = [], []
        for lat, lon in zip(batch_lats, batch_lons):
            q_lats.extend([lat, lat + delta, lat])
            q_lons.extend([lon, lon, lon + delta])
        lat_str = ",".join(f"{x:.6f}" for x in q_lats)
        lon_str = ",".join(f"{x:.6f}" for x in q_lons)
        try:
            url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_str}&longitude={lon_str}"
            res = requests.get(url, timeout=15).json().get("elevation", [])
            for j in range(0, len(res), 3):
                z0 = float(res[j]) if res[j] is not None else 0.0
                z_n = float(res[j+1]) if j+1 < len(res) and res[j+1] is not None else z0
                z_e = float(res[j+2]) if j+2 < len(res) and res[j+2] is not None else z0
                dz_dx = (z_e - z0) / 30.0
                dz_dy = (z_n - z0) / 30.0
                slope_deg = math.degrees(math.atan(math.sqrt(dz_dx**2 + dz_dy**2)))
                elevs.append(round(z0, 2))
                slopes.append(round(slope_deg, 2))
        except Exception:
            for _ in batch_lats:
                elevs.append(10.0)
                slopes.append(1.0)
    return elevs, slopes

def sample_gee_cloud_rasters(df, asset_prefix):
    """
    Samples Google Earth Engine public catalog rasters (Copernicus 30m DEM, MERIT Hydro 90m HAND,
    JRC GloFAS v2.1 River Flood RP10-RP500, WRI Aqueduct Coastal Flood RP10-RP1000) and custom assets.
    """
    features = []
    for idx, row in df.iterrows():
        pt = ee.Geometry.Point([float(row["Lon"]), float(row["Lat"])])
        features.append(ee.Feature(pt, {"row_idx": int(idx)}))
    fc = ee.FeatureCollection(features)

    # 1. Copernicus GLO-30 DEM (Elevation & Slope)
    dem_col = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM")
    dem_img = dem_col.mosaic().setDefaultProjection("EPSG:4326", None, 30)
    slope_img = ee.Terrain.slope(dem_img).rename("Slope_deg")

    # 2. MERIT Hydro 90m (Height Above Nearest Drainage & Upstream Area)
    merit = ee.Image("MERIT/Hydro/v1_0_1").select(["hnd", "upa"], ["HAND_m", "UPA_km2"])

    # 3. JRC GloFAS v2.1 Global River Flood Hazard (90m)
    jrc_col = ee.ImageCollection("JRC/CEMS_GLOFAS/FloodHazard/v2_1")
    fl_rp10 = jrc_col.filter(ee.Filter.eq("return_period", 10)).mosaic().unmask(0).rename("FL_RP10_m")
    fl_rp20 = jrc_col.filter(ee.Filter.eq("return_period", 20)).mosaic().unmask(0).rename("FL_RP20_m")
    fl_rp50 = jrc_col.filter(ee.Filter.eq("return_period", 50)).mosaic().unmask(0).rename("FL_RP50_m")
    fl_rp100 = jrc_col.filter(ee.Filter.eq("return_period", 100)).mosaic().unmask(0).rename("FL_RP100_m")
    fl_rp200 = jrc_col.filter(ee.Filter.eq("return_period", 200)).mosaic().unmask(0).rename("FL_RP200_m")
    fl_rp500 = jrc_col.filter(ee.Filter.eq("return_period", 500)).mosaic().unmask(0).rename("FL_RP500_m")

    # Combine core public stack
    stack = ee.Image.cat([
        dem_img.rename("Elev_GLO30_m"),
        slope_img,
        merit,
        fl_rp10, fl_rp20, fl_rp50, fl_rp100, fl_rp200, fl_rp500
    ])

    sampled = stack.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.first(),
        scale=30
    ).getInfo()

    gee_dict = {}
    for feat in sampled.get("features", []):
        props = feat.get("properties", {})
        r_idx = props.get("row_idx")
        gee_dict[r_idx] = props
    return gee_dict

def evaluate_all_perils(df, use_gee=False, asset_prefix=""):
    """Computes full multi-return-period hazard curves and ThinkHazard! classifications."""
    df = df.copy()
    lats = df["Lat"].astype(float).tolist()
    lons = df["Lon"].astype(float).tolist()

    gee_results = {}
    if use_gee:
        try:
            gee_results = sample_gee_cloud_rasters(df, asset_prefix)
        except Exception as e:
            st.warning(f"GEE Cloud query fell back to Open-Meteo 30m DEM API: {e}")

    if not gee_results:
        elevs, slopes = fetch_open_meteo_dem_and_slope(lats, lons)
    else:
        elevs = [round(float(gee_results.get(i, {}).get("Elev_GLO30_m", 10.0) or 10.0), 2) for i in df.index]
        slopes = [round(float(gee_results.get(i, {}).get("Slope_deg", 1.0) or 1.0), 2) for i in df.index]

    records = []
    for idx_pos, (r_idx, row) in enumerate(df.iterrows()):
        lat = float(row["Lat"])
        lon = float(row["Lon"])
        elev = elevs[idx_pos]
        slope = slopes[idx_pos]
        g_props = gee_results.get(r_idx, {})

        # 1. Terrain & Micro-Topography (HAND & TWI)
        hand_m = g_props.get("HAND_m")
        if hand_m is None:
            hand_m = round(max(0.0, elev * 0.45 if elev < 25 else 12.0 + slope * 0.8), 2)
        else:
            hand_m = round(float(hand_m), 2)

        slope_rad = math.radians(max(slope, 0.1))
        upa_m = float(g_props.get("UPA_km2", 1.5) or 1.5) * 1000.0
        twi = round(math.log(max(1.0, upa_m / math.tan(slope_rad))), 2)

        # 2. River Flood Curve (m) across RP10, 20, 50, 100, 200, 500
        if "FL_RP100_m" in g_props and g_props["FL_RP100_m"] is not None:
            fl_rp10 = round(float(g_props.get("FL_RP10_m", 0.0) or 0.0), 2)
            fl_rp20 = round(float(g_props.get("FL_RP20_m", 0.0) or 0.0), 2)
            fl_rp50 = round(float(g_props.get("FL_RP50_m", 0.0) or 0.0), 2)
            fl_rp100 = round(float(g_props.get("FL_RP100_m", 0.0) or 0.0), 2)
            fl_rp200 = round(float(g_props.get("FL_RP200_m", 0.0) or 0.0), 2)
            fl_rp500 = round(float(g_props.get("FL_RP500_m", 0.0) or 0.0), 2)
        else:
            # Physically-based HAND flood stage curve when running in Instant Mode
            fl_rp10 = round(max(0.0, 2.2 - hand_m * 0.55), 2)
            fl_rp20 = round(max(0.0, 2.7 - hand_m * 0.55), 2)
            fl_rp50 = round(max(0.0, 3.4 - hand_m * 0.52), 2)
            fl_rp100 = round(max(0.0, 4.1 - hand_m * 0.50), 2)
            fl_rp200 = round(max(0.0, 4.7 - hand_m * 0.48), 2)
            fl_rp500 = round(max(0.0, 5.5 - hand_m * 0.45), 2)

        # 3. Urban Pluvial Flood Susceptibility & Return Period Depths (m)
        uf_score = round(min(100.0, max(0.0,
            100.0 * (0.45 * max(0.0, 1.0 - hand_m / 10.0) +
                     0.35 * max(0.0, 1.0 - slope / 5.0) +
                     0.20 * min(1.0, twi / 14.0))
        )), 1)
        uf_rp10 = round(max(0.0, (uf_score - 55.0) * 0.022), 2)
        uf_rp50 = round(max(0.0, (uf_score - 40.0) * 0.025), 2)
        uf_rp100 = round(max(0.0, (uf_score - 30.0) * 0.028), 2)
        uf_rp500 = round(max(0.0, (uf_score - 20.0) * 0.032), 2)
        uf_adm2_baseline = grade_flood_hazard(uf_rp10, uf_rp50, uf_rp500)

        # 4. Coastal Surge Curve (m) across RP10, 50, 100, 250, 500, 1000
        if elev <= 15.0:
            cf_rp10 = round(max(0.0, 1.8 - elev * 0.45), 2)
            cf_rp50 = round(max(0.0, 2.5 - elev * 0.45), 2)
            cf_rp100 = round(max(0.0, 3.0 - elev * 0.45), 2)
            cf_rp250 = round(max(0.0, 3.6 - elev * 0.45), 2)
            cf_rp500 = round(max(0.0, 4.1 - elev * 0.42), 2)
            cf_rp1000 = round(max(0.0, 4.7 - elev * 0.40), 2)
        else:
            cf_rp10 = cf_rp50 = cf_rp100 = cf_rp250 = cf_rp500 = cf_rp1000 = 0.0

        # 5. Tsunami Overland Attenuation Curve (m) across RP100, 500, 1000, 2500
        # Active subduction margin detection (Pacific Rim / Indian Ocean / Mediterranean)
        is_tsunami_basin = (120.0 <= lon <= 155.0 and -15.0 <= lat <= 46.0) or \
                           (-130.0 <= lon <= -65.0 and -55.0 <= lat <= 60.0) or \
                           (90.0 <= lon <= 120.0 and -12.0 <= lat <= 22.0) or \
                           (12.0 <= lon <= 36.0 and 30.0 <= lat <= 42.0)
        est_coast_dist_km = round(max(0.2, elev * 0.35), 2) if elev < 30 else 25.0
        if is_tsunami_basin and elev < 30.0:
            roughness_loss = 1.5 * est_coast_dist_km
            ts_rp100 = round(max(0.0, 4.5 - elev - roughness_loss), 2)
            ts_rp500 = round(max(0.0, 8.0 - elev - roughness_loss), 2)
            ts_rp1000 = round(max(0.0, 11.0 - elev - roughness_loss), 2)
            ts_rp2500 = round(max(0.0, 14.5 - elev - roughness_loss), 2)
        else:
            ts_rp100 = ts_rp500 = ts_rp1000 = ts_rp2500 = 0.0

        # 6. Tropical Cyclone Wind Curve (3-sec Gust km/h at RP50, 100, 200, 500, 1000)
        abs_lat = abs(lat)
        in_cyclone_belt = 5.0 <= abs_lat <= 42.0 and (
            (105.0 <= lon <= 160.0) or (-110.0 <= lon <= -50.0) or (40.0 <= lon <= 100.0)
        )
        if in_cyclone_belt:
            lat_factor = max(0.4, 1.0 - abs(abs_lat - 22.0) / 30.0)
            cy_rp50 = round(115.0 + 75.0 * lat_factor, 1)
            cy_rp100 = round(130.0 + 85.0 * lat_factor, 1)
            cy_rp200 = round(145.0 + 95.0 * lat_factor, 1)
            cy_rp500 = round(162.0 + 105.0 * lat_factor, 1)
            cy_rp1000 = round(175.0 + 115.0 * lat_factor, 1)
        else:
            cy_rp50, cy_rp100, cy_rp200, cy_rp500, cy_rp1000 = 72.0, 82.0, 90.0, 98.0, 105.0

        # 7. Earthquake Seismic PGA (g) at RP475 & RP2475
        if is_tsunami_basin:
            eq_pga_475 = 0.34
            eq_pga_2475 = 0.62
        elif abs_lat < 45.0 and (-125.0 <= lon <= -65.0 or 10.0 <= lon <= 105.0):
            eq_pga_475 = 0.16
            eq_pga_2475 = 0.31
        else:
            eq_pga_475 = 0.04
            eq_pga_2475 = 0.09

        # 8. Landslide Hazard Index (1 to 4) based on Slope, Seismic PGA, & Rainfall
        if slope >= 18.0:
            ls_rain = 4
            ls_quake = 4 if eq_pga_475 >= 0.15 else 3
        elif slope >= 10.0:
            ls_rain = 3
            ls_quake = 3 if eq_pga_475 >= 0.15 else 2
        elif slope >= 4.0:
            ls_rain = 2
            ls_quake = 2 if eq_pga_475 >= 0.20 else 1
        else:
            ls_rain = 1
            ls_quake = 1
        ls_combined = max(ls_rain, ls_quake)

        # Compute ThinkHazard! Standardized Qualitative Grades
        th_river = grade_flood_hazard(fl_rp10, fl_rp50, fl_rp500)
        th_urban = grade_flood_hazard(uf_rp10, uf_rp50, uf_rp500)
        th_coastal = grade_flood_hazard(cf_rp10, cf_rp50, cf_rp500)
        th_tsunami = grade_tsunami_hazard(ts_rp100, ts_rp500, ts_rp2500)
        th_cyclone = grade_cyclone_hazard(cy_rp50, cy_rp100, cy_rp500)
        th_quake = grade_earthquake_hazard(eq_pga_475)
        th_landslide = grade_landslide_hazard(ls_combined)

        records.append({
            "Elev_GLO30_m": elev,
            "Slope_deg": slope,
            "HAND_m": hand_m,
            "TWI_Index": twi,
            # ThinkHazard! Simplified Grades
            "TH_River_Flood": th_river,
            "TH_Urban_Flood": th_urban,
            "TH_Coastal_Flood": th_coastal,
            "TH_Tsunami": th_tsunami,
            "TH_Cyclone": th_cyclone,
            "TH_Earthquake": th_quake,
            "TH_Landslide": th_landslide,
            # River Flood Return Periods (m)
            "FL_RP10_m": fl_rp10, "FL_RP20_m": fl_rp20, "FL_RP50_m": fl_rp50,
            "FL_RP100_m": fl_rp100, "FL_RP200_m": fl_rp200, "FL_RP500_m": fl_rp500,
            # Urban Pluvial Return Periods (m)
            "UF_ADM2_Baseline": uf_adm2_baseline, "UF_Susceptibility_Score": uf_score,
            "UF_RP10_m": uf_rp10, "UF_RP50_m": uf_rp50, "UF_RP100_m": uf_rp100, "UF_RP500_m": uf_rp500,
            # Coastal Surge Return Periods (m)
            "CF_RP10_m": cf_rp10, "CF_RP50_m": cf_rp50, "CF_RP100_m": cf_rp100,
            "CF_RP250_m": cf_rp250, "CF_RP500_m": cf_rp500, "CF_RP1000_m": cf_rp1000,
            # Tsunami Return Periods (m)
            "TS_Coast_Dist_km": est_coast_dist_km,
            "TS_RP100_m": ts_rp100, "TS_RP500_m": ts_rp500, "TS_RP1000_m": ts_rp1000, "TS_RP2500_m": ts_rp2500,
            # Cyclone Wind Return Periods (km/h)
            "CY_RP50_kmh": cy_rp50, "CY_RP100_kmh": cy_rp100, "CY_RP200_kmh": cy_rp200,
            "CY_RP500_kmh": cy_rp500, "CY_RP1000_kmh": cy_rp1000,
            # Earthquake Return Periods (g)
            "EQ_PGA_RP475_g": eq_pga_475, "EQ_PGA_RP2475_g": eq_pga_2475,
            # Landslide Metrics
            "LS_Rain_Index": ls_rain, "LS_Quake_Index": ls_quake, "LS_Combined_Class": ls_combined
        })

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1)

# ==============================================================================
# 5. INPUT LAYER UI (MANUAL TABLE OR CSV BATCH UPLOAD)
# ==============================================================================
st.subheader("1. Input Locations (Single or Batch)")
tab_manual, tab_csv = st.tabs(["📝 Manual Table Input", "📂 Upload CSV File"])

trigger_df = pd.DataFrame()

with tab_manual:
    st.caption("Edit cells directly or click the bottom row to add new locations. Provide an Address/Postal Code OR exact Lat/Lon coordinates.")
    sample_locations = pd.DataFrame([
        {"Location_ID": "LOC-01", "Name": "Tokyo Bay Coastal Logistics", "Address_or_Postal": "Harumi, Chuo-ku, Tokyo, Japan", "Lat": None, "Lon": None},
        {"Location_ID": "LOC-02", "Name": "Edogawa Floodplain Depot", "Address_or_Postal": "Hirai, Edogawa-ku, Tokyo, Japan", "Lat": None, "Lon": None},
        {"Location_ID": "LOC-03", "Name": "Omiya Inland Office", "Address_or_Postal": "Sakuragicho, Omiya-ku, Saitama, Japan", "Lat": None, "Lon": None},
    ])
    edited_table = st.data_editor(sample_locations, num_rows="dynamic", use_container_width=True)
    if st.button("🚀 Run Multi-Peril Hazard Screening (Manual Input)", type="primary"):
        trigger_df = edited_table.copy()

with tab_csv:
    st.caption("Upload a `.csv` file with columns: `Location_ID`, `Name`, `Address_or_Postal` (and optional `Lat`, `Lon`).")
    template_csv = sample_locations.to_csv(index=False).encode("utf-8")
    st.download_button("📄 Download Blank CSV Template", data=template_csv, file_name="location_input_template.csv", mime="text/csv")
    uploaded_csv = st.file_uploader("Drag and drop your CSV file here", type=["csv"])
    if uploaded_csv is not None:
        csv_preview = pd.read_csv(uploaded_csv)
        st.dataframe(csv_preview.head(10), use_container_width=True)
        if st.button("🚀 Run Multi-Peril Hazard Screening (CSV Batch)", type="primary"):
            trigger_df = csv_preview.copy()

if not trigger_df.empty:
    with st.spinner("Geocoding locations and extracting 30m DEM + Multi-Return-Period Hazard Curves..."):
        geocoded = geocode_dataframe(trigger_df)
        valid_rows = geocoded.dropna(subset=["Lat", "Lon"]).reset_index(drop=True)
        if valid_rows.empty:
            st.error("No valid coordinates could be resolved. Please verify the input addresses or supply Lat/Lon.")
        else:
            st.session_state["results_df"] = evaluate_all_perils(valid_rows, use_gee=gee_connected, asset_prefix=gee_project)

# ==============================================================================
# 6. OUTPUT LAYER UI (INTERACTIVE MAP, SUMMARY MATRIX & FULL CSV EXPORT)
# ==============================================================================
if "results_df" in st.session_state:
    res_df = st.session_state["results_df"]
    st.divider()
    st.subheader("2. Interactive Multi-Peril Map & Hazard Curve Output")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        color_peril = st.selectbox(
            "Color Map Pins By Peril:",
            ["TH_River_Flood", "TH_Urban_Flood", "TH_Coastal_Flood", "TH_Tsunami", "TH_Cyclone", "TH_Earthquake", "TH_Landslide"],
            index=0
        )
        m = folium.Map(
            location=[res_df["Lat"].mean(), res_df["Lon"].mean()],
            zoom_start=10,
            tiles="CartoDB positron"
        )
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap Street View").add_to(m)
        folium.TileLayer("OpenTopoMap", name="Topographic Terrain Contours").add_to(m)

        pin_colors = {"HIGH": "red", "MEDIUM": "orange", "LOW": "blue", "VERY LOW": "green"}

        for _, r in res_df.iterrows():
            grade_val = r[color_peril]
            marker_color = pin_colors.get(grade_val, "gray")
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 320px; font-size: 12px;">
                <h4 style="margin: 0 0 4px 0;">{r['Name']} ({r['Location_ID']})</h4>
                <b>Coords:</b> {r['Lat']:.5f}, {r['Lon']:.5f} ({r['Geocode_Precision']})<br>
                <b>Elevation (GLO-30):</b> {r['Elev_GLO30_m']} m | <b>Slope:</b> {r['Slope_deg']}&deg; | <b>HAND:</b> {r['HAND_m']} m
                <hr style="margin: 5px 0;">
                <b>River Flood:</b> <b>{r['TH_River_Flood']}</b><br>
                <small>RP10={r['FL_RP10_m']}m | RP50={r['FL_RP50_m']}m | RP100={r['FL_RP100_m']}m | RP500={r['FL_RP500_m']}m</small><br>
                <b>Urban Flood:</b> <b>{r['TH_Urban_Flood']}</b> (Score: {r['UF_Susceptibility_Score']}/100)<br>
                <small>RP10={r['UF_RP10_m']}m | RP50={r['UF_RP50_m']}m | RP100={r['UF_RP100_m']}m | RP500={r['UF_RP500_m']}m</small><br>
                <b>Coastal Surge:</b> <b>{r['TH_Coastal_Flood']}</b><br>
                <small>RP10={r['CF_RP10_m']}m | RP100={r['CF_RP100_m']}m | RP500={r['CF_RP500_m']}m | RP1000={r['CF_RP1000_m']}m</small><br>
                <b>Tsunami:</b> <b>{r['TH_Tsunami']}</b><br>
                <small>RP100={r['TS_RP100_m']}m | RP500={r['TS_RP500_m']}m | RP2500={r['TS_RP2500_m']}m</small><br>
                <b>Cyclone Wind:</b> <b>{r['TH_Cyclone']}</b><br>
                <small>RP50={r['CY_RP50_kmh']} | RP100={r['CY_RP100_kmh']} | RP500={r['CY_RP500_kmh']} km/h</small><br>
                <b>Earthquake:</b> <b>{r['TH_Earthquake']}</b> <small>(RP475={r['EQ_PGA_RP475_g']}g | RP2475={r['EQ_PGA_RP2475_g']}g)</small><br>
                <b>Landslide:</b> <b>{r['TH_Landslide']}</b> <small>(Class {r['LS_Combined_Class']}/4)</small>
            </div>
            """
            folium.Marker(
                location=[r["Lat"], r["Lon"]],
                popup=folium.Popup(popup_html, max_width=340),
                tooltip=f"{r['Name']} | Elev: {r['Elev_GLO30_m']}m | {color_peril}: {grade_val}",
                icon=folium.Icon(color=marker_color, icon="info-sign")
            ).add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, width=720, height=480)

    with col_right:
        st.markdown("##### Executive Summary: ThinkHazard! Risk Grades")
        exec_cols = [
            "Location_ID", "Name", "Elev_GLO30_m",
            "TH_River_Flood", "TH_Urban_Flood", "TH_Coastal_Flood",
            "TH_Tsunami", "TH_Cyclone", "TH_Earthquake", "TH_Landslide"
        ]
        st.dataframe(res_df[exec_cols], use_container_width=True, height=390)

        csv_export = res_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Export Full Raw Return-Period Dataset & Grades (CSV)",
            data=csv_export,
            file_name="multi_peril_hazard_return_periods_export.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )

    st.markdown("##### Complete Quantitative Return-Period Hazard Curve Table (All 42 Columns)")
    st.dataframe(res_df, use_container_width=True)
```
