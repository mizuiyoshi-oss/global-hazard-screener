import json
import math
import re
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim, ArcGIS

try:
    import ee
    from google.oauth2 import service_account
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

# ==============================================================================
# 1. PAGE CONFIGURATION & SESSION STATE INITIALIZATION
# ==============================================================================
st.set_page_config(
    page_title="Global High-Res Multi-Peril Hazard Screener",
    page_icon="🌍",
    layout="wide"
)

if "result_version" not in st.session_state:
    st.session_state["result_version"] = 0
if "picker_version" not in st.session_state:
    st.session_state["picker_version"] = 0
if "picker_center" not in st.session_state:
    st.session_state["picker_center"] = [35.6812, 139.7671]  # Default: Tokyo
if "picker_zoom" not in st.session_state:
    st.session_state["picker_zoom"] = 5

st.title("🌍 Open-Source High-Resolution Multi-Peril Natural Hazard & Elevation Screener")
st.markdown(
    "**Source-Model Native Data Edition (Zero Proprietary Hazard Transformation):**\n"
    "- **A. Quantitative Source-Model Hazards:** Extracts native outputs and ThinkHazard! classifications for **River Flood (JRC)**, **Tropical Cyclone (STORM)**, **Earthquake (GEM)**, and **Landslide (GFDRR)**.\n"
    "- **B. Terrain-Based Pluvial Screening:** Computes **Topographic Wetness Index (`TWI_Index`)** alongside **HAND**, **Slope**, **Upstream Area (`UPA`)**, and **Elevation** to screen surface-water accumulation potential without inventing synthetic pluvial flood depths.\n"
    "- **C. Contextual Hazard Guidance (Not Site-Modelled):** Displays **UNDRR GAR15 coastal tsunami source heights** and terrain/proximity indicators as screening triggers to consult local authoritative coastal surge and tsunami inundation maps."
)

# ==============================================================================
# 2. GOOGLE EARTH ENGINE INITIALIZATION (AUTO-FIXES MISSING TOML FIELDS)
# ==============================================================================
@st.cache_resource
def initialize_gee():
    if not GEE_AVAILABLE:
        return False, "earthengine-api not installed"
    try:
        if "gee_service_account" in st.secrets:
            sa_raw = st.secrets["gee_service_account"]
            if isinstance(sa_raw, str):
                sa_info = json.loads(sa_raw, strict=False)
            else:
                sa_info = dict(sa_raw)

            sa_info.setdefault("type", "service_account")
            sa_info.setdefault("auth_uri", "https://accounts.google.com/o/oauth2/auth")
            sa_info.setdefault("token_uri", "https://oauth2.googleapis.com/token")
            sa_info.setdefault("auth_provider_x509_cert_url", "https://www.googleapis.com/oauth2/v1/certs")

            if "private_key" in sa_info and isinstance(sa_info["private_key"], str):
                sa_info["private_key"] = sa_info["private_key"].replace("\\n", "\n")

            scopes = getattr(
                ee.oauth,
                "SCOPES",
                [
                    "https://www.googleapis.com/auth/earthengine",
                    "https://www.googleapis.com/auth/cloud-platform",
                ],
            )
            credentials = service_account.Credentials.from_service_account_info(
                sa_info, scopes=scopes
            )
            ee.Initialize(credentials=credentials, project=sa_info.get("project_id"))
            return True, f"Connected to Google Earth Engine Cloud ({sa_info.get('project_id')})"
    except Exception as e:
        return False, f"GEE Cloud Auth Error: {type(e).__name__}: {str(e)}"
    return False, "GEE Not Connected (Add GEE Service Account in Secrets)"

gee_connected, gee_status_msg = initialize_gee()

default_project_id = "my-hazard-screener-2026"
if "gee_service_account" in st.secrets:
    try:
        default_project_id = st.secrets["gee_service_account"].get("project_id", default_project_id)
    except Exception:
        pass

with st.sidebar:
    st.header("⚙️ System Status & Settings")
    if gee_connected:
        st.success(f"✅ {gee_status_msg}")
    else:
        st.warning(f"⚠️ {gee_status_msg}")

    st.markdown("---")
    gee_project = st.text_input(
        "GEE Project Asset Folder",
        value=f"projects/{default_project_id}/assets"
    ).rstrip("/")

    st.markdown("---")
    st.subheader("Methodological Framework")
    st.markdown(
        "**A. Quantitative Source-Model Hazards:**\n"
        "- **River Flood:** JRC GloFAS v2.1 ($90\\text{ m}$, $RP10\\text{--}500$)\n"
        "- **Cyclone:** STORM 10-min Sustained Wind ($\\text{m/s}$, $RP50\\text{--}1000$)\n"
        "- **Seismic:** GEM Global Seismic ($PGA$ at $RP475$)\n"
        "- **Landslide:** World Bank GFDRR ($1\\text{ km}$, Class 1–4)\n\n"
        "**B. Terrain-Based Pluvial Screening:**\n"
        "- **TWI:** $\\ln(a / \\tan\\beta)$ (Wetness Index)\n"
        "- **Elevation & Slope:** Copernicus GLO-30 ($30\\text{ m}$)\n"
        "- **HAND & UPA:** MERIT Hydro v1.0.1 ($90\\text{ m}$)\n\n"
        "**C. Contextual Guidance (Not Site-Modelled):**\n"
        "- **Tsunami Source $H_0$:** UNDRR GAR15 PTHA ($RP100\\text{--}2500$)\n"
        "- **Coastal Surge:** Contextual elevation & proximity triggers only"
    )

# ==============================================================================
# 3. CLASSIFICATION & TERRAIN FUNCTIONS (SOURCE-MODEL NATIVE THRESHOLDS)
# ==============================================================================
def compute_twi(upa_km2, slope_deg, cell_size_m=90.0):
    """
    Computes Topographic Wetness Index: TWI = ln(a / tan(beta))
    where:
      a = specific catchment area (m^2 / m) derived from MERIT Hydro UPA_km2
      beta = local terrain slope in radians from Copernicus GLO-30 Slope_deg
    """
    if pd.isna(upa_km2) or pd.isna(slope_deg):
        return np.nan
    a_specific = max(float(upa_km2) * 1e6 / cell_size_m, cell_size_m)
    beta_rad = math.radians(max(float(slope_deg), 0.0))
    tan_beta = max(math.tan(beta_rad), 0.001)
    return round(math.log(a_specific / tan_beta), 2)

def grade_river_flood_hazard(rp10_m, rp50_m, rp500_m, threshold_m=0.50):
    """ThinkHazard! river flood criteria applied to source-model JRC depths (m)."""
    if pd.isna(rp10_m) or pd.isna(rp50_m) or pd.isna(rp500_m):
        return "N/A"
    if rp10_m >= threshold_m:
        return "HIGH"
    elif rp50_m >= threshold_m:
        return "MEDIUM"
    elif rp500_m > 0.10:
        return "LOW"
    return "VERY LOW"

def grade_cyclone_hazard_ms(rp50_ms, rp100_ms, rp500_ms):
    """
    Grades Tropical Cyclone hazard on source-model STORM 10-minute sustained winds (m/s):
    - HIGH: RP50 >= 33.0 m/s (Typhoon / Hurricane force at 50-yr return period)
    - MEDIUM: RP100 >= 29.0 m/s (Severe Tropical Storm force at 100-yr return period)
    - LOW: RP500 >= 20.0 m/s (Gale / Tropical Storm force at 500-yr return period)
    """
    if pd.isna(rp50_ms) or pd.isna(rp100_ms) or pd.isna(rp500_ms):
        return "N/A"
    if rp50_ms >= 33.0:
        return "HIGH"
    elif rp100_ms >= 29.0:
        return "MEDIUM"
    elif rp500_ms >= 20.0:
        return "LOW"
    return "VERY LOW"

def grade_earthquake_hazard(pga_475_g):
    """Grades Earthquake hazard on source-model GEM RP475 PGA (g)."""
    if pd.isna(pga_475_g):
        return "N/A"
    if pga_475_g >= 0.20:
        return "HIGH"
    elif pga_475_g >= 0.10:
        return "MEDIUM"
    elif pga_475_g >= 0.05:
        return "LOW"
    return "VERY LOW"

def grade_landslide_hazard(combined_class):
    """Grades Landslide hazard on source-model GFDRR Combined Class (1 to 4)."""
    if pd.isna(combined_class):
        return "N/A"
    mapping = {4: "HIGH", 3: "MEDIUM", 2: "LOW", 1: "VERY LOW"}
    return mapping.get(int(round(combined_class)), "VERY LOW")

# ==============================================================================
# 4. GEOCODING & SOURCE-MODEL NATIVE SAMPLING ENGINE
# ==============================================================================
def reverse_geocode_point(lat, lon):
    try:
        arcgis_coder = ArcGIS(timeout=8)
        loc = arcgis_coder.reverse((lat, lon))
        if loc and loc.address:
            return loc.address, "Clicked Map Point (ArcGIS Reverse)"
    except Exception:
        pass
    try:
        nominatim_coder = Nominatim(user_agent="global_multi_peril_hazard_screener_v11", timeout=8)
        loc_osm = nominatim_coder.reverse((lat, lon), language="en")
        if loc_osm and loc_osm.address:
            return loc_osm.address, "Clicked Map Point (OSM Reverse)"
    except Exception:
        pass
    return f"Clicked Coordinate ({lat:.5f}, {lon:.5f})", "Clicked Map Coordinate"

def geocode_dataframe(df):
    arcgis_coder = ArcGIS(timeout=10)
    nominatim_coder = Nominatim(user_agent="global_multi_peril_hazard_screener_v11", timeout=10)

    df = df.copy()
    if "Lat" not in df.columns:
        df["Lat"] = np.nan
    if "Lon" not in df.columns:
        df["Lon"] = np.nan
    if "Matched_Address" not in df.columns:
        df["Matched_Address"] = ""
    if "Geocode_Precision" not in df.columns:
        df["Geocode_Precision"] = ""

    # Filter out completely empty rows from manual input
    valid_input_indices = []
    for idx, row in df.iterrows():
        addr_str = str(row.get("Address_or_Postal", "")).strip()
        has_addr = addr_str != "" and addr_str.lower() not in ("none", "nan")
        lat_val, lon_val = row.get("Lat"), row.get("Lon")
        has_coords = (
            pd.notna(lat_val) and pd.notna(lon_val)
            and str(lat_val).strip() != "" and str(lon_val).strip() != ""
        )
        if has_addr or has_coords:
            valid_input_indices.append(idx)

    df = df.loc[valid_input_indices].reset_index(drop=True)
    if df.empty:
        return df

    progress = st.progress(0, text="Geocoding locations...")
    total = max(1, len(df))
    coord_pattern = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")

    for i, (idx, row) in enumerate(df.iterrows()):
        # Ensure Location_ID is populated if left blank
        loc_id_str = str(row.get("Location_ID", "")).strip()
        if not loc_id_str or loc_id_str.lower() in ("none", "nan"):
            df.at[idx, "Location_ID"] = f"LOC-{i + 1:02d}"

        lat_val = row.get("Lat")
        lon_val = row.get("Lon")
        has_coords = (
            pd.notna(lat_val) and pd.notna(lon_val)
            and str(lat_val).strip() != "" and str(lon_val).strip() != ""
        )

        if has_coords:
            df.at[idx, "Lat"] = float(lat_val)
            df.at[idx, "Lon"] = float(lon_val)
            if not str(row.get("Matched_Address", "")).strip():
                df.at[idx, "Matched_Address"] = "User-Provided Coordinates"
            if not str(row.get("Geocode_Precision", "")).strip():
                df.at[idx, "Geocode_Precision"] = "Exact Coordinate"
        else:
            query = str(row.get("Address_or_Postal", "")).strip()
            if query and query.lower() not in ("none", "nan"):
                coord_match = coord_pattern.match(query)
                if coord_match:
                    df.at[idx, "Lat"] = round(float(coord_match.group(1)), 6)
                    df.at[idx, "Lon"] = round(float(coord_match.group(2)), 6)
                    df.at[idx, "Matched_Address"] = "Parsed from Address Column"
                    df.at[idx, "Geocode_Precision"] = "Exact Coordinate"
                else:
                    try:
                        loc = arcgis_coder.geocode(query)
                        if loc:
                            df.at[idx, "Lat"] = round(float(loc.latitude), 6)
                            df.at[idx, "Lon"] = round(float(loc.longitude), 6)
                            df.at[idx, "Matched_Address"] = loc.address
                            score = loc.raw.get("score", 100)
                            df.at[idx, "Geocode_Precision"] = f"ArcGIS (Score: {score})"
                        else:
                            time.sleep(1.05)
                            loc_osm = nominatim_coder.geocode(query, addressdetails=True, language="en")
                            if loc_osm:
                                df.at[idx, "Lat"] = round(float(loc_osm.latitude), 6)
                                df.at[idx, "Lon"] = round(float(loc_osm.longitude), 6)
                                df.at[idx, "Matched_Address"] = loc_osm.address
                                add_type = loc_osm.raw.get("addresstype", "place")
                                df.at[idx, "Geocode_Precision"] = f"OSM ({add_type})"
                            else:
                                df.at[idx, "Matched_Address"] = "GEOCODE_NOT_FOUND"
                                df.at[idx, "Geocode_Precision"] = "Failed"
                    except Exception as e:
                        df.at[idx, "Matched_Address"] = f"Error: {str(e)[:60]}"
                        df.at[idx, "Geocode_Precision"] = "Error"

        # Ensure Name is populated if left blank
        name_str = str(row.get("Name", "")).strip()
        if not name_str or name_str.lower() in ("none", "nan"):
            matched_str = str(df.at[idx, "Matched_Address"]).strip()
            addr_str = str(row.get("Address_or_Postal", "")).strip()
            if matched_str and matched_str not in ("GEOCODE_NOT_FOUND", "User-Provided Coordinates", "Parsed from Address Column"):
                df.at[idx, "Name"] = matched_str.split(",")[0].strip()
            elif addr_str and addr_str.lower() not in ("none", "nan"):
                df.at[idx, "Name"] = addr_str.split(",")[0].strip()
            else:
                df.at[idx, "Name"] = f"Site {i + 1}"

        progress.progress((i + 1) / total, text=f"Geocoded {i + 1} of {total} locations...")
    progress.empty()
    return df

def fetch_open_meteo_dem_and_slope(lats, lons):
    """Fallback Copernicus GLO-90/GLO-30 API query only for elevation & slope if GEE is offline."""
    elevs, slopes = [], []
    delta = 0.00027
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
                z0 = float(res[j]) if res[j] is not None else np.nan
                z_n = float(res[j+1]) if j+1 < len(res) and res[j+1] is not None else z0
                z_e = float(res[j+2]) if j+2 < len(res) and res[j+2] is not None else z0
                if pd.notna(z0):
                    dz_dx = (z_e - z0) / 30.0
                    dz_dy = (z_n - z0) / 30.0
                    slope_deg = math.degrees(math.atan(math.sqrt(dz_dx**2 + dz_dy**2)))
                    elevs.append(round(z0, 2))
                    slopes.append(round(slope_deg, 2))
                else:
                    elevs.append(np.nan)
                    slopes.append(np.nan)
        except Exception:
            for _ in batch_lats:
                elevs.append(np.nan)
                slopes.append(np.nan)
    return elevs, slopes

def sample_gee_cloud_rasters(df, asset_prefix):
    gee_dict = {int(idx): {} for idx in df.index}
    diagnostics = []

    features = []
    for idx, row in df.iterrows():
        pt = ee.Geometry.Point([float(row["Lon"]), float(row["Lat"])])
        features.append(ee.Feature(pt, {"row_idx": int(idx)}))
    fc = ee.FeatureCollection(features)

    # --- GROUP A: Public Catalog (Copernicus GLO-30 DEM + MERIT Hydro + JRC Flood v2.1) ---
    try:
        dem_img = (
            ee.ImageCollection("COPERNICUS/DEM/GLO30")
            .select("DEM")
            .mosaic()
            .setDefaultProjection("EPSG:4326", None, 30)
            .unmask(0)
        )
        slope_img = ee.Terrain.slope(dem_img).rename("Slope_deg")
        merit = ee.Image("MERIT/Hydro/v1_0_1").select(["hnd", "upa"], ["HAND_m", "UPA_km2"]).unmask(0)

        jrc_mosaic = (
            ee.ImageCollection("JRC/CEMS_GLOFAS/FloodHazard/v2_1")
            .select(
                ["RP10_depth", "RP20_depth", "RP50_depth", "RP100_depth", "RP200_depth", "RP500_depth"],
                ["FL_RP10_m", "FL_RP20_m", "FL_RP50_m", "FL_RP100_m", "FL_RP200_m", "FL_RP500_m"]
            )
            .mosaic()
            .unmask(0)
        )

        public_stack = ee.Image.cat([dem_img.rename("Elev_GLO30_m"), slope_img, merit, jrc_mosaic])
        pub_sampled = public_stack.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=30).getInfo()
        for feat in pub_sampled.get("features", []):
            props = feat.get("properties", {})
            r_idx = props.get("row_idx")
            if r_idx in gee_dict:
                gee_dict[r_idx].update(props)
        diagnostics.append("✅ Public GEE Catalog (Copernicus GLO-30 DEM, MERIT Hydro HAND/UPA, JRC Flood RP10–RP500): Loaded")
    except Exception as e:
        diagnostics.append(f"⚠️ Public GEE Catalog error: {str(e)[:120]}")

    # --- GROUP B: Private Uploaded Source-Model Rasters (Native Units, No Multipliers) ---
    private_specs = [
        ("gem_pga_475", "EQ_PGA_RP475_g", 1.0, 0.0),
        ("gfdrr_landslide_th", "LS_Combined_Class", 1.0, 1.0),
        ("gfdrr_landslide_rain", "LS_Rain_Index", 1.0, 0.0),
        ("gfdrr_landslide_quake", "LS_Quake_Index", 1.0, 0.0),
        ("storm_wind_rp50", "CY_RP50_ms", 1.0, 0.0),
        ("storm_wind_rp100", "CY_RP100_ms", 1.0, 0.0),
        ("storm_wind_rp200", "CY_RP200_ms", 1.0, 0.0),
        ("storm_wind_rp500", "CY_RP500_ms", 1.0, 0.0),
        ("storm_wind_rp1000", "CY_RP1000_ms", 1.0, 0.0),
    ]

    try:
        priv_imgs = []
        for asset_name, col_name, mult, default_val in private_specs:
            raw_img = ee.Image(f"{asset_prefix}/{asset_name}").select([0]).unmask(default_val)
            if col_name == "LS_Combined_Class":
                clean_img = raw_img.where(raw_img.gt(4).Or(raw_img.lt(1)), 1).rename(col_name)
            else:
                clean_img = raw_img.where(raw_img.lt(0), 0.0).multiply(mult).rename(col_name)
            priv_imgs.append(clean_img)

        priv_stack = ee.Image.cat(priv_imgs)
        priv_sampled = priv_stack.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=90).getInfo()
        for feat in priv_sampled.get("features", []):
            props = feat.get("properties", {})
            r_idx = props.get("row_idx")
            if r_idx in gee_dict:
                gee_dict[r_idx].update(props)
        diagnostics.append(f"✅ All 9 Private Source-Model Raster Assets in `{asset_prefix}` (Native Units): Loaded")
    except Exception as batch_err:
        diagnostics.append(f"ℹ️ Batch private raster query fell back to individual asset check ({str(batch_err)[:80]})")
        for asset_name, col_name, mult, default_val in private_specs:
            try:
                raw_img = ee.Image(f"{asset_prefix}/{asset_name}").select([0]).unmask(default_val)
                if col_name == "LS_Combined_Class":
                    img = raw_img.where(raw_img.gt(4).Or(raw_img.lt(1)), 1).rename(col_name)
                else:
                    img = raw_img.where(raw_img.lt(0), 0.0).multiply(mult).rename(col_name)
                single_res = img.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=90).getInfo()
                for feat in single_res.get("features", []):
                    props = feat.get("properties", {})
                    r_idx = props.get("row_idx")
                    if r_idx in gee_dict and "first" in props:
                        gee_dict[r_idx][col_name] = props["first"]
                    elif r_idx in gee_dict and col_name in props:
                        gee_dict[r_idx][col_name] = props[col_name]
                diagnostics.append(f"  ✅ `{asset_name}` -> `{col_name}`: Loaded")
            except Exception as single_err:
                diagnostics.append(f"  ❌ `{asset_prefix}/{asset_name}` failed: {str(single_err)[:90]}")

    # --- GROUP C: Contextual GAR15 Tsunami Source Table (Native Shoreline H0, Site Inundation Not Modelled) ---
    try:
        tsunami_fc = ee.FeatureCollection(f"{asset_prefix}/gar15_tsunami_ptha")
        spatial_filter = ee.Filter.withinDistance(distance=80000, leftField=".geo", rightField=".geo", maxError=50)
        nearest_join = ee.Join.saveBest(matchKey="nearest_ptha", measureKey="coast_dist_m")
        joined_fc = nearest_join.apply(fc, tsunami_fc, spatial_filter)
        for feat in joined_fc.getInfo().get("features", []):
            props = feat.get("properties", {})
            r_idx = props.get("row_idx")
            ptha_match = props.get("nearest_ptha")
            if ptha_match and "properties" in ptha_match and r_idx in gee_dict:
                p_props = ptha_match["properties"]
                raw_dist_km = float(p_props.get("coast_dist_m", 0.0)) / 1000.0
                gee_dict[r_idx]["TS_Nearest_Point_Dist_km"] = round(raw_dist_km, 2)
                gee_dict[r_idx]["TS_Shore_H0_RP100_m"] = round(float(p_props.get("ari100", 0.0) or 0.0), 2)
                gee_dict[r_idx]["TS_Shore_H0_RP500_m"] = round(float(p_props.get("ari500", 0.0) or 0.0), 2)
                gee_dict[r_idx]["TS_Shore_H0_RP1000_m"] = round(float(p_props.get("ari1000", 0.0) or 0.0), 2)
                gee_dict[r_idx]["TS_Shore_H0_RP2500_m"] = round(float(p_props.get("ari2500", 0.0) or 0.0), 2)
        diagnostics.append(f"✅ Contextual Tsunami Source Table (`{asset_prefix}/gar15_tsunami_ptha` Native H0): Loaded")
    except Exception as ts_err:
        diagnostics.append(f"❌ Tsunami PTHA Table (`{asset_prefix}/gar15_tsunami_ptha`) failed: {str(ts_err)[:100]}")

    return gee_dict, diagnostics

def evaluate_all_perils(df, use_gee=False, asset_prefix=""):
    df = df.copy()
    lats = df["Lat"].astype(float).tolist()
    lons = df["Lon"].astype(float).tolist()

    gee_results = {}
    diagnostics = []
    if use_gee:
        gee_results, diagnostics = sample_gee_cloud_rasters(df, asset_prefix)

    om_elevs, om_slopes = fetch_open_meteo_dem_and_slope(lats, lons)

    records = []
    for idx_pos, (r_idx, row) in enumerate(df.iterrows()):
        g_props = gee_results.get(int(r_idx), {})

        # 1. Category B: Terrain-Based Pluvial Screening Metrics (GLO-30 + MERIT Hydro + TWI)
        elev = g_props.get("Elev_GLO30_m")
        elev = round(float(elev), 2) if elev is not None else om_elevs[idx_pos]

        slope = g_props.get("Slope_deg")
        slope = round(float(slope), 2) if slope is not None else om_slopes[idx_pos]

        hand_raw = g_props.get("HAND_m")
        hand_m = round(float(hand_raw), 2) if hand_raw is not None else np.nan

        upa_raw = g_props.get("UPA_km2")
        upa_km2 = round(float(upa_raw), 2) if upa_raw is not None else np.nan

        twi_index = compute_twi(upa_km2, slope)

        # 2. Category A1: River Flood Depth Curve (m) - JRC GloFAS v2.1 Source Model
        if "FL_RP100_m" in g_props and g_props["FL_RP100_m"] is not None:
            fl_rp10 = round(float(g_props.get("FL_RP10_m", 0.0) or 0.0), 2)
            fl_rp20 = round(float(g_props.get("FL_RP20_m", 0.0) or 0.0), 2)
            fl_rp50 = round(float(g_props.get("FL_RP50_m", 0.0) or 0.0), 2)
            fl_rp100 = round(float(g_props.get("FL_RP100_m", 0.0) or 0.0), 2)
            fl_rp200 = round(float(g_props.get("FL_RP200_m", 0.0) or 0.0), 2)
            fl_rp500 = round(float(g_props.get("FL_RP500_m", 0.0) or 0.0), 2)
        else:
            fl_rp10 = fl_rp20 = fl_rp50 = fl_rp100 = fl_rp200 = fl_rp500 = np.nan

        # 3. Category A2: Tropical Cyclone Wind Curve (m/s) - STORM Source Model (10-min Sustained)
        if "CY_RP100_ms" in g_props and g_props["CY_RP100_ms"] is not None:
            cy_rp50 = round(float(g_props.get("CY_RP50_ms", 0.0) or 0.0), 1)
            cy_rp100 = round(float(g_props.get("CY_RP100_ms", 0.0) or 0.0), 1)
            cy_rp200 = round(float(g_props.get("CY_RP200_ms", 0.0) or 0.0), 1)
            cy_rp500 = round(float(g_props.get("CY_RP500_ms", 0.0) or 0.0), 1)
            cy_rp1000 = round(float(g_props.get("CY_RP1000_ms", 0.0) or 0.0), 1)
        else:
            cy_rp50 = cy_rp100 = cy_rp200 = cy_rp500 = cy_rp1000 = np.nan

        # 4. Category A3: Earthquake Seismic PGA (g) - GEM Source Model (RP475)
        if "EQ_PGA_RP475_g" in g_props and g_props["EQ_PGA_RP475_g"] is not None:
            eq_pga_475 = round(float(g_props.get("EQ_PGA_RP475_g", 0.0) or 0.0), 3)
        else:
            eq_pga_475 = np.nan

        # 5. Category A4: Landslide Hazard Index (Class 1 to 4) - World Bank GFDRR Source Model
        if "LS_Combined_Class" in g_props and g_props["LS_Combined_Class"] is not None:
            raw_ls = int(round(float(g_props.get("LS_Combined_Class", 1) or 1)))
            ls_combined = raw_ls if 1 <= raw_ls <= 4 else 1
            raw_rain = float(g_props.get("LS_Rain_Index", 0.0) or 0.0)
            raw_quake = float(g_props.get("LS_Quake_Index", 0.0) or 0.0)
            ls_rain = round(raw_rain if 0.0 <= raw_rain <= 10.0 else 0.0, 5)
            ls_quake = round(raw_quake if 0.0 <= raw_quake <= 10.0 else 0.0, 5)
        else:
            ls_combined = np.nan
            ls_rain = np.nan
            ls_quake = np.nan

        # 6. Category C: Contextual UNDRR GAR15 Tsunami Source Height (m) & Distance (km)
        ts_dist_km = g_props.get("TS_Nearest_Point_Dist_km", np.nan)
        ts_h0_100 = g_props.get("TS_Shore_H0_RP100_m", 0.0 if pd.notna(ts_dist_km) else np.nan)
        ts_h0_500 = g_props.get("TS_Shore_H0_RP500_m", 0.0 if pd.notna(ts_dist_km) else np.nan)
        ts_h0_1000 = g_props.get("TS_Shore_H0_RP1000_m", 0.0 if pd.notna(ts_dist_km) else np.nan)
        ts_h0_2500 = g_props.get("TS_Shore_H0_RP2500_m", 0.0 if pd.notna(ts_dist_km) else np.nan)

        # 7. Qualitative Grades (Strictly for the 4 Quantitative Source-Model Hazards)
        th_river = grade_river_flood_hazard(fl_rp10, fl_rp50, fl_rp500)
        th_cyclone = grade_cyclone_hazard_ms(cy_rp50, cy_rp100, cy_rp500)
        th_quake = grade_earthquake_hazard(eq_pga_475)
        th_landslide = grade_landslide_hazard(ls_combined)

        records.append({
            # --- Category B: Terrain-Based Pluvial Screening Metrics ---
            "Elev_GLO30_m": elev,
            "TWI_Index": twi_index,
            "HAND_m": hand_m,
            "Slope_deg": slope,
            "UPA_km2": upa_km2,
            # --- Category C: Contextual Tsunami Source Information (Not Site Inundation) ---
            "TS_Nearest_Point_Dist_km": ts_dist_km,
            "TS_Shore_H0_RP100_m": ts_h0_100,
            "TS_Shore_H0_RP500_m": ts_h0_500,
            "TS_Shore_H0_RP1000_m": ts_h0_1000,
            "TS_Shore_H0_RP2500_m": ts_h0_2500,
            # --- Category A: 4 Quantitative Source-Model Peril Grades ---
            "TH_River_Flood": th_river,
            "TH_Cyclone": th_cyclone,
            "TH_Earthquake": th_quake,
            "TH_Landslide": th_landslide,
            # --- Category A: Source-Model Native Return-Period Curves ---
            "FL_RP10_m": fl_rp10, "FL_RP20_m": fl_rp20, "FL_RP50_m": fl_rp50,
            "FL_RP100_m": fl_rp100, "FL_RP200_m": fl_rp200, "FL_RP500_m": fl_rp500,
            "CY_RP50_ms": cy_rp50, "CY_RP100_ms": cy_rp100, "CY_RP200_ms": cy_rp200,
            "CY_RP500_ms": cy_rp500, "CY_RP1000_ms": cy_rp1000,
            "EQ_PGA_RP475_g": eq_pga_475,
            "LS_Rain_Index": ls_rain, "LS_Quake_Index": ls_quake, "LS_Combined_Class": ls_combined,
        })

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1), diagnostics

def run_single_point_screening(lat, lon, custom_name=""):
    with st.spinner(f"Identifying location ({lat:.6f}, {lon:.6f}) and sampling Google Earth Engine Source-Model Rasters..."):
        matched_addr, precision_label = reverse_geocode_point(lat, lon)
        site_label = custom_name.strip() if custom_name.strip() else matched_addr.split(",")[0].strip()
        if not site_label:
            site_label = f"Point ({lat:.4f}, {lon:.4f})"

        clicked_df = pd.DataFrame([{
            "Location_ID": "CLICK-01",
            "Name": site_label,
            "Address_or_Postal": matched_addr,
            "Lat": lat,
            "Lon": lon,
            "Matched_Address": matched_addr,
            "Geocode_Precision": precision_label
        }])
        res_df, diag_logs = evaluate_all_perils(clicked_df, use_gee=gee_connected, asset_prefix=gee_project)
        st.session_state["results_df"] = res_df
        st.session_state["diag_logs"] = diag_logs
        st.session_state["result_version"] += 1

# ==============================================================================
# 5. INPUT LAYER UI (MANUAL TABLE, CSV BATCH, AND INTERACTIVE MAP CLICK)
# ==============================================================================
st.subheader("1. Input Locations (Single or Batch)")
tab_manual, tab_csv, tab_map = st.tabs([
    "📝 Manual Table Input",
    "📂 Upload CSV File",
    "🗺️ Interactive Map Click"
])

trigger_df = pd.DataFrame()

empty_locations_schema = pd.DataFrame({
    "Location_ID": pd.Series(dtype="str"),
    "Name": pd.Series(dtype="str"),
    "Address_or_Postal": pd.Series(dtype="str"),
    "Lat": pd.Series(dtype="float"),
    "Lon": pd.Series(dtype="float"),
})

with tab_manual:
    st.caption("Click the `+` row to add locations. Provide an Address/Postal Code OR exact Lat/Lon coordinates (`Location_ID` and `Name` will auto-fill if left blank).")
    edited_table = st.data_editor(
        empty_locations_schema,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Location_ID": st.column_config.TextColumn("Location_ID", help="Optional reference ID (auto-filled as LOC-01 if blank)"),
            "Name": st.column_config.TextColumn("Name", help="Optional site name (auto-filled if blank)"),
            "Address_or_Postal": st.column_config.TextColumn("Address_or_Postal", help="Street address, postal code, or 'Lat, Lon'"),
            "Lat": st.column_config.NumberColumn("Lat", format="%.6f"),
            "Lon": st.column_config.NumberColumn("Lon", format="%.6f"),
        }
    )
    if st.button("🚀 Run Multi-Peril Hazard Screening (Manual Input)", type="primary"):
        trigger_df = edited_table.copy()

with tab_csv:
    st.caption("Upload a `.csv` file with columns: `Location_ID`, `Name`, `Address_or_Postal` (and optional `Lat`, `Lon`).")
    template_csv = empty_locations_schema.to_csv(index=False).encode("utf-8")
    st.download_button("📄 Download Blank CSV Template", data=template_csv, file_name="location_input_template.csv", mime="text/csv")
    uploaded_csv = st.file_uploader("Drag and drop your CSV file here", type=["csv"])
    if uploaded_csv is not None:
        csv_preview = pd.read_csv(uploaded_csv)
        st.dataframe(csv_preview.head(10))
        if st.button("🚀 Run Multi-Peril Hazard Screening (CSV Batch)", type="primary"):
            trigger_df = csv_preview.copy()

with tab_map:
    st.caption("Click any location on the map below (or search a place) to automatically extract coordinates, reverse-geocode the address, and refresh Section 2.")

    col_search, col_btn_zoom, col_custom_name = st.columns([3, 1, 2])
    with col_search:
        zoom_query = st.text_input(
            "🔎 Search a city, address, or landmark to move the map",
            placeholder="e.g., Taylor, TX or Kochi, Japan or Jurong Island, Singapore",
            label_visibility="collapsed"
        )
    with col_custom_name:
        custom_click_name = st.text_input(
            "Optional custom site name",
            value="",
            placeholder="Optional custom site name (auto-named if blank)",
            label_visibility="collapsed"
        )
    with col_btn_zoom:
        if st.button("🔎 Zoom & Screen") and zoom_query.strip():
            try:
                loc_zoom = ArcGIS(timeout=10).geocode(zoom_query.strip())
                if loc_zoom:
                    z_lat = round(float(loc_zoom.latitude), 6)
                    z_lon = round(float(loc_zoom.longitude), 6)
                    st.session_state["picker_center"] = [z_lat, z_lon]
                    st.session_state["picker_zoom"] = 14
                    st.session_state["last_clicked_point"] = (z_lat, z_lon)
                    st.session_state["last_evaluated_click"] = (z_lat, z_lon)
                    st.session_state["picker_version"] += 1
                    run_single_point_screening(z_lat, z_lon, custom_click_name)
                    st.rerun()
            except Exception as e:
                st.warning(f"Search error: {e}")

    picker_m = folium.Map(
        location=st.session_state["picker_center"],
        zoom_start=st.session_state["picker_zoom"],
        tiles="OpenStreetMap"
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri World Imagery (Satellite)",
        overlay=False,
        control=True
    ).add_to(picker_m)
    folium.TileLayer("OpenTopoMap", name="Topographic Terrain Contours").add_to(picker_m)
    folium.LatLngPopup().add_to(picker_m)

    if "last_clicked_point" in st.session_state:
        prev_lat, prev_lon = st.session_state["last_clicked_point"]
        folium.Marker(
            location=[prev_lat, prev_lon],
            tooltip=f"Selected: {prev_lat:.6f}, {prev_lon:.6f}",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(picker_m)

    folium.LayerControl().add_to(picker_m)

    map_click_data = st_folium(
        picker_m,
        width=920,
        height=430,
        returned_objects=["last_clicked"],
        key=f"interactive_picker_map_{st.session_state['picker_version']}"
    )

    if map_click_data and map_click_data.get("last_clicked"):
        clicked_lat = round(float(map_click_data["last_clicked"]["lat"]), 6)
        clicked_lon = round(float(map_click_data["last_clicked"]["lng"]), 6)
        clicked_pair = (clicked_lat, clicked_lon)
        st.session_state["last_clicked_point"] = clicked_pair
        st.session_state["picker_center"] = [clicked_lat, clicked_lon]

        if st.session_state.get("last_evaluated_click") != clicked_pair:
            st.session_state["last_evaluated_click"] = clicked_pair
            run_single_point_screening(clicked_lat, clicked_lon, custom_click_name)
            st.rerun()

    if "last_clicked_point" in st.session_state:
        sel_lat, sel_lon = st.session_state["last_clicked_point"]
        col_info, col_refresh = st.columns([3, 1])
        with col_info:
            st.info(f"📌 **Selected Map Coordinate:** `{sel_lat:.6f}, {sel_lon:.6f}`")
        with col_refresh:
            if st.button("🔄 Run / Refresh Screening for Selected Point", type="primary"):
                st.session_state["last_evaluated_click"] = (sel_lat, sel_lon)
                run_single_point_screening(sel_lat, sel_lon, custom_click_name)
                st.rerun()

# Execute batch/manual trigger if clicked from Tab 1 or Tab 2
if not trigger_df.empty:
    with st.spinner("Geocoding locations and sampling Google Earth Engine Source-Model Rasters..."):
        geocoded = geocode_dataframe(trigger_df)
        valid_rows = geocoded.dropna(subset=["Lat", "Lon"]).reset_index(drop=True)
        if valid_rows.empty:
            st.error("No valid coordinates could be resolved. Please enter at least one valid address or Lat/Lon coordinate.")
        else:
            res_df, diag_logs = evaluate_all_perils(valid_rows, use_gee=gee_connected, asset_prefix=gee_project)
            st.session_state["results_df"] = res_df
            st.session_state["diag_logs"] = diag_logs
            st.session_state["result_version"] += 1
            st.rerun()

# ==============================================================================
# 6. OUTPUT LAYER UI (CATEGORIES A, B, AND C)
# ==============================================================================
if "results_df" in st.session_state:
    res_df = st.session_state["results_df"]
    diag_logs = st.session_state.get("diag_logs", [])
    ver = st.session_state.get("result_version", 0)

    st.divider()
    hdr_left, hdr_mid, hdr_right = st.columns([3, 1, 1])
    with hdr_left:
        st.subheader("2. Interactive Multi-Peril Map & Source-Model Native Output")
    with hdr_mid:
        if st.button("🔄 Refresh Output View"):
            st.session_state["result_version"] += 1
            st.rerun()
    with hdr_right:
        if st.button("🗑️ Clear Results"):
            st.session_state.pop("results_df", None)
            st.session_state["result_version"] += 1
            st.rerun()

    if diag_logs:
        with st.expander("🔍 GEE Cloud Layer Diagnostics (Click to inspect asset status)", expanded=False):
            for msg in diag_logs:
                st.write(msg)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        color_peril = st.selectbox(
            "Color Map Pins By Quantitative Source-Model Peril Grade:",
            ["TH_River_Flood", "TH_Cyclone", "TH_Earthquake", "TH_Landslide"],
            index=0
        )
        default_zoom = 13 if len(res_df) == 1 else 3
        m = folium.Map(
            location=[float(res_df["Lat"].mean()), float(res_df["Lon"].mean())],
            zoom_start=default_zoom,
            tiles="OpenStreetMap"
        )
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Esri World Imagery (Satellite)",
            overlay=False,
            control=True
        ).add_to(m)
        folium.TileLayer("OpenTopoMap", name="Topographic Terrain Contours").add_to(m)

        pin_colors = {"HIGH": "red", "MEDIUM": "orange", "LOW": "blue", "VERY LOW": "green", "N/A": "gray"}

        for _, r in res_df.iterrows():
            grade_val = r[color_peril]
            marker_color = pin_colors.get(grade_val, "gray")
            ts_dist_str = f"{r['TS_Nearest_Point_Dist_km']} km" if pd.notna(r["TS_Nearest_Point_Dist_km"]) else ">80 km (Inland)"
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 350px; font-size: 12px;">
                <h4 style="margin: 0 0 4px 0;">{r['Name']} ({r['Location_ID']})</h4>
                <b>Coords:</b> {r['Lat']:.5f}, {r['Lon']:.5f} ({r['Geocode_Precision']})<br>
                <hr style="margin: 5px 0;">
                <b><u>A. Quantitative Source-Model Hazards</u></b><br>
                <b>River Flood (JRC):</b> <b>{r['TH_River_Flood']}</b><br>
                <small>RP10={r['FL_RP10_m']}m | RP50={r['FL_RP50_m']}m | RP100={r['FL_RP100_m']}m | RP500={r['FL_RP500_m']}m</small><br>
                <b>Cyclone 10-min Wind (STORM):</b> <b>{r['TH_Cyclone']}</b><br>
                <small>RP50={r['CY_RP50_ms']} | RP100={r['CY_RP100_ms']} | RP500={r['CY_RP500_ms']} | RP1000={r['CY_RP1000_ms']} m/s</small><br>
                <b>Earthquake (GEM):</b> <b>{r['TH_Earthquake']}</b> <small>(RP475 PGA = {r['EQ_PGA_RP475_g']} g)</small><br>
                <b>Landslide (GFDRR):</b> <b>{r['TH_Landslide']}</b> <small>(Class {r['LS_Combined_Class']}/4)</small>
                <hr style="margin: 5px 0;">
                <b><u>B. Terrain-Based Pluvial Screening</u></b><br>
                <b>TWI:</b> {r['TWI_Index']} | <b>HAND:</b> {r['HAND_m']} m | <b>Slope:</b> {r['Slope_deg']}&deg;<br>
                <b>Elevation (GLO-30):</b> {r['Elev_GLO30_m']} m | <b>UPA:</b> {r['UPA_km2']} km&sup2;
                <hr style="margin: 5px 0;">
                <b><u>C. Contextual Hazard Guidance (Not Site-Modelled)</u></b><br>
                <b>GAR15 Tsunami Nearest Point:</b> {ts_dist_str}<br>
                <small>Coastal Source H0: RP100={r['TS_Shore_H0_RP100_m']}m | RP500={r['TS_Shore_H0_RP500_m']}m | RP2500={r['TS_Shore_H0_RP2500_m']}m</small>
            </div>
            """
            folium.Marker(
                location=[float(r["Lat"]), float(r["Lon"])],
                popup=folium.Popup(popup_html, max_width=370),
                tooltip=f"{r['Name']} | Elev: {r['Elev_GLO30_m']}m | TWI: {r['TWI_Index']} | {color_peril}: {grade_val}",
                icon=folium.Icon(color=marker_color, icon="info-sign")
            ).add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(
            m,
            width=720,
            height=480,
            returned_objects=[],
            key=f"results_output_map_v{ver}_{color_peril}"
        )

    with col_right:
        st.markdown("##### Executive Summary: Source-Model Hazards, Terrain Screening & Contextual Guidance")
        exec_cols = [
            "Location_ID", "Name",
            "TH_River_Flood", "TH_Cyclone", "TH_Earthquake", "TH_Landslide",
            "TWI_Index", "HAND_m", "Slope_deg", "UPA_km2", "Elev_GLO30_m",
            "TS_Nearest_Point_Dist_km", "TS_Shore_H0_RP500_m", "TS_Shore_H0_RP2500_m"
        ]
        st.dataframe(res_df[exec_cols], height=390, key=f"exec_df_v{ver}")

        csv_export = res_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Export Source-Model Native Dataset & Terrain Metrics (CSV)",
            data=csv_export,
            file_name="multi_peril_source_model_native_export.csv",
            mime="text/csv",
            type="primary",
            key=f"dl_btn_v{ver}"
        )

    st.markdown("##### Complete Source-Model Native & Terrain Screening Table (All 32 Columns)")
    st.dataframe(res_df, key=f"full_df_v{ver}")
