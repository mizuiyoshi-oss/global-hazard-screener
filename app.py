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
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Global High-Res Multi-Peril Hazard Screener",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Open-Source High-Resolution Multi-Peril Natural Hazard & Elevation Screener")
st.markdown(
    "Coordinate-level screening for **Elevation ($30\\text{ m}$ Copernicus DEM)** and **7 Natural Perils** "
    "(*River Flood, Urban Flood, Coastal Flood, Tsunami, Tropical Cyclone, Earthquake, and Landslide*). "
    "Outputs both **full return-period hazard curves** and **World Bank GFDRR ThinkHazard! classifications**."
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

            # Inject required OAuth2 fields if omitted from Streamlit Secrets TOML
            sa_info.setdefault("type", "service_account")
            sa_info.setdefault("auth_uri", "https://accounts.google.com/o/oauth2/auth")
            sa_info.setdefault("token_uri", "https://oauth2.googleapis.com/token")
            sa_info.setdefault("auth_provider_x509_cert_url", "https://www.googleapis.com/oauth2/v1/certs")

            # Normalize escaped newlines in private_key
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
    return False, "Running in Instant API + Terrain Proxy Mode (Add GEE Service Account in Secrets)"

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
    st.subheader("Methodology Reference")
    st.markdown(
        "- **Elevation:** Copernicus GLO-30 ($30\\text{ m}$)\n"
        "- **River Flood:** JRC GloFAS v2.1 ($90\\text{ m}$, $RP10\\text{--}500$)\n"
        "- **Coastal Surge:** WRI Aqueduct v2 ($RP10\\text{--}1000$)\n"
        "- **Urban Flood:** $30\\text{ m}/90\\text{ m}$ HAND + TWI Pluvial Model\n"
        "- **Tsunami:** UNDRR GAR15 PTHA + $30\\text{ m}$ DEM Attenuation\n"
        "- **Cyclone:** STORM Constant Baseline ($RP50\\text{--}1000$)\n"
        "- **Seismic:** GEM Global Seismic Hazard ($PGA$ at $RP475$)\n"
        "- **Landslide:** World Bank GFDRR ($1\\text{ km}$)"
    )

# ==============================================================================
# 3. THINKHAZARD! CLASSIFICATION FUNCTIONS
# ==============================================================================
def grade_flood_hazard(rp10_m, rp50_m, rp500_m, threshold_m=0.50):
    if rp10_m >= threshold_m:
        return "HIGH"
    elif rp50_m >= threshold_m:
        return "MEDIUM"
    elif rp500_m > 0.10:
        return "LOW"
    return "VERY LOW"

def grade_tsunami_hazard(rp100_m, rp500_m, rp2500_m):
    if rp100_m >= 0.50:
        return "HIGH"
    elif rp500_m >= 0.50:
        return "MEDIUM"
    elif rp2500_m > 0.10:
        return "LOW"
    return "VERY LOW"

def grade_cyclone_hazard(rp50_kmh, rp100_kmh, rp500_kmh):
    if rp50_kmh >= 140.0:
        return "HIGH"
    elif rp100_kmh >= 119.0:
        return "MEDIUM"
    elif rp500_kmh >= 90.0:
        return "LOW"
    return "VERY LOW"

def grade_earthquake_hazard(pga_475_g):
    if pga_475_g >= 0.20:
        return "HIGH"
    elif pga_475_g >= 0.10:
        return "MEDIUM"
    elif pga_475_g >= 0.05:
        return "LOW"
    return "VERY LOW"

def grade_landslide_hazard(combined_class):
    mapping = {4: "HIGH", 3: "MEDIUM", 2: "LOW", 1: "VERY LOW"}
    return mapping.get(int(round(combined_class)), "VERY LOW")

# ==============================================================================
# 4. GEOCODING & FAULT-TOLERANT MULTI-PERIL GEE ENGINE
# ==============================================================================
def geocode_dataframe(df):
    arcgis_coder = ArcGIS(timeout=10)
    nominatim_coder = Nominatim(user_agent="global_multi_peril_hazard_screener_v5", timeout=10)

    df = df.copy()
    if "Lat" not in df.columns:
        df["Lat"] = np.nan
    if "Lon" not in df.columns:
        df["Lon"] = np.nan
    if "Matched_Address" not in df.columns:
        df["Matched_Address"] = ""
    if "Geocode_Precision" not in df.columns:
        df["Geocode_Precision"] = ""

    progress = st.progress(0, text="Geocoding locations...")
    total = max(1, len(df))
    coord_pattern = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")

    for i, (idx, row) in enumerate(df.iterrows()):
        lat_val = row.get("Lat")
        lon_val = row.get("Lon")
        has_coords = (
            pd.notna(lat_val) and pd.notna(lon_val)
            and str(lat_val).strip() != "" and str(lon_val).strip() != ""
        )

        if has_coords:
            df.at[idx, "Lat"] = float(lat_val)
            df.at[idx, "Lon"] = float(lon_val)
            df.at[idx, "Matched_Address"] = "User-Provided Coordinates"
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
        progress.progress((i + 1) / total, text=f"Geocoded {i + 1} of {total} locations...")
    progress.empty()
    return df

def fetch_open_meteo_dem_and_slope(lats, lons):
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
    gee_dict = {int(idx): {} for idx in df.index}
    diagnostics = []

    features = []
    for idx, row in df.iterrows():
        pt = ee.Geometry.Point([float(row["Lon"]), float(row["Lat"])])
        features.append(ee.Feature(pt, {"row_idx": int(idx)}))
    fc = ee.FeatureCollection(features)

    # --- GROUP A: Public Catalog (DEM + MERIT Hydro + JRC Flood v2.1) ---
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
        diagnostics.append("✅ Public GEE Catalog (Copernicus DEM, MERIT HAND, JRC Flood RP10–RP500): Loaded")
    except Exception as e:
        diagnostics.append(f"⚠️ Public GEE Catalog error: {str(e)[:120]}")

    # --- GROUP B: Private Uploaded Raster Assets ---
    gust_factor = 4.428
    private_specs = [
        ("gem_pga_475", "EQ_PGA_RP475_g", 1.0, 0.0),
        ("gfdrr_landslide_th", "LS_Combined_Class", 1.0, 1.0),
        ("gfdrr_landslide_rain", "LS_Rain_Index", 1.0, 0.0),
        ("gfdrr_landslide_quake", "LS_Quake_Index", 1.0, 0.0),
        ("storm_wind_rp50", "CY_RP50_kmh", gust_factor, 0.0),
        ("storm_wind_rp100", "CY_RP100_kmh", gust_factor, 0.0),
        ("storm_wind_rp200", "CY_RP200_kmh", gust_factor, 0.0),
        ("storm_wind_rp500", "CY_RP500_kmh", gust_factor, 0.0),
        ("storm_wind_rp1000", "CY_RP1000_kmh", gust_factor, 0.0),
    ]

    try:
        priv_imgs = [
            ee.Image(f"{asset_prefix}/{asset_name}").select([0]).unmask(default_val).multiply(mult).rename(col_name)
            for asset_name, col_name, mult, default_val in private_specs
        ]
        priv_stack = ee.Image.cat(priv_imgs)
        priv_sampled = priv_stack.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=90).getInfo()
        for feat in priv_sampled.get("features", []):
            props = feat.get("properties", {})
            r_idx = props.get("row_idx")
            if r_idx in gee_dict:
                gee_dict[r_idx].update(props)
        diagnostics.append(f"✅ All 9 Private Raster Assets in `{asset_prefix}`: Loaded")
    except Exception as batch_err:
        diagnostics.append(f"ℹ️ Batch private raster query fell back to individual asset check ({str(batch_err)[:80]})")
        for asset_name, col_name, mult, default_val in private_specs:
            try:
                img = ee.Image(f"{asset_prefix}/{asset_name}").select([0]).unmask(default_val).multiply(mult).rename(col_name)
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

    # --- GROUP C: Private Uploaded Tsunami Table (gar15_tsunami_ptha) ---
    try:
        tsunami_fc = ee.FeatureCollection(f"{asset_prefix}/gar15_tsunami_ptha")
        spatial_filter = ee.Filter.withinDistance(distance=50000, leftField=".geo", rightField=".geo", maxError=50)
        nearest_join = ee.Join.saveBest(matchKey="nearest_ptha", measureKey="coast_dist_m")
        joined_fc = nearest_join.apply(fc, tsunami_fc, spatial_filter)
        for feat in joined_fc.getInfo().get("features", []):
            props = feat.get("properties", {})
            r_idx = props.get("row_idx")
            ptha_match = props.get("nearest_ptha")
            if ptha_match and "properties" in ptha_match and r_idx in gee_dict:
                p_props = ptha_match["properties"]
                raw_dist_km = float(p_props.get("coast_dist_m", 0.0)) / 1000.0
                gee_dict[r_idx]["TS_Coast_Dist_km"] = round(raw_dist_km, 2)
                gee_dict[r_idx]["TS_H0_100_m"] = float(p_props.get("ari100", 0.0) or 0.0)
                gee_dict[r_idx]["TS_H0_500_m"] = float(p_props.get("ari500", 0.0) or 0.0)
                gee_dict[r_idx]["TS_H0_1000_m"] = float(p_props.get("ari1000", 0.0) or 0.0)
                gee_dict[r_idx]["TS_H0_2500_m"] = float(p_props.get("ari2500", 0.0) or 0.0)
        diagnostics.append(f"✅ Tsunami PTHA Table (`{asset_prefix}/gar15_tsunami_ptha`): Loaded")
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
        lat = float(row["Lat"])
        lon = float(row["Lon"])
        g_props = gee_results.get(int(r_idx), {})

        elev = g_props.get("Elev_GLO30_m")
        elev = round(float(elev), 2) if elev is not None else om_elevs[idx_pos]

        slope = g_props.get("Slope_deg")
        slope = round(float(slope), 2) if slope is not None and float(slope) > 0 else om_slopes[idx_pos]

        # 1. Terrain & Micro-Topography
        hand_m = g_props.get("HAND_m")
        if hand_m is None:
            hand_m = round(max(0.0, elev * 0.45 if elev < 25 else 12.0 + slope * 0.8), 2)
        else:
            hand_m = round(float(hand_m), 2)

        slope_rad = math.radians(max(slope, 0.1))
        upa_m = float(g_props.get("UPA_km2", 1.5) or 1.5) * 1000.0
        twi = round(math.log(max(1.0, upa_m / math.tan(slope_rad))), 2)

        # 2. River Flood Curve (m)
        if "FL_RP100_m" in g_props and g_props["FL_RP100_m"] is not None:
            fl_rp10 = round(float(g_props.get("FL_RP10_m", 0.0) or 0.0), 2)
            fl_rp20 = round(float(g_props.get("FL_RP20_m", 0.0) or 0.0), 2)
            fl_rp50 = round(float(g_props.get("FL_RP50_m", 0.0) or 0.0), 2)
            fl_rp100 = round(float(g_props.get("FL_RP100_m", 0.0) or 0.0), 2)
            fl_rp200 = round(float(g_props.get("FL_RP200_m", 0.0) or 0.0), 2)
            fl_rp500 = round(float(g_props.get("FL_RP500_m", 0.0) or 0.0), 2)
        else:
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

        # 4. Coastal Surge Curve (m)
        if elev <= 15.0:
            cf_rp10 = round(max(0.0, 1.8 - elev * 0.45), 2)
            cf_rp50 = round(max(0.0, 2.5 - elev * 0.45), 2)
            cf_rp100 = round(max(0.0, 3.0 - elev * 0.45), 2)
            cf_rp250 = round(max(0.0, 3.6 - elev * 0.45), 2)
            cf_rp500 = round(max(0.0, 4.1 - elev * 0.42), 2)
            cf_rp1000 = round(max(0.0, 4.7 - elev * 0.40), 2)
        else:
            cf_rp10 = cf_rp50 = cf_rp100 = cf_rp250 = cf_rp500 = cf_rp1000 = 0.0

        # 5. Tsunami Overland Attenuation Curve (m)
        if "TS_H0_500_m" in g_props:
            ptha_dist_km = float(g_props["TS_Coast_Dist_km"])
            overland_km = max(0.0, ptha_dist_km - 12.0) if elev < 20.0 else max(0.0, ptha_dist_km - 4.0)
            est_coast_dist_km = round(overland_km, 2)
            roughness_loss = 1.5 * overland_km
            ts_rp100 = round(max(0.0, g_props["TS_H0_100_m"] - elev - roughness_loss), 2)
            ts_rp500 = round(max(0.0, g_props["TS_H0_500_m"] - elev - roughness_loss), 2)
            ts_rp1000 = round(max(0.0, g_props["TS_H0_1000_m"] - elev - roughness_loss), 2)
            ts_rp2500 = round(max(0.0, g_props["TS_H0_2500_m"] - elev - roughness_loss), 2)
        else:
            est_coast_dist_km = 50.0
            ts_rp100 = ts_rp500 = ts_rp1000 = ts_rp2500 = 0.0

        # 6. Tropical Cyclone Wind Curve (3-sec Gust km/h)
        if "CY_RP100_kmh" in g_props and g_props["CY_RP100_kmh"] is not None:
            cy_rp50 = round(float(g_props.get("CY_RP50_kmh", 0.0) or 0.0), 1)
            cy_rp100 = round(float(g_props.get("CY_RP100_kmh", 0.0) or 0.0), 1)
            cy_rp200 = round(float(g_props.get("CY_RP200_kmh", 0.0) or 0.0), 1)
            cy_rp500 = round(float(g_props.get("CY_RP500_kmh", 0.0) or 0.0), 1)
            cy_rp1000 = round(float(g_props.get("CY_RP1000_kmh", 0.0) or 0.0), 1)
        else:
            cy_rp50, cy_rp100, cy_rp200, cy_rp500, cy_rp1000 = 0.0, 0.0, 0.0, 0.0, 0.0

        # 7. Earthquake Seismic PGA (g)
        if "EQ_PGA_RP475_g" in g_props and g_props["EQ_PGA_RP475_g"] is not None:
            eq_pga_475 = round(float(g_props.get("EQ_PGA_RP475_g", 0.0) or 0.0), 3)
            eq_pga_2475 = round(eq_pga_475 * 1.8, 3)
        else:
            eq_pga_475 = 0.0
            eq_pga_2475 = 0.0

        # 8. Landslide Hazard Index (1 to 4)
        if "LS_Combined_Class" in g_props and g_props["LS_Combined_Class"] is not None:
            ls_combined = max(1, int(round(float(g_props.get("LS_Combined_Class", 1) or 1))))
            ls_rain = round(float(g_props.get("LS_Rain_Index", 0.0) or 0.0), 5)
            ls_quake = round(float(g_props.get("LS_Quake_Index", 0.0) or 0.0), 5)
        else:
            ls_combined = 1
            ls_rain = 0.0
            ls_quake = 0.0

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
            "TH_River_Flood": th_river,
            "TH_Urban_Flood": th_urban,
            "TH_Coastal_Flood": th_coastal,
            "TH_Tsunami": th_tsunami,
            "TH_Cyclone": th_cyclone,
            "TH_Earthquake": th_quake,
            "TH_Landslide": th_landslide,
            "FL_RP10_m": fl_rp10, "FL_RP20_m": fl_rp20, "FL_RP50_m": fl_rp50,
            "FL_RP100_m": fl_rp100, "FL_RP200_m": fl_rp200, "FL_RP500_m": fl_rp500,
            "UF_ADM2_Baseline": uf_adm2_baseline, "UF_Susceptibility_Score": uf_score,
            "UF_RP10_m": uf_rp10, "UF_RP50_m": uf_rp50, "UF_RP100_m": uf_rp100, "UF_RP500_m": uf_rp500,
            "CF_RP10_m": cf_rp10, "CF_RP50_m": cf_rp50, "CF_RP100_m": cf_rp100,
            "CF_RP250_m": cf_rp250, "CF_RP500_m": cf_rp500, "CF_RP1000_m": cf_rp1000,
            "TS_Coast_Dist_km": est_coast_dist_km,
            "TS_RP100_m": ts_rp100, "TS_RP500_m": ts_rp500, "TS_RP1000_m": ts_rp1000, "TS_RP2500_m": ts_rp2500,
            "CY_RP50_kmh": cy_rp50, "CY_RP100_kmh": cy_rp100, "CY_RP200_kmh": cy_rp200,
            "CY_RP500_kmh": cy_rp500, "CY_RP1000_kmh": cy_rp1000,
            "EQ_PGA_RP475_g": eq_pga_475, "EQ_PGA_RP2475_g": eq_pga_2475,
            "LS_Rain_Index": ls_rain, "LS_Quake_Index": ls_quake, "LS_Combined_Class": ls_combined
        })

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1), diagnostics

# ==============================================================================
# 5. INPUT LAYER UI
# ==============================================================================
st.subheader("1. Input Locations (Single or Batch)")
tab_manual, tab_csv = st.tabs(["📝 Manual Table Input", "📂 Upload CSV File"])

trigger_df = pd.DataFrame()

with tab_manual:
    st.caption("Edit cells directly or click the bottom row to add new locations. Provide an Address/Postal Code OR exact Lat/Lon coordinates.")
    sample_locations = pd.DataFrame([
        {"Location_ID": "LOC-01", "Name": "Samsung Taylor", "Address_or_Postal": "1530 FM973, Taylor, TX 76574, United States", "Lat": np.nan, "Lon": np.nan},
        {"Location_ID": "LOC-02", "Name": "Prime CoreGen San Lorenzo", "Address_or_Postal": "San Pascual, Batangas, Philippines", "Lat": np.nan, "Lon": np.nan},
        {"Location_ID": "LOC-03", "Name": "Okada Manila", "Address_or_Postal": "Entertainment City, New Seaside Dr, Parañaque, 1701 Kalakhang Maynila, Philippines", "Lat": np.nan, "Lon": np.nan},
        {"Location_ID": "LOC-04", "Name": "The Golf Club at Crown Colony", "Address_or_Postal": "8851 Crown Colony Blvd, Fort Myers, FL 33908, United States", "Lat": np.nan, "Lon": np.nan},
        {"Location_ID": "LOC-05", "Name": "Yoshikawa Observation Deck", "Address_or_Postal": "Yoshikawacho Yoshihara, Konan, Kochi 781-5241, Japan", "Lat": np.nan, "Lon": np.nan},
    ])
    edited_table = st.data_editor(
        sample_locations,
        num_rows="dynamic",
        column_config={
            "Lat": st.column_config.NumberColumn("Lat", format="%.6f"),
            "Lon": st.column_config.NumberColumn("Lon", format="%.6f"),
        }
    )
    if st.button("🚀 Run Multi-Peril Hazard Screening (Manual Input)", type="primary"):
        trigger_df = edited_table.copy()

with tab_csv:
    st.caption("Upload a `.csv` file with columns: `Location_ID`, `Name`, `Address_or_Postal` (and optional `Lat`, `Lon`).")
    template_csv = sample_locations.to_csv(index=False).encode("utf-8")
    st.download_button("📄 Download Blank CSV Template", data=template_csv, file_name="location_input_template.csv", mime="text/csv")
    uploaded_csv = st.file_uploader("Drag and drop your CSV file here", type=["csv"])
    if uploaded_csv is not None:
        csv_preview = pd.read_csv(uploaded_csv)
        st.dataframe(csv_preview.head(10))
        if st.button("🚀 Run Multi-Peril Hazard Screening (CSV Batch)", type="primary"):
            trigger_df = csv_preview.copy()

if not trigger_df.empty:
    with st.spinner("Geocoding locations and sampling Google Earth Engine Hazard Rasters..."):
        geocoded = geocode_dataframe(trigger_df)
        valid_rows = geocoded.dropna(subset=["Lat", "Lon"]).reset_index(drop=True)
        if valid_rows.empty:
            st.error("No valid coordinates could be resolved. Please verify the input addresses or supply Lat/Lon.")
        else:
            res_df, diag_logs = evaluate_all_perils(valid_rows, use_gee=gee_connected, asset_prefix=gee_project)
            st.session_state["results_df"] = res_df
            st.session_state["diag_logs"] = diag_logs

# ==============================================================================
# 6. OUTPUT LAYER UI
# ==============================================================================
if "results_df" in st.session_state:
    res_df = st.session_state["results_df"]
    diag_logs = st.session_state.get("diag_logs", [])

    if diag_logs:
        with st.expander("🔍 GEE Cloud Layer Diagnostics (Click to inspect asset status)", expanded=False):
            for msg in diag_logs:
                st.write(msg)

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
            zoom_start=3,
            tiles="OpenStreetMap"
        )
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
        st.dataframe(res_df[exec_cols], height=390)

        csv_export = res_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Export Full Raw Return-Period Dataset & Grades (CSV)",
            data=csv_export,
            file_name="multi_peril_hazard_return_periods_export.csv",
            mime="text/csv",
            type="primary"
        )

    st.markdown("##### Complete Quantitative Return-Period Hazard Curve Table (All 42 Columns)")
    st.dataframe(res_df)
