import os
import json
import tempfile
import logging
import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

logging.getLogger('google_auth_httplib2').setLevel(logging.ERROR)

# ======================================================
# 0. PAGE CONFIG & EARTH ENGINE AUTHENTICATION
# ======================================================
st.set_page_config(
    page_title="Trees Outside Forest & Agroforestry Monitoring System",
    page_icon="🌳",
    layout="wide"
)

@st.cache_resource
def init_ee():
    if "EE_PRIVATE_KEY_JSON" in st.secrets:
        sa_email = st.secrets["EE_SERVICE_ACCOUNT"]
        sa_key_json = st.secrets["EE_PRIVATE_KEY_JSON"]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(sa_key_json)
            key_path = f.name
        credentials = ee.ServiceAccountCredentials(sa_email, key_path)
        ee.Initialize(credentials, project="ee-ciforicrafindia")
    else:
        ee.Initialize(project="ee-ciforicrafindia")

init_ee()

st.markdown("""
<style>
  .block-container {
      padding-top: 3.5rem !important;
      padding-bottom: 0.5rem !important;
      padding-left: 1.2rem !important;
      padding-right: 1.2rem !important;
      max-width: 1500px;
  }
  .header-box {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 6px 12px;
      margin-bottom: 8px;
  }
  .sec-title {
      font-size: 12.5px;
      font-weight: 700;
      color: #111827;
      margin-bottom: 4px;
  }
  .metric-card {
      border: 1px solid #e5e7eb;
      border-radius: 5px;
      padding: 5px 8px;
      margin-bottom: 6px;
      line-height: 1.25;
  }
  .footer-box {
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 6px 10px;
      margin-top: 6px;
      font-size: 11px;
      color: #374151;
      line-height: 1.35;
  }
  /* Compact Leaflet layer control */
  .leaflet-control-layers {
      font-size: 11px !important;
      padding: 4px 6px !important;
  }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 1. CONFIGURATION & GLOBAL DATASETS
# ======================================================
ASSET_FOLDER = 'projects/ee-ciforicrafindia/assets/TreeCover'

TARGET_DISTRICTS = {
    'Odisha': ['Nuapada', 'Balangir'],
    'Madhya Pradesh': ['Sehore', 'Dewas'],
    'Uttar Pradesh': ['Jhansi', 'Lalitpur'],
    'Karnataka': ['Bijapur', 'Bagalkot']
}

DISTRICT_F1_SCORES = {
    'Nuapada' : '82.9',
    'Balangir': '82.7',
    'Sehore'  : '88.1',
    'Dewas'   : '87.2',
    'Jhansi'  : '88.1',
    'Lalitpur': '90.1',
    'Bijapur' : '84.3',
    'Bagalkot': '82.6'
}

YEARS_LIST = [2022, 2023, 2024, 2025]

districts_fc = ee.FeatureCollection('users/GEOINFORMERS/India/India_Districts_April2020-2')
buildings_fc = ee.FeatureCollection('GOOGLE/Research/open-buildings/v3/polygons').filter('confidence >= 0.65')

def get_forest_non_tof_mask():
    rfa = ee.FeatureCollection('projects/ee-geoinformers/assets/RecordedForestArea/RFA_Odisha')
    rfa_poly_mask = ee.Image().byte().paint(rfa, 1).unmask(0)
    jrc_gfc2020 = ee.Image('JRC/GFC2020/V4').select('Map').unmask(0).eq(1)
    return rfa_poly_mask.Or(jrc_gfc2020).selfMask()

FOREST_NON_TOF_MASK = get_forest_non_tof_mask()

# ======================================================
# 2. PROCESSING FUNCTIONS
# ======================================================
def load_district(state, district_name):
    shapefile_state = 'Orissa' if state == 'Odisha' else state
    return (districts_fc
            .filter(ee.Filter.eq('NAME_1', shapefile_state))
            .filter(ee.Filter.eq('DISTRICT', district_name))
            .first()
            .geometry()
            .simplify(1000))

def get_building_mask(region):
    district_bldgs = buildings_fc.filterBounds(region)
    return ee.Image().byte().paint(district_bldgs, 1).unmask(0)

def load_exported_tof(district_name, year, region, bldg_mask):
    asset_id = f"{ASSET_FOLDER}/TreeCover_{district_name}_{year}"
    return (ee.Image(asset_id)
            .unmask(0)
            .eq(1)
            .And(bldg_mask.Not())
            .And(FOREST_NON_TOF_MASK.unmask(0).Not())
            .selfMask()
            .clip(region)
            .rename('TOF'))

def get_forest_non_tof(region, bldg_mask):
    return (FOREST_NON_TOF_MASK
            .unmask(0)
            .eq(1)
            .And(bldg_mask.Not())
            .selfMask()
            .clip(region)
            .rename('Forest_NonTOF'))

def combine_total_tree_cover(tof_image, forest_image, region):
    return (tof_image.unmask(0)
            .Or(forest_image.unmask(0))
            .selfMask()
            .clip(region)
            .rename('TotalTreeCover'))

def tree_change(base_tree, current_tree):
    return (base_tree.unmask(0).toByte().multiply(2)
            .add(current_tree.unmask(0).toByte())
            .rename('TreeChange'))

@st.cache_data(show_spinner="Loading district data and map layers...")
def get_district_bundle(state, district_name, base_year, current_year):
    region = load_district(state, district_name)
    bldg_mask = get_building_mask(region)
    tof_base = load_exported_tof(district_name, base_year, region, bldg_mask)
    tof_curr = load_exported_tof(district_name, current_year, region, bldg_mask)
    forest_layer = get_forest_non_tof(region, bldg_mask)

    total_base = combine_total_tree_cover(tof_base, forest_layer, region)
    total_curr = combine_total_tree_cover(tof_curr, forest_layer, region)
    change_map = tree_change(tof_base, tof_curr)
    boundary_img = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(region)]), 1, 2)

    pixel_ha = ee.Image.pixelArea().divide(10000)
    stacked = ee.Image.cat([
        total_base.unmask(0).multiply(pixel_ha).rename('totalBase'),
        total_curr.unmask(0).multiply(pixel_ha).rename('totalCurr'),
        tof_base.unmask(0).multiply(pixel_ha).rename('tofBase'),
        tof_curr.unmask(0).multiply(pixel_ha).rename('tofCurr'),
        forest_layer.unmask(0).multiply(pixel_ha).rename('forestNonTof'),
        change_map.eq(1).unmask(0).multiply(pixel_ha).rename('gain'),
        change_map.eq(2).unmask(0).multiply(pixel_ha).rename('loss')
    ])

    stats = stacked.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=30,
        maxPixels=1e13,
        tileScale=16,
        bestEffort=True
    )

    total_base_area = ee.Number(stats.get('totalBase', 0))
    total_curr_area = ee.Number(stats.get('totalCurr', 0))
    tof_base_area = ee.Number(stats.get('tofBase', 0))
    tof_curr_area = ee.Number(stats.get('tofCurr', 0))
    forest_area = ee.Number(stats.get('forestNonTof', 0))
    gain = ee.Number(stats.get('gain', 0))
    loss = ee.Number(stats.get('loss', 0))

    district_area_ha = region.area().divide(10000)
    pct_total_tree = total_curr_area.divide(district_area_ha).multiply(100)

    summary = ee.Dictionary({
        'District': district_name,
        'State': state,
        'BaseYear': base_year,
        'CurrentYear': current_year,
        'DistrictArea_ha': district_area_ha.format('%.0f'),
        'TotalTreeAreaBase_ha': total_base_area.format('%.0f'),
        'TotalTreeAreaCurrent_ha': total_curr_area.format('%.0f'),
        'TOFAreaBase_ha': tof_base_area.format('%.0f'),
        'TOFAreaCurrent_ha': tof_curr_area.format('%.0f'),
        'ForestNonTOFArea_ha': forest_area.format('%.0f'),
        'TreeGain_ha': gain.format('%.0f'),
        'TreeLoss_ha': loss.format('%.0f'),
        'NetChange_ha': gain.subtract(loss).format('%.0f'),
        'TotalTreeCoverPercent': pct_total_tree.format('%.1f'),
        'F1Score': DISTRICT_F1_SCORES.get(district_name, 'N/A')
    }).getInfo()

    centroid = region.centroid(100).coordinates().getInfo()

    # Mask value 0 (Non-tree) on change_map so satellite basemap shows clearly underneath
    change_masked = change_map.selfMask()

    tile_urls = {
        'change': ee.Image(change_masked).getMapId({'min': 1, 'max': 3, 'palette': ['00FF00', 'FF0000', '006400']})['tile_fetcher'].url_format,
        'tot_base': ee.Image(total_base).getMapId({'palette': ['00AA00']})['tile_fetcher'].url_format,
        'tot_curr': ee.Image(total_curr).getMapId({'palette': ['006400']})['tile_fetcher'].url_format,
        'tof_base': ee.Image(tof_base).getMapId({'palette': ['FFD700']})['tile_fetcher'].url_format,
        'tof_curr': ee.Image(tof_curr).getMapId({'palette': ['FF8C00']})['tile_fetcher'].url_format,
        'forest': ee.Image(forest_layer).getMapId({'palette': ['0D9488']})['tile_fetcher'].url_format,
        'boundary': ee.Image(boundary_img).getMapId({'palette': ['000000']})['tile_fetcher'].url_format,
    }

    return summary, centroid, tile_urls

# ======================================================
# 3. DASHBOARD UI LAYOUT
# ======================================================
st.markdown("""
<div class="header-box">
  <span style="font-size:19px; font-weight:800; color:#059669;">🌳 Trees Outside Forest & Agroforestry Monitoring System</span>
  <span style="font-size:12px; color:#6b7280; margin-left:10px;">🛰️ Satellite-based Monitoring at 10 meters</span>
</div>
""", unsafe_allow_html=True)

left_col, center_col, right_col = st.columns([0.95, 2.4, 1.1], gap="small")

with left_col:
    st.markdown("<div class='sec-title'>📍 Location & Period</div>", unsafe_allow_html=True)
    state = st.selectbox("State", list(TARGET_DISTRICTS.keys()), index=2)
    district = st.selectbox("District", TARGET_DISTRICTS[state], index=0)
    c1, c2 = st.columns(2)
    base_yr = c1.selectbox("Base Yr", YEARS_LIST, index=0)
    curr_yr = c2.selectbox("Curr Yr", YEARS_LIST, index=3)

    layer_op = st.slider("Overlay Opacity", 0.1, 1.0, 0.85, 0.05)

    st.markdown("""
    <div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:5px; padding:6px 8px; font-size:10.5px; line-height:1.4; margin-top:4px;">
      <b>🗺️ Legend</b><br>
      <b>TOF Change:</b><br>
      <span style="display:inline-block;width:8px;height:8px;background:#00ff00;"></span> Gain
      <span style="display:inline-block;width:8px;height:8px;background:#ff0000;margin-left:12px;"></span> Loss
      <span style="display:inline-block;width:8px;height:8px;background:#006400;margin-left:12px;"></span> Stable Tree<br>
      <b>TOF - AF & Forest Cover:</b><br>
      <span style="display:inline-block;width:8px;height:8px;background:#00aa00;"></span> Total Base
      <span style="display:inline-block;width:8px;height:8px;background:#006400;margin-left:8px;"></span> Total Curr<br>
      <span style="display:inline-block;width:8px;height:8px;background:#ffd700;"></span> TOF Base
      <span style="display:inline-block;width:8px;height:8px;background:#ff8c00;margin-left:10px;"></span> TOF Curr<br>
      <span style="display:inline-block;width:8px;height:8px;background:#0d9488;"></span> Forest and Non-TOF Trees
    </div>
    <div style="font-size:10px; color:#6b7280; margin-top:3px;">
      💡 Hover/click the top-right layer icon on the map to toggle layers.
    </div>
    """, unsafe_allow_html=True)

s, centroid, tile_urls = get_district_bundle(state, district, base_yr, curr_yr)

with center_col:
    st.markdown(f"<div class='sec-title'>📊 Analysis Summary (for Selected Current Year), {district}</div>", unsafe_allow_html=True)
    cards = [
        ('📐', 'District Area', f"{s['DistrictArea_ha']} ha", '#ecfdf5'),
        ('🌳', 'Total Tree Cover', f"{s['TotalTreeAreaCurrent_ha']} ha", '#ecfdf5'),
        ('🌾', 'TOF & Agroforestry', f"{s['TOFAreaCurrent_ha']} ha", '#fef3c7'),
        ('📊', 'Total Tree %', f"{s['TotalTreeCoverPercent']} %", '#ecfdf5'),
        ('📈', 'TOF Gain', f"{s['TreeGain_ha']} ha", '#d1fae5'),
        ('📉', 'TOF Loss', f"{s['TreeLoss_ha']} ha", '#fee2e2'),
        ('🔄', 'TOF Net Change', f"{s['NetChange_ha']} ha", '#dbeafe'),
        ('🎯', 'Accuracy-F1score', f"{s['F1Score']} %", '#ede9fe'),
    ]
    r1 = st.columns(4)
    r2 = st.columns(4)
    for idx, (icon, title, val, bg) in enumerate(cards):
        target_col = r1[idx] if idx < 4 else r2[idx - 4]
        target_col.markdown(f"""
        <div class="metric-card" style="background:{bg};">
          <div style="font-size:9.5px; color:#6b7280;">{icon} {title}</div>
          <div style="font-size:12.5px; font-weight:700; color:#111827;">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    # Map with collapsed LayerControl and hidden raw basemap URL
    m = folium.Map(
        location=[centroid[1], centroid[0]],
        zoom_start=9,
        max_zoom=12,
        tiles=None
    )

    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google Hybrid',
        name='Google Satellite Hybrid',
        overlay=False,
        control=False,
        max_zoom=12
    ).add_to(m)

    layers_spec = [
        (tile_urls['change'], 'TOF Change Map', True, layer_op),
        (tile_urls['tot_base'], f'Total Tree {base_yr}', False, layer_op),
        (tile_urls['tot_curr'], f'Total Tree {curr_yr}', False, layer_op),
        (tile_urls['tof_base'], f'TOF & AF {base_yr}', False, layer_op),
        (tile_urls['tof_curr'], f'TOF & AF {curr_yr}', False, layer_op),
        (tile_urls['forest'], 'Forest and Non-TOF Trees', False, layer_op),
        (tile_urls['boundary'], 'Boundary', True, 1.0),
    ]

    for url, name, show, op in layers_spec:
        folium.raster_layers.TileLayer(
            tiles=url,
            attr='Google Earth Engine',
            name=name,
            overlay=True,
            control=True,
            show=show,
            opacity=op,
            max_zoom=12
        ).add_to(m)

    folium.LayerControl(position='topright', collapsed=True).add_to(m)

    st_folium(
        m,
        height=330,
        use_container_width=True,
        returned_objects=[],
        key=f"map_{state}_{district}_{base_yr}_{curr_yr}_{layer_op}"
    )

with right_col:
    st.markdown("<div class='sec-title'>📈 Detailed Results</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px; padding:7px 9px; font-size:11px; line-height:1.45; margin-bottom:8px;">
      <b>{s['District'].upper()}, {s['State']}</b><br>
      <b>District Area:</b> {s['DistrictArea_ha']} ha<br>
      <b>Total Tree Cover:</b> {s['TotalTreeAreaCurrent_ha']} ha ({s['TotalTreeCoverPercent']}%)<br>
      <b>TOF & Agroforestry:</b> {s['TOFAreaCurrent_ha']} ha<br>
      <b>Forest & Non-TOF Trees:</b> {s['ForestNonTOFArea_ha']} ha<br>
      <b>TOF Gain / Loss:</b> +{s['TreeGain_ha']} / -{s['TreeLoss_ha']} ha<br>
      <b>TOF Net Change:</b> {s['NetChange_ha']} ha<br>
      <b>Accuracy (F1 Score):</b> {s['F1Score']} %
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sec-title'>📊 Area Comparison</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=110)
    labels = [f'Total\n{base_yr}', f'Total\n{curr_yr}', f'TOF\n{base_yr}', f'TOF\n{curr_yr}', 'Forest &\nNon-TOF']
    values = [
        float(s['TotalTreeAreaBase_ha']),
        float(s['TotalTreeAreaCurrent_ha']),
        float(s['TOFAreaBase_ha']),
        float(s['TOFAreaCurrent_ha']),
        float(s['ForestNonTOFArea_ha'])
    ]
    colors = ['#1B5E20', '#43A047', '#FBC02D', '#F57F17', '#0D9488']
    bars = ax.bar(labels, values, color=colors, width=0.52)
    ax.set_title(f"Area Comparison ({district.upper()})", fontsize=8.5, fontweight='bold')
    ax.set_ylabel("Area (ha)", fontsize=7.5)
    ax.tick_params(axis='both', labelsize=6.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{int(h):,}',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=6.2)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("""
<div class="footer-box">
  <div><b>🛰️ Methodology:</b> Trees Outside Forest and Agroforestry (TOF-AF) cover is mapped using supervised machine learning applied to Google’s 10 m Satellite Embedding dataset (61-band AI feature representations). District-level classification accuracy is evaluated against an independent validation sample using the F1 score.</div>
  <div style="margin-top:2px;"><b>⚠️ Limitations:</b> While 10 m resolution captures block plantations, orchards, and linear agroforestry effectively, isolated single trees and sparsely scattered canopies may experience omission or commission errors due to sub-pixel mixing.</div>
</div>
""", unsafe_allow_html=True)
