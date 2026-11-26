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

# Compact CSS to match your Colab dashboard aesthetics
st.markdown("""
<style>
  .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 1450px; }
  .header-box { background:#ffffff; border:1px solid #e5e7eb; border-radius:6px; padding:8px 14px; margin-bottom:8px; }
  .metric-card { border:1px solid #e5e7eb; border-radius:6px; padding:6px 10px; margin-bottom:6px; }
  .footer-box { background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px; padding:8px 12px; margin-top:8px; font-size:12px; color:#374151; }
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
            .filter(ee.Filter.eq('NAME_1', shapefileState))
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

@st.cache_data(show_spinner="Computing district statistics...")
def compute_district_stats(state, district_name, base_year, current_year):
    region = load_district(state, district_name)
    bldg_mask = get_building_mask(region)
    tof_base = load_exported_tof(district_name, base_year, region, bldg_mask)
    tof_curr = load_exported_tof(district_name, current_year, region, bldg_mask)
    forest_layer = get_forest_non_tof(region, bldg_mask)

    total_base = combine_total_tree_cover(tof_base, forest_layer, region)
    total_curr = combine_total_tree_cover(tof_curr, forest_layer, region)
    change_map = tree_change(tof_base, tof_curr)

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

    return ee.Dictionary({
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

# ======================================================
# 3. DASHBOARD UI LAYOUT
# ======================================================
st.markdown("""
<div class="header-box">
  <span style="font-size:21px; font-weight:800; color:#059669;">🌳 Trees Outside Forest & Agroforestry Monitoring System</span>
  <span style="font-size:12px; color:#6b7280; margin-left:12px;">🛰️ Satellite-based Monitoring at 10 meters</span>
</div>
""", unsafe_allow_html=True)

left_col, center_col, right_col = st.columns([1.05, 2.3, 1.15], gap="small")

with left_col:
    st.markdown("**📍 Location & Period**")
    state = st.selectbox("State", list(TARGET_DISTRICTS.keys()), index=2)
    district = st.selectbox("District", TARGET_DISTRICTS[state], index=0)
    c1, c2 = st.columns(2)
    base_yr = c1.selectbox("Base Yr", YEARS_LIST, index=0)
    curr_yr = c2.selectbox("Curr Yr", YEARS_LIST, index=3)

    st.markdown("**🗂️ Map Layers & Opacity**")
    show_change = st.checkbox("TOF Change Map", value=True)
    op_change = st.slider("Change Opacity", 0.0, 1.0, 0.75, 0.05, label_visibility="collapsed")

    show_tot_base = st.checkbox(f"Total Tree {base_yr}", value=False)
    show_tot_curr = st.checkbox(f"Total Tree {curr_yr}", value=False)
    show_tof_base = st.checkbox(f"TOF & AF {base_yr}", value=False)
    show_tof_curr = st.checkbox(f"TOF & AF {curr_yr}", value=False)
    show_forest = st.checkbox("Forest and Non-TOF Trees", value=False)
    layer_op = st.slider("Tree / Forest Layer Opacity", 0.0, 1.0, 0.80, 0.05)

    st.markdown("""
    <div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:5px; padding:6px 8px; font-size:11px; line-height:1.4;">
      <b>🗺️ Legend</b><br>
      <b>TOF Change:</b>
      <span style="display:inline-block;width:8px;height:8px;background:#ffffff;border:1px solid #ccc;"></span> Non-tree
      <span style="display:inline-block;width:8px;height:8px;background:#00ff00;margin-left:6px;"></span> Gain
      <span style="display:inline-block;width:8px;height:8px;background:#ff0000;margin-left:6px;"></span> Loss<br>
      <span style="display:inline-block;width:8px;height:8px;background:#006400;"></span> Stable Tree<br>
      <b>TOF - AF & Forest cover:</b><br>
      <span style="display:inline-block;width:8px;height:8px;background:#00aa00;"></span> Total Base
      <span style="display:inline-block;width:8px;height:8px;background:#006400;margin-left:8px;"></span> Total Curr<br>
      <span style="display:inline-block;width:8px;height:8px;background:#ffd700;"></span> TOF Base
      <span style="display:inline-block;width:8px;height:8px;background:#ff8c00;margin-left:10px;"></span> TOF Curr<br>
      <span style="display:inline-block;width:8px;height:8px;background:#0d9488;"></span> Forest and Non-TOF Trees
    </div>
    """, unsafe_allow_html=True)

s = compute_district_stats(state, district, base_yr, curr_yr)

# Top Analysis Summary Cards (Left-aligned)
with center_col:
    st.markdown(f"**📊 Analysis Summary (for Selected Current Year), {district}**")
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
          <div style="font-size:10px; color:#6b7280;">{icon} {title}</div>
          <div style="font-size:13px; font-weight:700; color:#111827;">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    # Render Map
    region = load_district(state, district)
    bldg_mask = get_building_mask(region)
    tof_base = load_exported_tof(district, base_yr, region, bldg_mask)
    tof_curr = load_exported_tof(district, curr_yr, region, bldg_mask)
    forest_layer = get_forest_non_tof(region, bldg_mask)
    total_base = combine_total_tree_cover(tof_base, forest_layer, region)
    total_curr = combine_total_tree_cover(tof_curr, forest_layer, region)
    change_map = tree_change(tof_base, tof_curr)

def add_ee_layer(fmap, ee_image, vis_params, name, show=True, opacity=1.0):
      map_id_dict = ee.Image(ee_image).getMapId(vis_params)
      folium.raster_layers.TileLayer(
          tiles=map_id_dict['tile_fetcher'].url_format,
          attr='Google Earth Engine',
          name=name,
          overlay=True,
          control=True,
          show=show,
          opacity=opacity
      ).add_to(fmap)

  centroid = region.centroid(100).coordinates().getInfo()
  m = folium.Map(
      location=[centroid[1], centroid[0]],
      zoom_start=9,
      max_zoom=12,
      tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
      attr='Google Hybrid'
  )

  if show_change:
      add_ee_layer(m, change_map, {'min': 0, 'max': 3, 'palette': ['FFFFFF', '00FF00', 'FF0000', '006400']}, 'TOF Change Map', True, op_change)
  if show_tot_base:
      add_ee_layer(m, total_base, {'palette': ['00AA00']}, f'Total Tree {base_yr}', True, layer_op)
  if show_tot_curr:
      add_ee_layer(m, total_curr, {'palette': ['006400']}, f'Total Tree {curr_yr}', True, layer_op)
  if show_tof_base:
      add_ee_layer(m, tof_base, {'palette': ['FFD700']}, f'TOF & AF {base_yr}', True, layer_op)
  if show_tof_curr:
      add_ee_layer(m, tof_curr, {'palette': ['FF8C00']}, f'TOF & AF {curr_yr}', True, layer_op)
  if show_forest:
      add_ee_layer(m, forest_layer, {'palette': ['0D9488']}, 'Forest and Non-TOF Trees', True, layer_op)

  boundary_img = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(region)]), 1, 2)
  add_ee_layer(m, boundary_img, {'palette': ['000000']}, 'Boundary', True, 1.0)
  st_folium(m, height=420, use_container_width=True, returned_objects=[])

with right_col:
    st.markdown("**📈 Detailed Results**")
    st.markdown(f"""
    <div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px; padding:8px; font-size:12px; line-height:1.5; margin-bottom:10px;">
      <b>{s['District'].upper()}, {s['State']}</b><br>
      <b>District Area:</b> {s['DistrictArea_ha']} ha<br>
      <b>Total Tree Cover:</b> {s['TotalTreeAreaCurrent_ha']} ha ({s['TotalTreeCoverPercent']}%)<br>
      <b>TOF & Agroforestry:</b> {s['TOFAreaCurrent_ha']} ha<br>
      <b>Forest and Non-TOF Trees:</b> {s['ForestNonTOFArea_ha']} ha<br>
      <b>TOF Gain / Loss:</b> +{s['TreeGain_ha']} / -{s['TreeLoss_ha']} ha<br>
      <b>TOF Net Change:</b> {s['NetChange_ha']} ha<br>
      <b>Accuracy (F1 Score):</b> {s['F1Score']} %
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**📊 Area Comparison**")
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=110)
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
    ax.set_title(f"Area Comparison ({district.upper()})", fontsize=9, fontweight='bold')
    ax.set_ylabel("Area (ha)", fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{int(h):,}',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=6.5)
    plt.tight_layout()
    st.pyplot(fig)

# Methodology & Limitations Footer
st.markdown("""
<div class="footer-box">
  <div><b>🛰️ Methodology:</b> Trees Outside Forest and Agroforestry (TOF-AF) cover is mapped using supervised machine learning applied to Google’s 10 m Satellite Embedding dataset (61-band AI feature representations). District-level classification accuracy is evaluated against an independent validation sample using the F1 score.</div>
  <div style="margin-top:3px;"><b>⚠️ Limitations:</b> While 10 m resolution captures block plantations, orchards, and linear agroforestry effectively, isolated single trees and sparsely scattered canopies may experience omission or commission errors due to sub-pixel mixing.</div>
</div>
""", unsafe_allow_html=True)
