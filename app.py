"""
広告面板 視認エリア計算アプリ
Plateau CityGML + 10次メッシュ LOS可視化
"""
import io
import math
import warnings
import re
import struct
import zlib
import urllib.request
import json as _json
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon, LineString, box
from pyproj import Transformer, CRS
from lxml import etree

try:
    import folium
    from streamlit_folium import st_folium
    _FOLIUM_OK = True
except ImportError:
    _FOLIUM_OK = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────────────────────────────────────
SECTOR_ANGLE_DEG         = 120.0
EYE_HEIGHT_M             = 1.5
SAMPLE_N                 = 5
VISIBLE_RATIO_THRESHOLD  = 0.80
LOS_TOLERANCE_M          = 0.1

COLORS = [
    "#e63946", "#2196f3", "#ff9800", "#4caf50",
    "#9c27b0", "#00bcd4", "#f44336", "#8bc34a",
]


def _hex_to_rgba(hex_color: str, alpha: float = 0.45) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ─────────────────────────────────────────────────────────────────────────────
# 10次メッシュ エンコード (JIS X 0410, 15桁)
# ─────────────────────────────────────────────────────────────────────────────

def encode_mesh10(lat: float, lon: float) -> str:
    p = int(lat * 1.5)
    q = int(lon - 100.0)
    lat_rem = lat - p / 1.5
    lon_rem = lon - (q + 100.0)
    lat_sz, lon_sz = 2.0 / 3.0, 1.0
    code = f"{p:02d}{q:02d}"

    lat_sz /= 8; lon_sz /= 8
    r2 = min(int(lat_rem / lat_sz), 7)
    c2 = min(int(lon_rem / lon_sz), 7)
    lat_rem -= r2 * lat_sz; lon_rem -= c2 * lon_sz
    code += f"{r2}{c2}"

    lat_sz /= 10; lon_sz /= 10
    r3 = min(int(lat_rem / lat_sz), 9)
    c3 = min(int(lon_rem / lon_sz), 9)
    lat_rem -= r3 * lat_sz; lon_rem -= c3 * lon_sz
    code += f"{r3}{c3}"

    for _ in range(7):
        lat_sz /= 2; lon_sz /= 2
        eps = 1e-12
        n = lat_rem >= lat_sz - eps
        e = lon_rem >= lon_sz - eps
        if n and e:    d = 4
        elif n:        d = 3
        elif e:        d = 2
        else:          d = 1
        if n: lat_rem -= lat_sz
        if e: lon_rem -= lon_sz
        code += str(d)

    return code


def mesh10_cell_size() -> tuple:
    lat_sz = (2.0 / 3.0) / 8 / 10 / (2 ** 7)
    lon_sz = 1.0 / 8 / 10 / (2 ** 7)
    return lat_sz, lon_sz


# ─────────────────────────────────────────────────────────────────────────────
# ジオメトリ補助
# ─────────────────────────────────────────────────────────────────────────────

def local_scale(lat: float):
    return 111320.0, 111320.0 * math.cos(math.radians(lat))


def create_sector(lat: float, lon: float, facing_deg: float,
                  radius_m: float = 500.0) -> Polygon:
    lat_sc, lon_sc = local_scale(lat)
    half = SECTOR_ANGLE_DEG / 2.0
    angles = np.linspace(facing_deg - half, facing_deg + half, 90)
    coords = [(lon, lat)]
    for az in angles:
        math_rad = math.radians(90.0 - az)
        dx = radius_m * math.cos(math_rad) / lon_sc
        dy = radius_m * math.sin(math_rad) / lat_sc
        coords.append((lon + dx, lat + dy))
    coords.append((lon, lat))
    return Polygon(coords)


# ─────────────────────────────────────────────────────────────────────────────
# CityGML パーサー
# ─────────────────────────────────────────────────────────────────────────────

_GML_NS   = "http://www.opengis.net/gml"
_BLDG_NS  = "http://www.opengis.net/citygml/building/1.0"
_BLDG_NS2 = "http://www.opengis.net/citygml/building/2.0"


def _detect_crs(root) -> str:
    srs = root.get("srsName", "")
    if not srs:
        for el in root.iter():
            srs = el.get("srsName", "")
            if srs:
                break

    # OGC URI: http://www.opengis.net/def/crs/EPSG/0/6697
    #          EPSG/VERSION/CODE → 末尾の数字がEPSGコード
    m = re.search(r"EPSG/\d+/(\d+)", srs, re.IGNORECASE)
    if m:
        return f"EPSG:{m.group(1)}"

    # URN: urn:ogc:def:crs:EPSG::6697 or urn:ogc:def:crs:EPSG:6.6:6697
    m = re.search(r"crs:EPSG:[^:]*:(\d+)", srs, re.IGNORECASE)
    if m:
        return f"EPSG:{m.group(1)}"

    # epsg.xml#NNNN 形式: http://www.opengis.net/gml/srs/epsg.xml#6697
    m = re.search(r"epsg\.xml#(\d+)", srs, re.IGNORECASE)
    if m:
        return f"EPSG:{m.group(1)}"

    # シンプル形式: EPSG:6697 (4桁以上を要求して EPSG:0 を回避)
    m = re.search(r"EPSG[:/](\d{4,})", srs, re.IGNORECASE)
    if m:
        return f"EPSG:{m.group(1)}"

    return "EPSG:6668"  # Plateau デフォルト (JGD2011)


def _detect_swap_xy(src_crs: str) -> bool:
    """
    CRS の第1軸が緯度（北）かどうかを判定。
    True → GML座標は (lat, lon) なので swap が必要。
    """
    try:
        crs_obj = CRS(src_crs)
        direction = crs_obj.axis_info[0].direction.lower()
        return direction in ("north", "south")
    except Exception:
        epsg_match = re.search(r"(\d{4,5})$", src_crs)
        if epsg_match:
            epsg = int(epsg_match.group(1))
            # 主要な地理座標系 (緯度先行) → swap 必要
            return epsg in (4326, 6668, 6697, 4019, 4612)
        return False


def _parse_pos_list(text: str, dim: int = 3) -> list:
    vals = [float(v) for v in text.split()]
    return [tuple(vals[i:i+dim]) for i in range(0, len(vals) - dim + 1, dim)]


def _polygon_from_pos_list(el, dim: int = 3, swap_xy: bool = False) -> Optional[Polygon]:
    ns = _GML_NS
    ring = el.find(f".//{{{ns}}}LinearRing")
    if ring is None:
        return None
    pos_el = ring.find(f"{{{ns}}}posList")
    if pos_el is None or not pos_el.text:
        return None
    pts = _parse_pos_list(pos_el.text, dim)
    if len(pts) < 3:
        return None
    if swap_xy:
        # (lat, lon, ...) → (lon, lat) for shapely
        return Polygon([(p[1], p[0]) for p in pts])
    return Polygon([(p[0], p[1]) for p in pts])


def parse_citygml(file_bytes: bytes) -> gpd.GeoDataFrame:
    """
    Plateau CityGML から建物フットプリント (GeoDataFrame, WGS84) を生成。
    座標軸順序を自動検出して正しく処理する。
    """
    root = etree.fromstring(file_bytes)
    src_crs = _detect_crs(root)
    swap_xy = _detect_swap_xy(src_crs)
    to_wgs84 = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    dim = 3
    for el in root.iter(f"{{{_GML_NS}}}posList"):
        sd = el.get("srsDimension")
        if sd:
            dim = int(sd)
            break

    bldg_ns = _BLDG_NS
    buildings = root.findall(f".//{{{bldg_ns}}}Building")
    if not buildings:
        bldg_ns = _BLDG_NS2
        buildings = root.findall(f".//{{{bldg_ns}}}Building")

    rows = []
    for bldg in buildings:
        h_el = bldg.find(f".//{{{bldg_ns}}}measuredHeight")
        height = float(h_el.text) if h_el is not None and h_el.text else 0.0

        footprint = None

        # lod0FootPrint 優先
        fp_el = bldg.find(f".//{{{bldg_ns}}}lod0FootPrint")
        if fp_el is not None:
            poly_el = fp_el.find(f".//{{{_GML_NS}}}Polygon")
            if poly_el is not None:
                footprint = _polygon_from_pos_list(poly_el, dim=2, swap_xy=swap_xy)

        # lod1Solid フォールバック
        if footprint is None:
            solid_el = bldg.find(f".//{{{bldg_ns}}}lod1Solid")
            if solid_el is None:
                solid_el = bldg.find(f".//{{{_GML_NS}}}Solid")
            if solid_el is not None:
                candidates = []
                for poly_el in solid_el.findall(f".//{{{_GML_NS}}}Polygon"):
                    pos_el = poly_el.find(f".//{{{_GML_NS}}}posList")
                    if pos_el is None or not pos_el.text:
                        continue
                    pts = _parse_pos_list(pos_el.text, dim)
                    if len(pts) < 3:
                        continue
                    z_idx = 2 if dim >= 3 else None
                    z_vals = [p[z_idx] for p in pts] if z_idx is not None else [0]
                    candidates.append((min(z_vals), pts))
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    pts = candidates[0][1]
                    if swap_xy:
                        footprint = Polygon([(p[1], p[0]) for p in pts])
                    else:
                        footprint = Polygon([(p[0], p[1]) for p in pts])

        if footprint is None or not footprint.is_valid or footprint.is_empty:
            continue

        try:
            # footprint coords are (lon, lat) after swap_xy → pass as (x, y) to always_xy transformer
            xcoords, ycoords = to_wgs84.transform(
                [c[0] for c in footprint.exterior.coords],
                [c[1] for c in footprint.exterior.coords],
            )
            footprint_wgs = Polygon(zip(xcoords, ycoords))
            rows.append({"height": height, "geometry": footprint_wgs})
        except Exception:
            continue

    if not rows:
        return gpd.GeoDataFrame(columns=["height", "geometry"], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf = gdf[gdf["height"] > 0].reset_index(drop=True)
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# Plateau 自動取得
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_plateau_catalog() -> dict:
    """
    geospatial.jp CKAN API から Plateau データセット一覧を取得。
    Returns: {muniCd: dataset_id}  (同一市区町村は最新年度のみ保持)
    """
    catalog = {}
    rows_per_page = 100
    start = 0

    while True:
        url = (
            f"https://www.geospatial.jp/ckan/api/3/action/package_search"
            f"?fq=tags:PLATEAU&rows={rows_per_page}&start={start}"
        )
        with urllib.request.urlopen(url, timeout=20) as r:
            data = _json.loads(r.read())

        results = data["result"]["results"]
        total   = data["result"]["count"]

        for item in results:
            name = item.get("name", "")
            m = re.match(r"^plateau-(\d{5})-.*-(\d{4})$", name)
            if m:
                muni_cd = m.group(1)
                year = int(m.group(2))
                existing = catalog.get(muni_cd, "")
                if not existing or int(existing.split("-")[-1]) < year:
                    catalog[muni_cd] = name

        start += rows_per_page
        if start >= total:
            break

    return catalog


def _gsi_reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """国土地理院 逆ジオコーダー API で緯度経度 → 市区町村コード"""
    url = (f"https://mreversegeocoder.gsi.go.jp/reverse-geocoder/"
           f"LonLatToAddress?lat={lat}&lon={lon}")
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = _json.loads(r.read())
        return data["results"]["muniCd"]
    except Exception:
        return None


def _get_plateau_zip_url(dataset_id: str) -> Optional[str]:
    """CKAN API で dataset_id → CityGML ZIP URL を取得 (v3 優先)"""
    url = f"https://www.geospatial.jp/ckan/api/3/action/package_show?id={dataset_id}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = _json.loads(r.read())
        resources = data["result"]["resources"]
        # v3 を優先し、なければ最初の CityGML zip
        v3_url = None
        fallback_url = None
        for res in resources:
            name = res.get("name", "")
            rurl = res.get("url", "")
            if "CityGML" in name and rurl.endswith(".zip"):
                if "v3" in name or "v3" in rurl:
                    v3_url = rurl
                elif fallback_url is None:
                    fallback_url = rurl
        return v3_url or fallback_url
    except Exception:
        return None


def _read_zip_cd(zip_url: str) -> dict:
    """
    HTTP Range リクエストで ZIP セントラルディレクトリを読む。
    Returns: {basename: (local_offset, comp_size, method)}
    """
    req = urllib.request.Request(zip_url, headers={"Range": "bytes=-65536"})
    with urllib.request.urlopen(req, timeout=30) as r:
        tail = r.read()

    sig = b"PK\x05\x06"
    pos = tail.rfind(sig)
    if pos == -1:
        raise ValueError("ZIP EOCD が見つかりません")

    eocd = tail[pos:]
    cd_size   = struct.unpack_from("<I", eocd, 12)[0]
    cd_offset = struct.unpack_from("<I", eocd, 16)[0]

    req2 = urllib.request.Request(
        zip_url, headers={"Range": f"bytes={cd_offset}-{cd_offset+cd_size-1}"}
    )
    with urllib.request.urlopen(req2, timeout=30) as r:
        cd_data = r.read()

    files = {}
    offset = 0
    while offset + 46 <= len(cd_data):
        if cd_data[offset:offset+4] != b"PK\x01\x02":
            break
        method      = struct.unpack_from("<H", cd_data, offset+10)[0]
        comp_size   = struct.unpack_from("<I", cd_data, offset+20)[0]
        fname_len   = struct.unpack_from("<H", cd_data, offset+28)[0]
        extra_len   = struct.unpack_from("<H", cd_data, offset+30)[0]
        comment_len = struct.unpack_from("<H", cd_data, offset+32)[0]
        local_off   = struct.unpack_from("<I", cd_data, offset+42)[0]
        fname = cd_data[offset+46:offset+46+fname_len].decode("utf-8", errors="replace")
        if "bldg" in fname and fname.endswith(".gml"):
            basename = fname.split("/")[-1]
            files[basename] = (local_off, comp_size, method)
        offset += 46 + fname_len + extra_len + comment_len

    return files


def _extract_gml_from_zip(zip_url: str, local_off: int,
                           comp_size: int, method: int) -> bytes:
    """ZIP から特定 GML ファイルを HTTP Range で抽出・解凍"""
    lh_req = urllib.request.Request(
        zip_url, headers={"Range": f"bytes={local_off}-{local_off+29}"}
    )
    with urllib.request.urlopen(lh_req, timeout=30) as r:
        lh = r.read()
    lh_fname_len = struct.unpack_from("<H", lh, 26)[0]
    lh_extra_len = struct.unpack_from("<H", lh, 28)[0]
    data_start = local_off + 30 + lh_fname_len + lh_extra_len

    data_req = urllib.request.Request(
        zip_url, headers={"Range": f"bytes={data_start}-{data_start+comp_size-1}"}
    )
    with urllib.request.urlopen(data_req, timeout=120) as r:
        comp_data = r.read()

    return zlib.decompress(comp_data, -15) if method == 8 else comp_data


def get_needed_3rd_mesh_prefixes(billboards_df: pd.DataFrame) -> set:
    """広告面板の扇形エリアに必要な 3 次メッシュコード（8 桁）セットを計算"""
    lat_sz_3 = (2.0 / 3.0) / 8 / 10
    lon_sz_3 = 1.0 / 8 / 10
    prefixes = set()
    for _, bb in billboards_df.iterrows():
        sector = create_sector(bb.latitude, bb.longitude, bb.facing_deg, bb.max_range_m)
        minlon, minlat, maxlon, maxlat = sector.bounds
        la = math.floor(minlat / lat_sz_3) * lat_sz_3
        while la <= maxlat:
            lo = math.floor(minlon / lon_sz_3) * lon_sz_3
            while lo <= maxlon:
                if sector.intersects(box(lo, la, lo + lon_sz_3, la + lat_sz_3)):
                    code = encode_mesh10(la + lat_sz_3 / 2, lo + lon_sz_3 / 2)
                    prefixes.add(code[:8])
                lo += lon_sz_3
            la += lat_sz_3
    return prefixes


def auto_fetch_citygml(billboards_df: pd.DataFrame,
                       log_box) -> Optional[gpd.GeoDataFrame]:
    """
    広告面板データから必要な Plateau CityGML を自動取得。
    log_box: st.empty() コンテナ（ログ表示用）
    """
    logs = []

    def log(msg: str):
        logs.append(msg)
        log_box.markdown("\n\n".join(logs))

    # ① カタログ取得
    log("📋 Plateau カタログを取得中...")
    try:
        catalog = _fetch_plateau_catalog()
    except Exception as e:
        log(f"❌ カタログ取得エラー: {e}")
        return None
    log(f"✅ カタログ取得完了（{len(catalog)} 市区町村がPlateau対応）")

    # ② 逆ジオコーディング
    log("📍 広告面板の市区町村を特定中...")
    muni_cds = set()
    for _, bb in billboards_df.iterrows():
        muni_cd = _gsi_reverse_geocode(bb.latitude, bb.longitude)
        if muni_cd:
            muni_cds.add(muni_cd)
    if not muni_cds:
        log("❌ 市区町村コードを取得できませんでした（ネットワークを確認してください）")
        return None
    log(f"✅ 市区町村コード: {', '.join(sorted(muni_cds))}")

    # ③ 必要な 3 次メッシュコードを計算
    log("🗺️ 必要なメッシュタイルを計算中...")
    needed_prefixes = get_needed_3rd_mesh_prefixes(billboards_df)
    log(f"✅ 対象 3 次メッシュ: {', '.join(sorted(needed_prefixes))}（{len(needed_prefixes)} タイル）")

    # ④ 各市区町村の GML を取得
    all_gdfs = []
    for muni_cd in sorted(muni_cds):
        dataset_id = catalog.get(muni_cd)
        if not dataset_id:
            log(f"⚠️ 市区町村 {muni_cd} の Plateau データが見つかりません（対応エリア外の可能性）")
            continue

        log(f"🔍 `{dataset_id}` の ZIP URL を取得中...")
        zip_url = _get_plateau_zip_url(dataset_id)
        if not zip_url:
            log(f"⚠️ `{dataset_id}` の ZIP URL が取得できませんでした")
            continue

        log(f"📦 ZIP インデックスを解析中（ファイル全体はダウンロードしません）...")
        try:
            cd = _read_zip_cd(zip_url)
        except Exception as e:
            log(f"❌ ZIP 解析エラー: {e}")
            continue

        needed = {
            fname: info for fname, info in cd.items()
            if any(fname.startswith(p) for p in needed_prefixes)
        }

        if not needed:
            log(f"⚠️ 対象メッシュの GML が ZIP 内に見つかりませんでした")
            continue

        log(f"⬇️ {len(needed)} 個の GML ファイルをダウンロード中...")
        for fname, (local_off, comp_size, method) in needed.items():
            log(f"　　`{fname}` ({comp_size // 1024:,} KB 圧縮)...")
            try:
                gml_bytes = _extract_gml_from_zip(zip_url, local_off, comp_size, method)
                gdf = parse_citygml(gml_bytes)
                if not gdf.empty:
                    all_gdfs.append(gdf)
                    log(f"　　✅ `{fname}`: 建物 {len(gdf):,} 棟")
                else:
                    log(f"　　⚠️ `{fname}`: 建物データが空でした")
            except Exception as e:
                log(f"　　❌ `{fname}` 取得失敗: {e}")

    if not all_gdfs:
        log("❌ 建物データを取得できませんでした")
        return None

    combined = gpd.GeoDataFrame(
        pd.concat(all_gdfs, ignore_index=True), crs="EPSG:4326"
    )
    # 3次メッシュ単位で余分に取得した扇形エリア外の建物を除去
    _sec_union = None
    for _, _bb in billboards_df.iterrows():
        _s = create_sector(_bb.latitude, _bb.longitude, _bb.facing_deg, _bb.max_range_m)
        _sec_union = _s if _sec_union is None else _sec_union.union(_s)
    if _sec_union is not None:
        before = len(combined)
        combined = combined[
            combined.geometry.intersects(_sec_union.buffer(0.0001))
        ].reset_index(drop=True)
        log(f"✂️ 扇形エリア外を除去: {before:,} → {len(combined):,} 棟")
    log(f"\n✅ **取得完了: 建物 {len(combined):,} 棟**")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# LOS 判定
# ─────────────────────────────────────────────────────────────────────────────

def _is_blocked(src_lon, src_lat, src_h,
                tgt_lon, tgt_lat, tgt_h,
                candidates: gpd.GeoDataFrame,
                lat_sc: float, lon_sc: float) -> bool:
    dx_m = (tgt_lon - src_lon) * lon_sc
    dy_m = (tgt_lat - src_lat) * lat_sc
    D_m  = math.sqrt(dx_m ** 2 + dy_m ** 2)
    if D_m < 1e-3:
        return False

    ray = LineString([(src_lon, src_lat), (tgt_lon, tgt_lat)])

    for _, bldg in candidates.iterrows():
        if not ray.intersects(bldg.geometry):
            continue
        inter = ray.intersection(bldg.geometry.boundary)
        if inter.is_empty:
            inter = ray.intersection(bldg.geometry)
        if inter.is_empty:
            continue

        pts = list(inter.geoms) if hasattr(inter, "geoms") else [inter]
        for ipt in pts:
            if not hasattr(ipt, "x"):
                continue
            dx_b = (ipt.x - src_lon) * lon_sc
            dy_b = (ipt.y - src_lat) * lat_sc
            d_b  = math.sqrt(dx_b ** 2 + dy_b ** 2)
            t    = min(d_b / D_m, 1.0)
            if bldg["height"] > src_h * (1 - t) + tgt_h * t + LOS_TOLERANCE_M:
                return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# 視認計算
# ─────────────────────────────────────────────────────────────────────────────

def compute_visibility(bb: dict, buildings_gdf: Optional[gpd.GeoDataFrame],
                       progress_cb=None) -> tuple:
    """
    視認エリア計算。
    - デッドゾーン: 原点から半径 height_m 以内を除外
    - メッシュ判定: メッシュ面積の80%以上が有効扇形内 → 有効メッシュ
    - 建物LOS: 建物データがある場合、重心点のLOS確認で完全遮蔽を除外
    """
    lat, lon  = bb["latitude"], bb["longitude"]
    h         = bb["height_m"]
    facing    = bb["facing_deg"]
    radius    = bb.get("max_range_m", 500.0)
    sid       = bb.get("screen_id", "B001")

    sector = create_sector(lat, lon, facing, radius)
    lat_sz, lon_sz = mesh10_cell_size()
    lat_sc, lon_sc = local_scale(lat)

    # ── デッドゾーン: 原点から height_m 以内の扇形を除外 ──────────────────
    dead_r_deg = h / ((lat_sc + lon_sc) / 2)
    dead_zone  = Point(lon, lat).buffer(dead_r_deg)
    eff_sector = sector.difference(dead_zone)

    if eff_sector.is_empty:
        return pd.DataFrame(), eff_sector, 0

    # ── 有効扇形と交差するメッシュを列挙 ────────────────────────────────
    minlon, minlat, maxlon, maxlat = eff_sector.bounds
    start_lat = math.floor(minlat / lat_sz) * lat_sz
    start_lon = math.floor(minlon / lon_sz) * lon_sz
    all_lats  = np.arange(start_lat, maxlat + lat_sz, lat_sz)
    all_lons  = np.arange(start_lon, maxlon + lon_sz, lon_sz)

    mesh_boxes = [
        {"lat": la + lat_sz/2, "lon": lo + lon_sz/2,
         "box": box(lo, la, lo+lon_sz, la+lat_sz)}
        for la in all_lats for lo in all_lons
        if eff_sector.intersects(box(lo, la, lo+lon_sz, la+lat_sz))
    ]
    total = len(mesh_boxes)
    if total == 0:
        return pd.DataFrame(), eff_sector, 0

    # ── 建物 sindex (LOS用) ────────────────────────────────────────────
    if buildings_gdf is not None and not buildings_gdf.empty:
        bldgs  = buildings_gdf[
            buildings_gdf.geometry.intersects(sector.buffer(0.00005))
        ].copy()
        sindex = bldgs.sindex if not bldgs.empty else None
    else:
        bldgs  = None
        sindex = None

    mesh_area = lat_sz * lon_sz  # 全メッシュ面積は共通

    visible_rows = []
    for i, m in enumerate(mesh_boxes):
        if progress_cb and i % 50 == 0:
            progress_cb(i / total, f"{sid}: メッシュ判定 {i}/{total}")

        mesh_box = m["box"]

        # ── 面積判定: メッシュ面積の80%以上が有効扇形内 ────────────────
        inter = eff_sector.intersection(mesh_box)
        if inter.is_empty:
            continue
        area_ratio = inter.area / mesh_area
        if area_ratio < VISIBLE_RATIO_THRESHOLD:
            continue

        # ── 建物LOSチェック: 重心点が完全遮蔽なら除外 ────────────────
        if bldgs is not None and sindex is not None:
            cx, cy = inter.centroid.x, inter.centroid.y
            ray    = LineString([(lon, lat), (cx, cy)])
            cands  = bldgs.iloc[list(sindex.intersection(ray.bounds))]
            if _is_blocked(lon, lat, h, cx, cy, EYE_HEIGHT_M,
                           cands, lat_sc, lon_sc):
                continue

        code   = encode_mesh10(m["lat"], m["lon"])
        dx_m   = (m["lon"] - lon) * lon_sc
        dy_m   = (m["lat"] - lat) * lat_sc
        dist_m = math.sqrt(dx_m**2 + dy_m**2)
        visible_rows.append({
            "billboard_id": sid,
            "mesh_code":    code,
            "center_lat":   round(m["lat"], 8),
            "center_lon":   round(m["lon"], 8),
            "distance_m":   round(dist_m, 1),
            "area_ratio":   round(area_ratio, 3),
        })

    return pd.DataFrame(visible_rows), eff_sector, total


# ─────────────────────────────────────────────────────────────────────────────
# 地図生成
# ─────────────────────────────────────────────────────────────────────────────

def build_map(billboards: list, sectors: list, visible_dfs: list,
              buildings_gdf: Optional[gpd.GeoDataFrame],
              mesh_colors: Optional[dict] = None) -> go.Figure:
    fig = go.Figure()

    if buildings_gdf is not None and not buildings_gdf.empty:
        # 各面板のフル扇形の合計エリア内の建物のみプロット
        _map_area = None
        for _bb in billboards:
            _s = create_sector(
                _bb["latitude"], _bb["longitude"],
                _bb["facing_deg"], _bb.get("max_range_m", 500.0),
            )
            _map_area = _s if _map_area is None else _map_area.union(_s)
        if _map_area is not None:
            buildings_gdf = buildings_gdf[
                buildings_gdf.geometry.intersects(_map_area.buffer(0.00005))
            ]

    if buildings_gdf is not None and not buildings_gdf.empty:
        # 建物を高さ5段階に分け各グループを1トレースに集約 (緑→赤グラデーション)
        max_h = buildings_gdf["height"].quantile(0.95) or 1
        bins  = [0, max_h*0.2, max_h*0.4, max_h*0.6, max_h*0.8, float("inf")]
        tier_fill = [
            "rgba(40,200,80,0.50)",    # 低  → 緑
            "rgba(160,220,0,0.50)",    #     → 黄緑
            "rgba(255,210,0,0.50)",    # 中  → 黄
            "rgba(255,110,0,0.55)",    #     → オレンジ
            "rgba(220,30,30,0.60)",    # 高  → 赤
        ]
        tier_line = [
            "rgba(20,150,50,0.8)",
            "rgba(100,170,0,0.8)",
            "rgba(200,160,0,0.8)",
            "rgba(200,70,0,0.8)",
            "rgba(170,0,0,0.8)",
        ]
        labels = [
            f"🟢 建物 〜{max_h*0.2:.0f}m（低）",
            f"🟡 建物 〜{max_h*0.4:.0f}m",
            f"🟡 建物 〜{max_h*0.6:.0f}m",
            f"🟠 建物 〜{max_h*0.8:.0f}m",
            f"🔴 建物 {max_h*0.8:.0f}m〜（高）",
        ]

        for tier in range(5):
            lo_h, hi_h = bins[tier], bins[tier + 1]
            subset = buildings_gdf[
                (buildings_gdf["height"] > lo_h) & (buildings_gdf["height"] <= hi_h)
            ]
            if subset.empty:
                continue

            all_lons: list = []
            all_lats: list = []
            for geom in subset["geometry"]:
                polys = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
                for poly in polys:
                    if poly.geom_type != "Polygon":
                        continue
                    xs, ys = poly.exterior.xy
                    all_lons.extend(list(xs) + [None])
                    all_lats.extend(list(ys) + [None])

            if not all_lons:
                continue

            fig.add_trace(go.Scattermapbox(
                lat=all_lats, lon=all_lons,
                mode="lines", fill="toself",
                fillcolor=tier_fill[tier],
                line=dict(color=tier_line[tier], width=0.5),
                name=labels[tier],
                hoverinfo="skip",
                showlegend=True,
            ))

    lat_sz, lon_sz = mesh10_cell_size()

    for idx, (bb, sector, vdf) in enumerate(zip(billboards, sectors, visible_dfs)):
        color = COLORS[idx % len(COLORS)]
        sid   = bb.get("screen_id", f"B{idx+1}")

        xs, ys = sector.exterior.xy
        fig.add_trace(go.Scattermapbox(
            lat=list(ys), lon=list(xs),
            mode="lines", fill="toself",
            fillcolor="rgba(255,220,0,0.08)",
            line=dict(color=color, width=1.5),
            name=f"{sid} 扇形エリア",
            hoverinfo="skip",
        ))

        # 有効メッシュ: 中心点ではなく実寸のメッシュ矩形ポリゴンで描画
        # 建物の緑〜赤と被らないよう青系の色を使用
        if not vdf.empty:
            box_lats: list = []
            box_lons: list = []
            box_texts: list = []
            for _, row in vdf.iterrows():
                la0 = row["center_lat"] - lat_sz / 2
                lo0 = row["center_lon"] - lon_sz / 2
                txt = (f"{row['mesh_code']}<br>"
                       f"距離: {row['distance_m']}m<br>"
                       f"扇形内面積比: {row['area_ratio']*100:.1f}%")
                # SW→SE→NE→NW→SW の順で閉じる
                box_lats.extend([la0,          la0,          la0+lat_sz, la0+lat_sz, la0,          None])
                box_lons.extend([lo0,          lo0+lon_sz,   lo0+lon_sz, lo0,        lo0,          None])
                box_texts.extend([txt, txt, txt, txt, txt, ""])

            _mc = mesh_colors.get(sid) if mesh_colors else None
            _fc = _hex_to_rgba(_mc, 0.45) if _mc else "rgba(30,130,255,0.45)"
            _lc = _hex_to_rgba(_mc, 0.85) if _mc else "rgba(0,70,210,0.85)"
            fig.add_trace(go.Scattermapbox(
                lat=box_lats, lon=box_lons,
                mode="lines", fill="toself",
                fillcolor=_fc,
                line=dict(color=_lc, width=1),
                name=f"● {sid} 有効メッシュ ({len(vdf):,}件)",
                text=box_texts,
                hovertemplate="%{text}<extra></extra>",
            ))


        lat, lon = bb["latitude"], bb["longitude"]
        facing   = bb["facing_deg"]
        lat_sc, lon_sc = local_scale(lat)
        arr_lat = lat + 40.0 * math.sin(math.radians(facing)) / lat_sc
        arr_lon = lon + 40.0 * math.cos(math.radians(90.0 - facing)) / lon_sc

        fig.add_trace(go.Scattermapbox(
            lat=[lat, arr_lat], lon=[lon, arr_lon],
            mode="lines", line=dict(color=color, width=4),
            name=f"{sid} 面向方向", hoverinfo="skip",
        ))
        fig.add_trace(go.Scattermapbox(
            lat=[lat], lon=[lon],
            mode="markers",
            marker=dict(size=14, color=color, symbol="circle"),
            name=f"{sid} ({bb['height_m']}m / {facing}°)",
            hovertemplate=(f"<b>{sid}</b><br>高さ: {bb['height_m']}m<br>"
                           f"方位: {facing}°<extra></extra>"),
        ))

    center_lat = np.mean([bb["latitude"]  for bb in billboards])
    center_lon = np.mean([bb["longitude"] for bb in billboards])
    fig.update_layout(
        mapbox=dict(style="open-street-map",
                    center=dict(lat=center_lat, lon=center_lon), zoom=16),
        height=680,
        margin=dict(r=0, t=0, l=0, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.88)"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="広告視認エリア計算", page_icon="👁️", layout="wide")

# ── パスワード認証 ──────────────────────────────────────────────────────────
def _check_password():
    if st.session_state.get("authenticated"):
        return
    st.title("🔒 ログインが必要です")
    st.caption("このアプリを利用するにはパスワードが必要です。")
    pwd = st.text_input("パスワード", type="password", key="pwd_input")
    if st.button("ログイン", type="primary"):
        if pwd == st.secrets.get("APP_PASSWORD", ""):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("パスワードが違います。")
    st.stop()

_check_password()
# ── 認証済みユーザーのみここから表示 ────────────────────────────────────────

st.title("👁️ 広告面板 視認エリア計算アプリ")
st.caption("Plateau CityGML × 10次メッシュ LOS 解析")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ データ入力")
    st.divider()

    # ① 広告面板データ
    st.subheader("① 広告面板データ")
    bb_input_mode = st.radio(
        "入力方法",
        ["📂 CSVアップロード", "✏️ 手入力（1面のみ）"],
        key="bb_input_mode",
        horizontal=True,
    )

    bb_file   = None
    manual_bb = None

    if bb_input_mode == "📂 CSVアップロード":
        st.markdown("""
**必須列**: `screen_id`, `latitude`, `longitude`, `height_m`, `facing_deg`, `panel_h_m`, `panel_w_m`

最大視認距離は `panel_h_m × panel_w_m × 7` で自動計算されます。
""")
        bb_file = st.file_uploader("CSVをアップロード", type=["csv"], key="bb_csv")
    else:
        st.caption("最大1面の情報を入力してください。Screen IDは6桁の番号（例: 000001）にしてください。")
        with st.form("manual_bb_form"):
            sid_raw = st.text_input("Screen ID（6桁）", placeholder="例: 000001")
            m_lat   = st.number_input("緯度 latitude",           value=35.6812,  format="%.6f")
            m_lon   = st.number_input("経度 longitude",          value=139.7671, format="%.6f")
            m_h     = st.number_input("高さ height_m (m)",       value=10.0,  min_value=0.1,  step=0.5)
            m_f     = st.number_input("面向角度 facing_deg (°)", value=180.0, min_value=0.0,  max_value=359.9, step=1.0)
            m_ph    = st.number_input("面の縦 panel_h_m (m)", value=3.0, min_value=0.1, step=0.1)
            m_pw    = st.number_input("面の横 panel_w_m (m)", value=6.0, min_value=0.1, step=0.1)
            submitted = st.form_submit_button("✅ 設定を反映", use_container_width=True)

        if submitted:
            sid_clean = str(sid_raw).strip()
            if len(sid_clean) == 0:
                st.warning("⚠️ Screen IDを入力してください。")
            elif len(sid_clean) < 6:
                st.warning(
                    f"⚠️ Screen IDは6桁にしてください。"
                    f"「**{sid_clean.zfill(6)}**」のように頭に0を付けて入力してください。"
                )
            else:
                st.session_state["manual_bb"] = {
                    "screen_id":   sid_clean,
                    "latitude":    m_lat,
                    "longitude":   m_lon,
                    "height_m":    m_h,
                    "facing_deg":  m_f,
                    "panel_h_m":   m_ph,
                    "panel_w_m":   m_pw,
                    "max_range_m": round(m_ph * m_pw * 7, 1),
                }

        if st.session_state.get("manual_bb"):
            manual_bb = st.session_state["manual_bb"]
            d = manual_bb
            st.success(
                f"✅ **{d['screen_id']}** 設定済み  \n"
                f"緯度 {d['latitude']:.5f} / 経度 {d['longitude']:.5f}  \n"
                f"高さ {d['height_m']}m｜方位 {d['facing_deg']}°  \n"
                f"面サイズ {d['panel_h_m']}×{d['panel_w_m']}m → 最大視認距離 {d['max_range_m']}m"
            )

    st.divider()

    # ② 建物データ
    st.subheader("② 建物データ（CityGML）")
    bldg_mode = st.radio(
        "取得方法",
        ["🚀 Plateau から自動取得", "📂 手動アップロード", "⛔ 使用しない"],
        help=(
            "自動取得: 広告面板の位置から必要な建物データをネット経由で自動ダウンロード\n"
            "手動: .gml ファイルをアップロード\n"
            "使用しない: 建物遮蔽なし（扇形全体を視認エリアとして計算）"
        ),
    )

    gml_file  = None
    fetch_btn = False

    if bldg_mode == "📂 手動アップロード":
        gml_file = st.file_uploader(
            "CityGML (.gml) をアップロード", type=["gml", "xml"], key="gml"
        )
    elif bldg_mode == "🚀 Plateau から自動取得":
        st.caption(
            "広告面板 CSV をアップロード後、ボタンで Plateau の建物データを自動ダウンロードします。"
            "インターネット接続が必要です。"
        )
        fetch_btn = st.button(
            "🏢 建物データを自動取得",
            disabled=(bb_file is None and not manual_bb),
            use_container_width=True,
            type="secondary",
        )
    else:
        st.caption("建物遮蔽なしで計算します。扇形内のメッシュすべてが有効になります。")

    st.divider()

    # ③ 方位角ガイド
    st.subheader("③ 方位角（facing_deg）の入力方法")
    st.markdown("""
| 方角 | 角度 |
|------|------|
| 北 (N) | **0°** |
| 北東 (NE) | **45°** |
| 東 (E) | **90°** |
| 南東 (SE) | **135°** |
| 南 (S) | **180°** |
| 南西 (SW) | **225°** |
| 西 (W) | **270°** |
| 北西 (NW) | **315°** |

> 地図上の**矢印**で方向を確認してください
""")

    st.divider()
    _has_input = (bb_file is not None) or bool(manual_bb)
    run_btn = st.button(
        "▶ 計算実行", type="primary", use_container_width=True,
        disabled=not _has_input,
    )

# ── Main ─────────────────────────────────────────────────────────────────────

_csv_mode = (bb_input_mode == "📂 CSVアップロード")

# 入力チェック
if (_csv_mode and bb_file is None) or (not _csv_mode and not manual_bb):
    st.info("👈 左のサイドバーから広告面板データを入力してください。")
    st.stop()

# bb_df の構築
if _csv_mode:
    try:
        bb_df = pd.read_csv(bb_file)
        required = {"screen_id", "latitude", "longitude", "height_m", "facing_deg",
                    "panel_h_m", "panel_w_m"}
        missing  = required - set(bb_df.columns)
        if missing:
            st.error(f"CSV に必要な列がありません: {missing}")
            st.stop()
        bb_df["max_range_m"] = (bb_df["panel_h_m"] * bb_df["panel_w_m"] * 7).round(1)
    except Exception as e:
        st.error(f"CSV 読み込みエラー: {e}")
        st.stop()
else:
    bb_df = pd.DataFrame([manual_bb])

st.success(f"広告面板 {len(bb_df)} 件を読み込みました")
st.dataframe(bb_df, use_container_width=True)

# ── 位置補正用: corrected_coords 初期化 (入力データが変わったらリセット) ──────
_src_sig = bb_df[["screen_id", "latitude", "longitude"]].to_csv(index=False)
if st.session_state.get("_corr_src") != _src_sig:
    st.session_state["_corr_src"] = _src_sig
    st.session_state["corrected_coords"] = {
        str(r["screen_id"]): {"latitude": float(r["latitude"]),
                               "longitude": float(r["longitude"])}
        for _, r in bb_df.iterrows()
    }
    if "finalized_master" in st.session_state:
        del st.session_state["finalized_master"]

# 補正座標を適用した作業用 DataFrame
_corr = st.session_state["corrected_coords"]
bb_df_w = bb_df.copy()
for _ci, _cr in bb_df_w.iterrows():
    _csid = str(_cr["screen_id"])
    if _csid in _corr:
        bb_df_w.at[_ci, "latitude"]  = _corr[_csid]["latitude"]
        bb_df_w.at[_ci, "longitude"] = _corr[_csid]["longitude"]
bb_df_w["max_range_m"] = (bb_df_w["panel_h_m"] * bb_df_w["panel_w_m"] * 7).round(1)

# 建物データ（手動アップロード）
if gml_file is not None:
    with st.spinner("CityGML を解析中..."):
        try:
            bldgs = parse_citygml(gml_file.read())
            if bldgs.empty:
                st.warning("CityGML から建物データを抽出できませんでした。")
            else:
                st.success(f"建物 {len(bldgs):,} 棟を読み込みました "
                           f"（高さ平均 {bldgs['height'].mean():.1f}m）")
                st.session_state["buildings_gdf"] = bldgs
        except Exception as e:
            st.error(f"CityGML 解析エラー: {e}")

# 建物データ（自動取得）
if fetch_btn:
    st.subheader("🏢 建物データ自動取得ログ")
    log_box = st.empty()
    with st.spinner("Plateau から建物データを取得中..."):
        bldgs = auto_fetch_citygml(bb_df_w, log_box)
    if bldgs is not None:
        st.session_state["buildings_gdf"] = bldgs
        st.success(f"✅ 建物 {len(bldgs):,} 棟の取得が完了しました")
    else:
        st.error("建物データの自動取得に失敗しました。手動アップロードをお試しください。")

# 建物データの状態表示
buildings_gdf = st.session_state.get("buildings_gdf") if bldg_mode != "⛔ 使用しない" else None
if bldg_mode == "🚀 Plateau から自動取得" and buildings_gdf is not None:
    st.info(f"🏢 取得済み建物データ: {len(buildings_gdf):,} 棟（高さ平均 {buildings_gdf['height'].mean():.1f}m）")

st.divider()

# プレビューマップ
st.subheader("📍 設定確認マップ（方向矢印を確認してください）")
prev_fig = go.Figure()
for idx, row in bb_df_w.iterrows():
    color  = COLORS[idx % len(COLORS)]
    sector = create_sector(row.latitude, row.longitude, row.facing_deg, row.max_range_m)
    # 有効扇形（デッドゾーン除外）
    lat_sc_p, lon_sc_p = local_scale(row.latitude)
    dead_r_deg_p = row.height_m / ((lat_sc_p + lon_sc_p) / 2)
    dead_zone_p  = Point(row.longitude, row.latitude).buffer(dead_r_deg_p)
    eff_sector_p = sector.difference(dead_zone_p)
    # 有効扇形ポリゴンを描画（複数ポリゴンになる場合も対応）
    polys_p = list(eff_sector_p.geoms) if eff_sector_p.geom_type.startswith("Multi") else [eff_sector_p]
    for pi, poly in enumerate(polys_p):
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        xs, ys = poly.exterior.xy
        prev_fig.add_trace(go.Scattermapbox(
            lat=list(ys), lon=list(xs), mode="lines", fill="toself",
            fillcolor="rgba(255,200,0,0.15)", line=dict(color=color, width=1.5),
            name=f"{row.screen_id} 有効扇形" if pi == 0 else f"{row.screen_id} 有効扇形_{pi}",
            hoverinfo="skip",
        ))
    lat_sc, lon_sc = local_scale(row.latitude)
    arr_lat = row.latitude  + 40.0 * math.sin(math.radians(row.facing_deg)) / lat_sc
    arr_lon = row.longitude + 40.0 * math.cos(math.radians(90.0 - row.facing_deg)) / lon_sc
    prev_fig.add_trace(go.Scattermapbox(
        lat=[row.latitude, arr_lat], lon=[row.longitude, arr_lon],
        mode="lines", line=dict(color=color, width=4),
        name=f"{row.screen_id} 方向矢印", hoverinfo="skip",
    ))
    prev_fig.add_trace(go.Scattermapbox(
        lat=[row.latitude], lon=[row.longitude], mode="markers",
        marker=dict(size=13, color=color),
        name=str(row.screen_id),
        hovertemplate=f"<b>{row.screen_id}</b><br>高さ: {row.height_m}m<br>方位: {row.facing_deg}°<extra></extra>",
    ))

center_lat = bb_df_w["latitude"].mean()
center_lon = bb_df_w["longitude"].mean()
prev_fig.update_layout(
    mapbox=dict(style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon), zoom=16),
    height=420, margin=dict(r=0, t=0, l=0, b=0),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                bgcolor="rgba(255,255,255,0.88)"),
)
st.plotly_chart(prev_fig, use_container_width=True)
st.caption("▲ 矢印が広告面板の面向方向です。下の補正マップで位置を調整し「最終確定」後に計算してください。")

# ── 位置補正マップ ────────────────────────────────────────────────────────────
st.divider()
st.subheader("✏️ 位置補正マップ")

if not _FOLIUM_OK:
    st.warning("folium / streamlit-folium が未インストールです。`pip install folium streamlit-folium` を実行してください。")
else:
    _sids_all = [str(r["screen_id"]) for _, r in bb_df_w.iterrows()]

    _cc_map, _cc_ctrl = st.columns([3, 2])

    with _cc_ctrl:
        _sel = st.selectbox(
            "移動する面を選択",
            _sids_all,
            key="corr_select",
            help="選択した面のマーカーが赤くなります。地図上をクリックして新しい位置を指定してください。",
        )

    # Folium マップ構築
    _fm = folium.Map(
        location=[bb_df_w["latitude"].mean(), bb_df_w["longitude"].mean()],
        zoom_start=16,
        tiles="OpenStreetMap",
    )
    for _fi, _fr in bb_df_w.iterrows():
        _fsid  = str(_fr["screen_id"])
        _flat  = _corr[_fsid]["latitude"]
        _flon  = _corr[_fsid]["longitude"]
        _fcolor = "red" if _fsid == _sel else "blue"
        # 扇形
        _fsec  = create_sector(_flat, _flon, _fr["facing_deg"], _fr["max_range_m"])
        folium.Polygon(
            locations=[[p[1], p[0]] for p in _fsec.exterior.coords],
            color=COLORS[_fi % len(COLORS)],
            fill=True, fill_opacity=0.10, weight=2,
            tooltip=f"{_fsid} 視認扇形",
        ).add_to(_fm)
        # マーカー
        folium.Marker(
            location=[_flat, _flon],
            popup=_fsid,
            tooltip=f"{_fsid}（{_flat:.5f}, {_flon:.5f}）",
            icon=folium.Icon(color=_fcolor, icon="flag"),
        ).add_to(_fm)

    with _cc_map:
        _map_res = st_folium(
            _fm,
            key="corr_folium",
            height=430,
            use_container_width=True,
            returned_objects=["last_clicked"],
        )

    with _cc_ctrl:
        _cur = _corr[_sel]
        # 元データと比較して変更検知
        _orig_row = bb_df[bb_df["screen_id"].astype(str) == _sel].iloc[0]
        _is_moved = (
            abs(_cur["latitude"]  - float(_orig_row["latitude"]))  > 1e-7 or
            abs(_cur["longitude"] - float(_orig_row["longitude"])) > 1e-7
        )
        _status_icon = "✏️" if _is_moved else "📍"
        st.markdown(
            f"**{_status_icon} {_sel} 現在位置**  \n"
            f"緯度: `{_cur['latitude']:.6f}`  \n"
            f"経度: `{_cur['longitude']:.6f}`"
        )

        # クリック位置の処理
        if _map_res and _map_res.get("last_clicked"):
            _clk_lat = round(_map_res["last_clicked"]["lat"], 7)
            _clk_lon = round(_map_res["last_clicked"]["lng"], 7)
            _already = (
                abs(_cur["latitude"]  - _clk_lat) < 1e-6 and
                abs(_cur["longitude"] - _clk_lon) < 1e-6
            )
            if _already:
                st.success(f"✓ この位置に移動済みです")
            else:
                st.info(
                    f"🔵 **クリック位置**  \n"
                    f"緯度: `{_clk_lat}`  \n"
                    f"経度: `{_clk_lon}`"
                )
                if st.button(
                    f"▶ {_sel} をこの位置に移動",
                    key="apply_corr", type="secondary", use_container_width=True,
                ):
                    st.session_state["corrected_coords"][_sel] = {
                        "latitude": _clk_lat, "longitude": _clk_lon,
                    }
                    if "finalized_master" in st.session_state:
                        del st.session_state["finalized_master"]
                    st.rerun()
        else:
            st.caption("地図上をクリックすると新しい位置を指定できます")

        # 個別リセット
        if _is_moved:
            if st.button(f"↩ {_sel} の位置をリセット", key="reset_corr", use_container_width=True):
                st.session_state["corrected_coords"][_sel] = {
                    "latitude":  float(_orig_row["latitude"]),
                    "longitude": float(_orig_row["longitude"]),
                }
                if "finalized_master" in st.session_state:
                    del st.session_state["finalized_master"]
                st.rerun()

        st.divider()
        st.markdown("**補正状況**")
        for _ss, _sp in _corr.items():
            _or = bb_df[bb_df["screen_id"].astype(str) == _ss].iloc[0]
            _mv = (abs(_sp["latitude"]  - float(_or["latitude"]))  > 1e-7 or
                   abs(_sp["longitude"] - float(_or["longitude"])) > 1e-7)
            st.caption(f"{'✏️' if _mv else '📍'} **{_ss}**: {_sp['latitude']:.5f}, {_sp['longitude']:.5f}")

# ── 最終確定 ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader("✅ 最終確定")
_fin_c1, _fin_c2 = st.columns([1, 1])

with _fin_c1:
    if st.button("✅ 位置を最終確定する", type="primary", key="finalize_btn", use_container_width=True):
        _fdf = bb_df.copy()
        for _fi2, _fr2 in _fdf.iterrows():
            _fs = str(_fr2["screen_id"])
            _fc = st.session_state.get("corrected_coords", {}).get(_fs)
            if _fc:
                _fdf.at[_fi2, "latitude"]  = _fc["latitude"]
                _fdf.at[_fi2, "longitude"] = _fc["longitude"]
        _fdf["max_range_m"] = (_fdf["panel_h_m"] * _fdf["panel_w_m"] * 7).round(1)
        st.session_state["finalized_master"] = _fdf

if "finalized_master" in st.session_state:
    _fmdf = st.session_state["finalized_master"]
    _out_cols = [c for c in ["screen_id", "latitude", "longitude", "height_m",
                              "facing_deg", "panel_h_m", "panel_w_m"] if c in _fmdf.columns]
    with _fin_c2:
        _csv_out = _fmdf[_out_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ 補正後マスターCSVをダウンロード",
            _csv_out, "corrected_master.csv", "text/csv",
            type="primary", use_container_width=True,
            key="dl_corrected_master",
        )
    st.success("✅ 確定済み。以下のデータで計算を実行します。")
    st.dataframe(_fmdf[_out_cols], use_container_width=True)

# ── 計算実行 ─────────────────────────────────────────────────────────────────
if run_btn:
    prog_bar   = st.progress(0, text="計算を開始しています...")
    all_visible = []
    all_sectors = []

    for i, row in bb_df_w.iterrows():
        bb = row.to_dict()

        def progress_cb(frac, text, _i=i):
            prog_bar.progress((_i + frac) / len(bb_df_w), text=text)

        vdf, sector, total = compute_visibility(bb, buildings_gdf, progress_cb)
        all_visible.append(vdf)
        all_sectors.append(sector)

    prog_bar.progress(1.0, text="完了！")

    result_df = (
        pd.concat(all_visible, ignore_index=True)
        if any(not v.empty for v in all_visible)
        else pd.DataFrame()
    )
    st.session_state["result_df"]     = result_df
    st.session_state["all_visible"]   = all_visible
    st.session_state["all_sectors"]   = all_sectors
    st.session_state["bb_list"]       = bb_df_w.to_dict("records")
    st.session_state["buildings_calc"]= buildings_gdf

# ── 結果表示 ─────────────────────────────────────────────────────────────────
if "result_df" in st.session_state:
    result_df     = st.session_state["result_df"]
    all_visible   = st.session_state["all_visible"]
    all_sectors   = st.session_state["all_sectors"]
    bb_list       = st.session_state["bb_list"]
    buildings_calc= st.session_state["buildings_calc"]

    st.divider()
    st.subheader("📊 計算結果")

    cols = st.columns(max(len(bb_list), 1))
    for idx, (bb, vdf) in enumerate(zip(bb_list, all_visible)):
        with cols[idx % len(cols)]:
            st.metric(
                label=str(bb["screen_id"]),
                value=f"{len(vdf):,} メッシュ",
                help=f"方位 {bb['facing_deg']}° / 高さ {bb['height_m']}m / 面 {bb.get('panel_h_m','?')}×{bb.get('panel_w_m','?')}m / 最大視認距離 {bb.get('max_range_m','?')}m",
            )

    if not result_df.empty:
        st.dataframe(result_df, use_container_width=True, height=280)

    st.divider()
    st.subheader("🗺️ 視認エリアマップ")
    st.caption("🟢→🔴 建物（低→高）　● 有効メッシュ矩形（扇形内面積比≥80%）　🟡 有効扇形（デッドゾーン除外）　→ 方向矢印")

    # ── 地図表示設定（フィルター + メッシュ色） ──────────────────────────
    with st.expander("🎛️ 地図表示設定", expanded=True):
        _n_bb  = len(bb_list)
        _fcols = st.columns(min(_n_bb, 4))
        _show  = {}
        _mcols = {}
        for _i, _bb in enumerate(bb_list):
            _s = str(_bb["screen_id"])
            with _fcols[_i % min(_n_bb, 4)]:
                _show[_s]  = st.checkbox(
                    f"表示: {_s}", value=True, key=f"show_{_s}"
                )
                _mcols[_s] = st.color_picker(
                    f"メッシュ色: {_s}",
                    value=COLORS[_i % len(COLORS)],
                    key=f"meshcol_{_s}",
                )

    # フィルタ適用
    _fbb  = [bb  for bb       in bb_list    if _show.get(str(bb["screen_id"]),  True)]
    _fvis = [vdf for bb, vdf  in zip(bb_list, all_visible) if _show.get(str(bb["screen_id"]), True)]
    _fsec = [sec for bb, sec  in zip(bb_list, all_sectors) if _show.get(str(bb["screen_id"]), True)]

    with st.spinner("地図を生成中..."):
        fig = build_map(_fbb, _fsec, _fvis, buildings_calc, mesh_colors=_mcols)
    st.plotly_chart(fig, use_container_width=True)

    # ── 視線遮蔽建物の除外補正 ────────────────────────────────────────────────
    st.divider()
    st.subheader("🚫 視線遮蔽建物の除外補正")

    _excl_key = "excluded_bldg_indices"
    if _excl_key not in st.session_state:
        st.session_state[_excl_key] = set()
    _excl_set: set = st.session_state[_excl_key]

    _excl_mode = st.session_state.get("exclusion_mode", False)

    _ec1, _ec2, _ec3 = st.columns([2, 2, 2])
    with _ec1:
        if st.button(
            "✏️ 除外建物を選択する" if not _excl_mode else "✅ 選択モードを閉じる",
            key="excl_mode_toggle",
            use_container_width=True,
        ):
            st.session_state["exclusion_mode"] = not _excl_mode
            st.rerun()
    if _excl_set:
        with _ec2:
            st.info(f"除外済み建物: {len(_excl_set)} 棟")
        with _ec3:
            if st.button("🗑️ 除外リストをクリア", key="clear_excl", use_container_width=True):
                st.session_state[_excl_key] = set()
                st.rerun()

    if _excl_mode:
        if buildings_calc is None or buildings_calc.empty:
            st.warning("建物データがありません。建物データありで計算した場合のみ除外補正が使用できます。")
        elif not _FOLIUM_OK:
            st.warning("folium / streamlit-folium が未インストールです。`pip install folium streamlit-folium` を実行してください。")
        else:
            # bb_listからフル扇形を再生成して建物を絞り込み
            _bldg_idx_in_area: set = set()
            _full_sectors_excl = []
            for _ebb in bb_list:
                _fs = create_sector(
                    _ebb["latitude"], _ebb["longitude"],
                    _ebb["facing_deg"], _ebb.get("max_range_m", 500.0),
                )
                _full_sectors_excl.append(_fs)
                _hits = buildings_calc[
                    buildings_calc.geometry.intersects(_fs.buffer(0.00005))
                ].index.tolist()
                _bldg_idx_in_area.update(_hits)
            _bldgs_in_area = buildings_calc.loc[sorted(_bldg_idx_in_area)].copy()

            st.caption(
                f"対象エリア内の建物: {len(_bldgs_in_area):,} 棟　｜　"
                "🔴 赤 = 除外済み（クリックで復活）　🔵 青 = 計算対象（クリックで除外）"
            )

            _center_lat = np.mean([_b["latitude"]  for _b in bb_list])
            _center_lon = np.mean([_b["longitude"] for _b in bb_list])
            _efm = folium.Map(
                location=[_center_lat, _center_lon], zoom_start=17, tiles="OpenStreetMap"
            )

            # 扇形エリアを薄く表示（フル扇形を使用）
            for _ebb, _fsec_e in zip(bb_list, _full_sectors_excl):
                _fpolys = (
                    list(_fsec_e.geoms)
                    if _fsec_e.geom_type.startswith("Multi")
                    else [_fsec_e]
                )
                for _fpoly in _fpolys:
                    if _fpoly.geom_type != "Polygon":
                        continue
                    folium.Polygon(
                        locations=[[p[1], p[0]] for p in _fpoly.exterior.coords],
                        color="gold", fill=True, fill_opacity=0.05, weight=1.5,
                        tooltip=f"{_ebb['screen_id']} 扇形エリア",
                    ).add_to(_efm)

            # 面板マーカー
            for _ebb in bb_list:
                folium.Marker(
                    location=[_ebb["latitude"], _ebb["longitude"]],
                    tooltip=str(_ebb["screen_id"]),
                    icon=folium.Icon(color="orange", icon="flag"),
                ).add_to(_efm)

            # 建物ポリゴン: GeoJson一括描画（ジオメトリ簡略化 + 単一レイヤー）
            _excl_str = {str(x) for x in _excl_set}
            _bldgs_in_area["_idx"] = _bldgs_in_area.index.astype(str)
            _bldgs_in_area["geometry"] = _bldgs_in_area["geometry"].simplify(
                0.00003, preserve_topology=True
            )
            folium.GeoJson(
                _bldgs_in_area[["geometry", "_idx", "height"]],
                style_function=lambda f, _es=_excl_str: {
                    "fillColor":   "red"     if f["properties"]["_idx"] in _es else "#2196f3",
                    "color":       "#cc0000" if f["properties"]["_idx"] in _es else "#0050b0",
                    "weight":      2         if f["properties"]["_idx"] in _es else 1,
                    "fillOpacity": 0.6       if f["properties"]["_idx"] in _es else 0.3,
                },
                # tooltip は _idx のみ（last_object_clicked_tooltip でパース）
                tooltip=folium.GeoJsonTooltip(
                    fields=["_idx"],
                    aliases=[""],
                    labels=False,
                ),
                popup=folium.GeoJsonPopup(
                    fields=["_idx", "height"],
                    aliases=["建物ID:", "高さ(m):"],
                ),
                name="buildings",
            ).add_to(_efm)

            _emap_res = st_folium(
                _efm,
                key="excl_folium",
                height=540,
                use_container_width=True,
                returned_objects=["last_object_clicked_tooltip"],
            )

            # クリックされた建物を除外セットに追加 / 解除
            if _emap_res and _emap_res.get("last_object_clicked_tooltip"):
                _tip = str(_emap_res["last_object_clicked_tooltip"]).strip()
                # tooltipは _idx の数値文字列のみ
                try:
                    _clicked_idx = int(_tip)
                    if _clicked_idx in _bldg_idx_in_area:
                        _new_excl = set(st.session_state.get(_excl_key, set()))
                        if _clicked_idx in _new_excl:
                            _new_excl.discard(_clicked_idx)
                        else:
                            _new_excl.add(_clicked_idx)
                        st.session_state[_excl_key] = _new_excl
                        st.rerun()
                except (ValueError, TypeError):
                    pass

    # 除外設定がある場合は再計算ボタンを表示
    if _excl_set:
        st.divider()
        if st.button(
            f"🔄 除外設定（{len(_excl_set)} 棟）で再計算する",
            type="primary",
            key="recalc_excl_btn",
            use_container_width=False,
        ):
            _filtered_bldgs = (
                buildings_calc[~buildings_calc.index.isin(_excl_set)].copy()
                if buildings_calc is not None
                else None
            )
            _rprog = st.progress(0, text="再計算中...")
            _new_visible_r: list = []
            _new_sectors_r: list = []
            for _ri, _rbb in enumerate(bb_list):
                def _rcb(frac, text, _i=_ri):
                    _rprog.progress((_i + frac) / len(bb_list), text=text)
                _rvdf, _rsec, _ = compute_visibility(_rbb, _filtered_bldgs, _rcb)
                _new_visible_r.append(_rvdf)
                _new_sectors_r.append(_rsec)
            _rprog.progress(1.0, text="再計算完了！")
            st.session_state["result_df"] = (
                pd.concat(_new_visible_r, ignore_index=True)
                if any(not _v.empty for _v in _new_visible_r)
                else pd.DataFrame()
            )
            st.session_state["all_visible"]    = _new_visible_r
            st.session_state["all_sectors"]    = _new_sectors_r
            st.session_state["exclusion_mode"] = False
            st.rerun()

    # ── メッシュコード CSV ダウンロード ───────────────────────────────────────
    st.divider()
    st.subheader("⬇️ メッシュコード CSV ダウンロード（面別）")
    st.caption("各ファイルはメッシュコードのみ・ヘッダーなし。ファイル名は No.Screen ID です。")
    _dl_any = False
    _dl_cols = st.columns(min(len(bb_list), 4))
    for _i, (_bb, _vdf) in enumerate(zip(bb_list, all_visible)):
        _sid = str(_bb["screen_id"])
        if not _vdf.empty:
            _dl_any = True
            _mesh_csv = _vdf["mesh_code"].to_csv(index=False, header=False).encode("utf-8")
            with _dl_cols[_i % min(len(bb_list), 4)]:
                st.download_button(
                    label=f"⬇️ No.{_sid}.csv ({len(_vdf):,}件)",
                    data=_mesh_csv,
                    file_name=f"No.{_sid}.csv",
                    mime="text/csv",
                    key=f"dl_{_sid}_{_i}",
                    type="primary",
                    use_container_width=True,
                )
    if not _dl_any:
        st.warning("有効メッシュが 0 件でした。設定を見直してください。")
