"""
広告面板 円形視認エリア計算アプリ
Plateau CityGML + 10次メッシュ LOS可視化（円形エリア版）
"""
import io
import math
import hashlib
import warnings
import re
import struct
import zlib
import zipfile
import urllib.request
import json as _json
from concurrent.futures import ThreadPoolExecutor, as_completed
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
EYE_HEIGHT_M             = 1.5
SAMPLE_N                 = 5
VISIBLE_RATIO_THRESHOLD  = 0.80
LOS_TOLERANCE_M          = 0.1
MAX_SITES                = 30
DEFAULT_RADIUS_M         = 150.0

COLORS = [
    "#e63946", "#2196f3", "#ff9800", "#4caf50",
    "#9c27b0", "#00bcd4", "#f44336", "#8bc34a",
]


def _hex_to_rgba(hex_color: str, alpha: float = 0.45) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ─────────────────────────────────────────────────────────────────────────────
# 10次メッシュ エンコード/デコード (JIS X 0410, 15桁)
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


def decode_mesh10(code: str) -> tuple:
    """メッシュコード（15桁）→ (center_lat, center_lon, lat_sz, lon_sz)"""
    p  = int(code[0:2])
    q  = int(code[2:4])
    lat = p / 1.5
    lon = float(q + 100)
    lat_sz = 2.0 / 3.0
    lon_sz = 1.0
    lat_sz /= 8;  lon_sz /= 8
    lat += int(code[4]) * lat_sz;  lon += int(code[5]) * lon_sz
    lat_sz /= 10; lon_sz /= 10
    lat += int(code[6]) * lat_sz;  lon += int(code[7]) * lon_sz
    for i in range(7):
        lat_sz /= 2; lon_sz /= 2
        d = int(code[8 + i])
        if d in (3, 4): lat += lat_sz
        if d in (2, 4): lon += lon_sz
    return lat + lat_sz / 2, lon + lon_sz / 2, lat_sz, lon_sz


def reconstruct_from_meshes(bb_list: list, sid_to_meshes: dict) -> tuple:
    """
    メッシュコードリストから all_visible / all_candidates / all_circles を再構築する。
    候補メッシュ = 円形エリア内のメッシュ（1%以上のオーバーラップ）のうち
    アップロード済み可視メッシュに含まれないもの。
    """
    all_visible:    list = []
    all_circles:    list = []
    all_candidates: list = []
    lat_sz, lon_sz = mesh10_cell_size()
    mesh_area = lat_sz * lon_sz

    for bb in bb_list:
        sid    = str(bb["site_id"])
        lat    = float(bb["latitude"])
        lon    = float(bb["longitude"])
        radius = float(bb.get("radius", DEFAULT_RADIUS_M))
        lat_sc, lon_sc = local_scale(lat)

        circle = create_circle(lat, lon, radius)
        all_circles.append(circle)

        visible_codes = set(sid_to_meshes.get(sid, []))

        vis_rows = []
        for code in visible_codes:
            try:
                clat, clon, _, _ = decode_mesh10(code)
            except Exception:
                continue
            dx_m = (clon - lon) * lon_sc
            dy_m = (clat - lat) * lat_sc
            vis_rows.append({
                "billboard_id": sid,
                "mesh_code":    code,
                "center_lat":   round(clat, 8),
                "center_lon":   round(clon, 8),
                "distance_m":   round(math.sqrt(dx_m ** 2 + dy_m ** 2), 1),
                "area_ratio":   1.0,
            })
        all_visible.append(pd.DataFrame(vis_rows) if vis_rows else pd.DataFrame())

        cand_rows = []
        if not circle.is_empty:
            minlon_, minlat_, maxlon_, maxlat_ = circle.bounds
            _lats = np.arange(math.floor(minlat_ / lat_sz) * lat_sz, maxlat_ + lat_sz, lat_sz)
            _lons = np.arange(math.floor(minlon_ / lon_sz) * lon_sz, maxlon_ + lon_sz, lon_sz)
            for _la in _lats:
                for _lo in _lons:
                    _mbox = box(_lo, _la, _lo + lon_sz, _la + lat_sz)
                    if not circle.intersects(_mbox):
                        continue
                    inter = circle.intersection(_mbox)
                    if inter.is_empty or inter.area / mesh_area < 0.01:
                        continue
                    _code = encode_mesh10(_la + lat_sz / 2, _lo + lon_sz / 2)
                    if _code in visible_codes:
                        continue
                    clat_, clon_ = _la + lat_sz / 2, _lo + lon_sz / 2
                    dx_m = (clon_ - lon) * lon_sc
                    dy_m = (clat_ - lat) * lat_sc
                    cand_rows.append({
                        "billboard_id": sid,
                        "mesh_code":    _code,
                        "center_lat":   round(clat_, 8),
                        "center_lon":   round(clon_, 8),
                        "distance_m":   round(math.sqrt(dx_m ** 2 + dy_m ** 2), 1),
                        "area_ratio":   round(inter.area / mesh_area, 3),
                    })
        all_candidates.append(pd.DataFrame(cand_rows) if cand_rows else pd.DataFrame())

    return all_visible, all_circles, all_candidates


# ─────────────────────────────────────────────────────────────────────────────
# ジオメトリ補助
# ─────────────────────────────────────────────────────────────────────────────

def local_scale(lat: float):
    return 111320.0, 111320.0 * math.cos(math.radians(lat))


def create_circle(lat: float, lon: float, radius_m: float = DEFAULT_RADIUS_M) -> Polygon:
    """緯度・経度スケールを考慮した円形ポリゴンを生成する"""
    lat_sc, lon_sc = local_scale(lat)
    # 経度方向と緯度方向でスケールが異なるため楕円近似
    angles = np.linspace(0, 2 * math.pi, 72)
    coords = [
        (lon + radius_m * math.cos(a) / lon_sc,
         lat + radius_m * math.sin(a) / lat_sc)
        for a in angles
    ]
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
    m = re.search(r"EPSG/\d+/(\d+)", srs, re.IGNORECASE)
    if m:
        return f"EPSG:{m.group(1)}"
    m = re.search(r"crs:EPSG:[^:]*:(\d+)", srs, re.IGNORECASE)
    if m:
        return f"EPSG:{m.group(1)}"
    m = re.search(r"epsg\.xml#(\d+)", srs, re.IGNORECASE)
    if m:
        return f"EPSG:{m.group(1)}"
    m = re.search(r"EPSG[:/](\d{4,})", srs, re.IGNORECASE)
    if m:
        return f"EPSG:{m.group(1)}"
    return "EPSG:6668"


def _detect_swap_xy(src_crs: str) -> bool:
    try:
        crs_obj = CRS(src_crs)
        direction = crs_obj.axis_info[0].direction.lower()
        return direction in ("north", "south")
    except Exception:
        epsg_match = re.search(r"(\d{4,5})$", src_crs)
        if epsg_match:
            epsg = int(epsg_match.group(1))
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
        return Polygon([(p[1], p[0]) for p in pts])
    return Polygon([(p[0], p[1]) for p in pts])


def parse_citygml(gml_bytes: bytes) -> gpd.GeoDataFrame:
    try:
        root = etree.fromstring(gml_bytes)
    except Exception:
        return gpd.GeoDataFrame()

    src_crs  = _detect_crs(root)
    swap_xy  = _detect_swap_xy(src_crs)
    need_prj = (src_crs != "EPSG:4326")
    if need_prj:
        try:
            transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        except Exception:
            transformer = None
    else:
        transformer = None

    ns_b  = _BLDG_NS
    ns_b2 = _BLDG_NS2
    rows  = []

    for bldg in root.iter(f"{{{ns_b}}}Building", f"{{{ns_b2}}}Building"):
        h_el = (bldg.find(f"{{{ns_b}}}measuredHeight") or
                bldg.find(f"{{{ns_b2}}}measuredHeight"))
        try:
            height = float(h_el.text) if h_el is not None and h_el.text else 0.0
        except ValueError:
            height = 0.0

        polys = []
        for surf in bldg.iter(f"{{{_GML_NS}}}Polygon"):
            poly = _polygon_from_pos_list(surf, swap_xy=swap_xy)
            if poly and poly.is_valid and not poly.is_empty:
                polys.append(poly)

        if not polys:
            continue

        geom = polys[0] if len(polys) == 1 else polys[0].union(polys[1]) if len(polys) == 2 else polys[0]
        for p in polys[1:]:
            try:
                geom = geom.union(p)
            except Exception:
                pass

        if transformer:
            try:
                coords = [transformer.transform(x, y) for x, y in geom.exterior.coords]
                geom   = Polygon(coords)
            except Exception:
                continue

        if geom.is_valid and not geom.is_empty:
            rows.append({"geometry": geom, "height": height})

    if not rows:
        return gpd.GeoDataFrame()
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


# ─────────────────────────────────────────────────────────────────────────────
# Plateau 自動取得
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_plateau_catalog() -> dict:
    """geospatial.jp CKAN から Plateau データセット一覧を取得 → {muniCd: dataset_id}"""
    url  = "https://www.geospatial.jp/ckan/api/3/action/package_search?q=plateau&rows=1000"
    catalog: dict = {}
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = _json.loads(r.read())
        for item in data.get("result", {}).get("results", []):
            name = item.get("name", "")
            m = re.match(r"^plateau-(\d{5})-.*-(\d{4})$", name)
            if m:
                muni_cd = m.group(1)
                if muni_cd not in catalog:
                    catalog[muni_cd] = name
    except Exception:
        pass
    return catalog


def _gsi_reverse_geocode(lat: float, lon: float) -> Optional[str]:
    url = (f"https://mreversegeocoder.gsi.go.jp/reverse-geocoder/"
           f"LonLatToAddress?lat={lat}&lon={lon}")
    for _ in range(2):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = _json.loads(r.read())
            return data["results"]["muniCd"]
        except Exception:
            pass
    return None


def _get_plateau_zip_url(dataset_id: str) -> Optional[str]:
    url = f"https://www.geospatial.jp/ckan/api/3/action/package_show?id={dataset_id}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = _json.loads(r.read())
        resources = data["result"]["resources"]
        v4_url = v3_url = fallback_url = None
        for res in resources:
            name  = res.get("name", "")
            rurl  = res.get("url", "")
            name_l = name.lower()
            if "citygml" in name_l and rurl.lower().endswith(".zip"):
                if "v4" in name_l:   v4_url      = v4_url      or rurl
                elif "v3" in name_l: v3_url      = v3_url      or rurl
                else:                fallback_url = fallback_url or rurl
        return v4_url or v3_url or fallback_url
    except Exception:
        return None


def _read_zip_cd(zip_url: str) -> dict:
    """ZIP セントラルディレクトリのみ取得 → {filename: (local_offset, comp_size, method)}"""
    head_req = urllib.request.Request(zip_url, headers={"Range": "bytes=-65536"})
    with urllib.request.urlopen(head_req, timeout=60) as r:
        tail = r.read()
    eocd_off = tail.rfind(b"PK\x05\x06")
    if eocd_off < 0:
        raise ValueError("EOCD not found")
    eocd     = tail[eocd_off:]
    cd_size  = struct.unpack_from("<I", eocd, 12)[0]
    cd_off   = struct.unpack_from("<I", eocd, 16)[0]
    cd_req   = urllib.request.Request(zip_url, headers={"Range": f"bytes={cd_off}-{cd_off+cd_size-1}"})
    with urllib.request.urlopen(cd_req, timeout=60) as r:
        cd_data = r.read()
    entries = {}
    pos = 0
    while pos < len(cd_data) - 4:
        if cd_data[pos:pos+4] != b"PK\x01\x02":
            break
        method    = struct.unpack_from("<H", cd_data, pos + 10)[0]
        comp_size = struct.unpack_from("<I", cd_data, pos + 20)[0]
        fname_len = struct.unpack_from("<H", cd_data, pos + 28)[0]
        extra_len = struct.unpack_from("<H", cd_data, pos + 30)[0]
        comm_len  = struct.unpack_from("<H", cd_data, pos + 32)[0]
        local_off = struct.unpack_from("<I", cd_data, pos + 42)[0]
        fname     = cd_data[pos + 46: pos + 46 + fname_len].decode("utf-8", errors="replace")
        fname_short = fname.split("/")[-1].replace(".gml", "").replace(".GML", "")
        if fname_short:
            entries[fname_short] = (local_off, comp_size, method)
        pos += 46 + fname_len + extra_len + comm_len
    return entries


def _extract_gml_from_zip(zip_url: str, local_off: int, comp_size: int, method: int) -> bytes:
    lh_req = urllib.request.Request(zip_url, headers={"Range": f"bytes={local_off}-{local_off+29}"})
    with urllib.request.urlopen(lh_req, timeout=60) as r:
        lh = r.read()
    lh_fname_len = struct.unpack_from("<H", lh, 26)[0]
    lh_extra_len = struct.unpack_from("<H", lh, 28)[0]
    data_start   = local_off + 30 + lh_fname_len + lh_extra_len
    data_req = urllib.request.Request(
        zip_url, headers={"Range": f"bytes={data_start}-{data_start+comp_size-1}"}
    )
    with urllib.request.urlopen(data_req, timeout=120) as r:
        comp_data = r.read()
    return zlib.decompress(comp_data, -15) if method == 8 else comp_data


def get_needed_3rd_mesh_prefixes(billboards_df: pd.DataFrame) -> set:
    """円形エリアに必要な 3 次メッシュコード（8 桁）セットを計算"""
    lat_sz_3 = (2.0 / 3.0) / 8 / 10
    lon_sz_3 = 1.0 / 8 / 10
    prefixes = set()
    for _, bb in billboards_df.iterrows():
        circle = create_circle(bb.latitude, bb.longitude, float(bb.get("radius", DEFAULT_RADIUS_M)))
        minlon, minlat, maxlon, maxlat = circle.bounds
        la = math.floor(minlat / lat_sz_3) * lat_sz_3
        while la <= maxlat:
            lo = math.floor(minlon / lon_sz_3) * lon_sz_3
            while lo <= maxlon:
                if circle.intersects(box(lo, la, lo + lon_sz_3, la + lat_sz_3)):
                    code = encode_mesh10(la + lat_sz_3 / 2, lo + lon_sz_3 / 2)
                    prefixes.add(code[:8])
                lo += lon_sz_3
            la += lat_sz_3
    return prefixes


def auto_fetch_citygml(billboards_df: pd.DataFrame,
                       log_box) -> Optional[gpd.GeoDataFrame]:
    logs = []

    def log(msg: str):
        logs.append(msg)
        log_box.markdown("\n\n".join(logs))

    log("📋 Plateau カタログを取得中...")
    try:
        catalog = _fetch_plateau_catalog()
    except Exception as e:
        log(f"❌ カタログ取得エラー: {e}")
        return None
    log(f"✅ カタログ取得完了（{len(catalog)} 市区町村がPlateau対応）")

    log("📍 広告面板の市区町村を特定中...")
    muni_cds = set()
    _unique_coords = list({(round(row.latitude, 3), round(row.longitude, 3))
                           for _, row in billboards_df.iterrows()})
    log(f"   {len(billboards_df)}面板 → {len(_unique_coords)}ユニーク座標でジオコーディング")
    with ThreadPoolExecutor(max_workers=min(len(_unique_coords), 6)) as _ex:
        for muni_cd in _ex.map(lambda p: _gsi_reverse_geocode(*p), _unique_coords):
            if muni_cd:
                muni_cds.add(muni_cd)
    if not muni_cds:
        log("❌ 市区町村コードを取得できませんでした（GSI APIへの接続を確認してください）")
        return None
    log(f"✅ 市区町村コード: {', '.join(sorted(muni_cds))}")

    log("🗺️ 必要なメッシュタイルを計算中...")
    needed_prefixes = get_needed_3rd_mesh_prefixes(billboards_df)
    log(f"✅ 対象 3 次メッシュ: {', '.join(sorted(needed_prefixes))}（{len(needed_prefixes)} タイル）")

    all_gdfs = []
    for muni_cd in sorted(muni_cds):
        dataset_id = catalog.get(muni_cd) or catalog.get(muni_cd[:4] + "0")
        if not dataset_id:
            log(f"⚠️ 市区町村 {muni_cd} の Plateau データが見つかりません（対応エリア外の可能性）")
            continue

        log(f"🔍 `{dataset_id}` の ZIP URL を取得中...")
        zip_url = _get_plateau_zip_url(dataset_id)
        if not zip_url:
            log(f"⚠️ `{dataset_id}` の ZIP URL が取得できませんでした")
            continue

        log("📦 ZIP インデックスを解析中...")
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
            log("⚠️ 対象メッシュの GML が ZIP 内に見つかりませんでした")
            continue

        log(f"⬇️ {len(needed)} 個の GML ファイルをダウンロード中（並列）...")

        def _fetch_one_gml(item):
            fname, (local_off, comp_size, method) = item
            gml_bytes = _extract_gml_from_zip(zip_url, local_off, comp_size, method)
            return fname, comp_size, parse_citygml(gml_bytes)

        with ThreadPoolExecutor(max_workers=min(len(needed), 6)) as _gex:
            _gfuts = {_gex.submit(_fetch_one_gml, item): item[0]
                      for item in needed.items()}
            for _fut in as_completed(_gfuts):
                _fname = _gfuts[_fut]
                try:
                    _fn, _cs, gdf = _fut.result()
                    if not gdf.empty:
                        all_gdfs.append(gdf)
                        log(f"　　✅ `{_fn}`: 建物 {len(gdf):,} 棟")
                    else:
                        log(f"　　⚠️ `{_fn}`: 建物データが空でした")
                except Exception as e:
                    log(f"　　❌ `{_fname}` 取得失敗: {e}")

    if not all_gdfs:
        log("❌ 建物データを取得できませんでした")
        return None

    combined = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs="EPSG:4326")
    _area_union = None
    for _, _bb in billboards_df.iterrows():
        _c = create_circle(_bb.latitude, _bb.longitude, float(_bb.get("radius", DEFAULT_RADIUS_M)))
        _area_union = _c if _area_union is None else _area_union.union(_c)
    if _area_union is not None:
        before = len(combined)
        combined = combined[
            combined.geometry.intersects(_area_union.buffer(0.0001))
        ].reset_index(drop=True)
        log(f"✂️ 円形エリア外を除去: {before:,} → {len(combined):,} 棟")
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
    if D_m < LOS_TOLERANCE_M:
        return False
    if candidates.empty:
        return False
    for _, bldg in candidates.iterrows():
        bh = float(bldg.get("height", 0) or 0)
        if bh <= src_h:
            continue
        geom = bldg["geometry"]
        polys = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
        for poly in polys:
            if poly.geom_type != "Polygon":
                continue
            ring = list(poly.exterior.coords)
            for k in range(len(ring) - 1):
                wx0, wy0 = ring[k]
                wx1, wy1 = ring[k + 1]
                wall_line = LineString([(wx0, wy0), (wx1, wy1)])
                los_line  = LineString([(src_lon, src_lat), (tgt_lon, tgt_lat)])
                if not los_line.intersects(wall_line):
                    continue
                ip = los_line.intersection(wall_line)
                if ip.is_empty:
                    continue
                try:
                    ix, iy = ip.x, ip.y
                except AttributeError:
                    continue
                t = math.sqrt((ix - src_lon) ** 2 * lon_sc ** 2 +
                              (iy - src_lat) ** 2 * lat_sc ** 2) / D_m
                t = max(0.0, min(1.0, t))
                wall_h_at_t = src_h + t * (bh - src_h)
                los_h_at_t  = src_h + t * (tgt_h - src_h)
                if wall_h_at_t > los_h_at_t + LOS_TOLERANCE_M:
                    return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 視認計算
# ─────────────────────────────────────────────────────────────────────────────

def compute_visibility(bb: dict, buildings_gdf: Optional[gpd.GeoDataFrame]) -> tuple:
    """
    円形視認エリア計算。
    - 円形エリア: 指定半径の円
    - メッシュ判定: メッシュ面積の80%以上が円形エリア内 → 有効メッシュ
    - 建物LOS: SAMPLE_N 点全て遮蔽の場合のみ除外
    """
    lat    = bb["latitude"]
    lon    = bb["longitude"]
    radius = float(bb.get("radius", DEFAULT_RADIUS_M))
    sid    = bb.get("site_id", "B001")

    circle = create_circle(lat, lon, radius)
    lat_sz, lon_sz = mesh10_cell_size()
    lat_sc, lon_sc = local_scale(lat)

    if circle.is_empty:
        return pd.DataFrame(), pd.DataFrame(), circle, 0

    minlon, minlat, maxlon, maxlat = circle.bounds
    start_lat = math.floor(minlat / lat_sz) * lat_sz
    start_lon = math.floor(minlon / lon_sz) * lon_sz
    all_lats  = np.arange(start_lat, maxlat + lat_sz, lat_sz)
    all_lons  = np.arange(start_lon, maxlon + lon_sz, lon_sz)

    mesh_boxes = [
        {"lat": la + lat_sz/2, "lon": lo + lon_sz/2,
         "box": box(lo, la, lo+lon_sz, la+lat_sz)}
        for la in all_lats for lo in all_lons
        if circle.intersects(box(lo, la, lo+lon_sz, la+lat_sz))
    ]
    total = len(mesh_boxes)
    if total == 0:
        return pd.DataFrame(), pd.DataFrame(), circle, 0

    if buildings_gdf is not None and not buildings_gdf.empty:
        bldgs  = buildings_gdf[
            buildings_gdf.geometry.intersects(circle.buffer(0.00005))
        ].copy()
        sindex = bldgs.sindex if not bldgs.empty else None
    else:
        bldgs  = None
        sindex = None

    mesh_area     = lat_sz * lon_sz
    visible_rows  = []
    candidate_rows = []

    for m in mesh_boxes:
        mesh_box = m["box"]
        inter = circle.intersection(mesh_box)
        if inter.is_empty:
            continue
        area_ratio = inter.area / mesh_area
        if area_ratio < VISIBLE_RATIO_THRESHOLD:
            if area_ratio >= 0.01:
                _cand_blocked = False
                if bldgs is not None and sindex is not None:
                    _cpt = (inter.centroid.x, inter.centroid.y)
                    _cray = LineString([(lon, lat), _cpt])
                    _ccands = bldgs.iloc[list(sindex.intersection(_cray.bounds))]
                    _cand_blocked = _is_blocked(lon, lat, 0, _cpt[0], _cpt[1],
                                                EYE_HEIGHT_M, _ccands, lat_sc, lon_sc)
                if not _cand_blocked:
                    _cdx = (m["lon"] - lon) * lon_sc
                    _cdy = (m["lat"] - lat) * lat_sc
                    candidate_rows.append({
                        "billboard_id": sid,
                        "mesh_code":    encode_mesh10(m["lat"], m["lon"]),
                        "center_lat":   round(m["lat"], 8),
                        "center_lon":   round(m["lon"], 8),
                        "distance_m":   round(math.sqrt(_cdx**2 + _cdy**2), 1),
                        "area_ratio":   round(area_ratio, 3),
                    })
            continue

        if bldgs is not None and sindex is not None:
            sample_pts = [(inter.centroid.x, inter.centroid.y)]
            if SAMPLE_N > 1:
                minx, miny, maxx, maxy = inter.bounds
                nx = max(2, int(math.ceil(math.sqrt(SAMPLE_N * 2))))
                for _gi in range(nx):
                    for _gj in range(nx):
                        _px = minx + (maxx - minx) * (_gi + 0.5) / nx
                        _py = miny + (maxy - miny) * (_gj + 0.5) / nx
                        if inter.contains(Point(_px, _py)):
                            sample_pts.append((_px, _py))
                        if len(sample_pts) >= SAMPLE_N:
                            break
                    if len(sample_pts) >= SAMPLE_N:
                        break
            sample_pts = sample_pts[:SAMPLE_N]

            all_blocked = True
            for _sx, _sy in sample_pts:
                _ray   = LineString([(lon, lat), (_sx, _sy)])
                _cands = bldgs.iloc[list(sindex.intersection(_ray.bounds))]
                if not _is_blocked(lon, lat, 0, _sx, _sy, EYE_HEIGHT_M,
                                   _cands, lat_sc, lon_sc):
                    all_blocked = False
                    break
            if all_blocked:
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

    return pd.DataFrame(visible_rows), pd.DataFrame(candidate_rows), circle, total


# ─────────────────────────────────────────────────────────────────────────────
# 地図生成
# ─────────────────────────────────────────────────────────────────────────────

def build_map(billboards: list, circles: list, visible_dfs: list,
              buildings_gdf: Optional[gpd.GeoDataFrame],
              mesh_colors: Optional[dict] = None,
              focus_center: Optional[tuple] = None,
              focus_zoom: int = 16,
              candidates_dfs=None,
              activated_codes=None,
              deactivated_codes=None) -> go.Figure:
    fig = go.Figure()

    if buildings_gdf is not None and not buildings_gdf.empty:
        _map_area = None
        for _bb in billboards:
            _c = create_circle(_bb["latitude"], _bb["longitude"],
                               float(_bb.get("radius", DEFAULT_RADIUS_M)))
            _map_area = _c if _map_area is None else _map_area.union(_c)
        if _map_area is not None:
            buildings_gdf = buildings_gdf[
                buildings_gdf.geometry.intersects(_map_area.buffer(0.00005))
            ]

    if buildings_gdf is not None and not buildings_gdf.empty:
        max_h = buildings_gdf["height"].quantile(0.95) or 1
        bins  = [0, max_h*0.2, max_h*0.4, max_h*0.6, max_h*0.8, float("inf")]
        tier_fill = [
            "rgba(40,200,80,0.50)", "rgba(160,220,0,0.50)", "rgba(255,210,0,0.50)",
            "rgba(255,110,0,0.55)", "rgba(220,30,30,0.60)",
        ]
        tier_line = [
            "rgba(20,150,50,0.8)", "rgba(100,170,0,0.8)", "rgba(200,160,0,0.8)",
            "rgba(200,70,0,0.8)",  "rgba(170,0,0,0.8)",
        ]
        labels = [
            f"🟢 建物 〜{max_h*0.2:.0f}m（低）", f"🟡 建物 〜{max_h*0.4:.0f}m",
            f"🟡 建物 〜{max_h*0.6:.0f}m",       f"🟠 建物 〜{max_h*0.8:.0f}m",
            f"🔴 建物 {max_h*0.8:.0f}m〜（高）",
        ]
        for tier in range(5):
            lo_h, hi_h = bins[tier], bins[tier + 1]
            subset = buildings_gdf[
                (buildings_gdf["height"] > lo_h) & (buildings_gdf["height"] <= hi_h)
            ]
            if subset.empty:
                continue
            all_lons_b: list = []
            all_lats_b: list = []
            for geom in subset["geometry"]:
                polys = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
                for poly in polys:
                    if poly.geom_type != "Polygon":
                        continue
                    xs, ys = poly.exterior.xy
                    all_lons_b.extend(list(xs) + [None])
                    all_lats_b.extend(list(ys) + [None])
            if not all_lons_b:
                continue
            fig.add_trace(go.Scattermapbox(
                lat=all_lats_b, lon=all_lons_b,
                mode="lines", fill="toself",
                fillcolor=tier_fill[tier],
                line=dict(color=tier_line[tier], width=0.5),
                name=labels[tier], hoverinfo="skip", showlegend=True,
            ))

    lat_sz, lon_sz = mesh10_cell_size()

    for idx, (bb, circle, vdf) in enumerate(zip(billboards, circles, visible_dfs)):
        color = COLORS[idx % len(COLORS)]
        sid   = bb.get("site_id", f"B{idx+1}")

        # 円形エリア描画
        xs, ys = circle.exterior.xy
        fig.add_trace(go.Scattermapbox(
            lat=list(ys), lon=list(xs),
            mode="lines", fill="toself",
            fillcolor="rgba(255,220,0,0.08)",
            line=dict(color=color, width=1.5),
            name=f"{sid} 円形エリア",
            hoverinfo="skip",
        ))

        _edit_mode = candidates_dfs is not None
        _deact_set = set(deactivated_codes or {})
        if not vdf.empty:
            _mc = mesh_colors.get(idx) if mesh_colors else None
            _fc = _hex_to_rgba(_mc, 0.45) if _mc else "rgba(30,130,255,0.45)"
            _lc = _hex_to_rgba(_mc, 0.85) if _mc else "rgba(0,70,210,0.85)"

            _vdf_active = vdf[~vdf["mesh_code"].isin(_deact_set)]
            if not _vdf_active.empty:
                box_lats, box_lons, box_texts = [], [], []
                for _, row in _vdf_active.iterrows():
                    la0 = row["center_lat"] - lat_sz / 2
                    lo0 = row["center_lon"] - lon_sz / 2
                    txt = (f"{row['mesh_code']}<br>"
                           f"距離: {row['distance_m']}m<br>"
                           f"円形内面積比: {row['area_ratio']*100:.1f}%")
                    box_lats.extend([la0, la0, la0+lat_sz, la0+lat_sz, la0, None])
                    box_lons.extend([lo0, lo0+lon_sz, lo0+lon_sz, lo0, lo0, None])
                    box_texts.extend([txt, txt, txt, txt, txt, ""])
                fig.add_trace(go.Scattermapbox(
                    lat=box_lats, lon=box_lons,
                    mode="lines", fill="toself",
                    fillcolor=_fc, line=dict(color=_lc, width=1),
                    name=f"● {sid} 有効メッシュ ({len(_vdf_active):,}件)",
                    text=box_texts, hovertemplate="%{text}<extra></extra>",
                ))
                if _edit_mode:
                    fig.add_trace(go.Scattermapbox(
                        lat=_vdf_active["center_lat"].tolist(),
                        lon=_vdf_active["center_lon"].tolist(),
                        mode="markers",
                        marker=dict(size=12, color=_lc, symbol="circle", opacity=0.7),
                        customdata=[[idx, row["mesh_code"], "v"] for _, row in _vdf_active.iterrows()],
                        name=f"{sid} 有効メッシュクリック",
                        hovertemplate="<b>自動メッシュ（クリックで取消）</b><br>%{customdata[1]}<extra></extra>",
                        showlegend=False,
                    ))

            _vdf_deact = vdf[vdf["mesh_code"].isin(_deact_set)]
            if not _vdf_deact.empty:
                d_lats, d_lons = [], []
                for _, row in _vdf_deact.iterrows():
                    la0 = row["center_lat"] - lat_sz / 2
                    lo0 = row["center_lon"] - lon_sz / 2
                    d_lats.extend([la0, la0, la0+lat_sz, la0+lat_sz, la0, None])
                    d_lons.extend([lo0, lo0+lon_sz, lo0+lon_sz, lo0, lo0, None])
                fig.add_trace(go.Scattermapbox(
                    lat=d_lats, lon=d_lons, mode="lines", fill="toself",
                    fillcolor="rgba(150,150,150,0.2)",
                    line=dict(color="rgba(120,120,120,0.6)", width=1),
                    name=f"{sid} 取り消し済み ({len(_vdf_deact)}件)",
                    hoverinfo="skip", showlegend=True,
                ))
                fig.add_trace(go.Scattermapbox(
                    lat=_vdf_deact["center_lat"].tolist(),
                    lon=_vdf_deact["center_lon"].tolist(),
                    mode="markers",
                    marker=dict(size=12, color="gray", symbol="circle", opacity=0.6),
                    customdata=[[idx, row["mesh_code"], "v"] for _, row in _vdf_deact.iterrows()],
                    name=f"{sid} 取り消し済みクリック",
                    hovertemplate="<b>取り消し済み（再クリックで復元）</b><br>%{customdata[1]}<extra></extra>",
                    showlegend=False,
                ))

        if candidates_dfs and idx < len(candidates_dfs):
            cdf = candidates_dfs[idx]
            if cdf is not None and not cdf.empty:
                _act_set = set(activated_codes or {})
                _pending = cdf[~cdf["mesh_code"].isin(_act_set)]
                if not _pending.empty:
                    _cbox_lats, _cbox_lons = [], []
                    for _, _cm in _pending.iterrows():
                        _cla, _clo = _cm["center_lat"], _cm["center_lon"]
                        _cbox_lats += [_cla-lat_sz/2, _cla+lat_sz/2, _cla+lat_sz/2, _cla-lat_sz/2, _cla-lat_sz/2, None]
                        _cbox_lons += [_clo-lon_sz/2, _clo-lon_sz/2, _clo+lon_sz/2, _clo+lon_sz/2, _clo-lon_sz/2, None]
                    fig.add_trace(go.Scattermapbox(
                        lat=_cbox_lats, lon=_cbox_lons,
                        mode="lines", fill="toself",
                        fillcolor="rgba(0,200,0,0.25)",
                        line=dict(color="rgba(0,180,0,0.8)", width=1.5),
                        name=f"{sid} 候補メッシュ (クリックで有効化)",
                        hoverinfo="skip", showlegend=True,
                    ))
                    fig.add_trace(go.Scattermapbox(
                        lat=_pending["center_lat"].tolist(),
                        lon=_pending["center_lon"].tolist(),
                        mode="markers",
                        marker=dict(size=14, color="rgba(0,200,0,0.7)", symbol="circle"),
                        customdata=[[idx, row["mesh_code"], "c"] for _, row in _pending.iterrows()],
                        name=f"{sid} 候補クリック",
                        hovertemplate="<b>候補メッシュ（クリックで有効化）</b><br>%{customdata[1]}<extra></extra>",
                        showlegend=False,
                    ))

        if activated_codes and candidates_dfs and idx < len(candidates_dfs):
            cdf = candidates_dfs[idx]
            if cdf is not None and not cdf.empty:
                _act_set = set(activated_codes or {})
                _actdf = cdf[cdf["mesh_code"].isin(_act_set)]
                if not _actdf.empty:
                    _abox_lats, _abox_lons = [], []
                    _act_col = mesh_colors.get(idx, color) if mesh_colors else color
                    for _, _am in _actdf.iterrows():
                        _ala, _alo = _am["center_lat"], _am["center_lon"]
                        _abox_lats += [_ala-lat_sz/2, _ala+lat_sz/2, _ala+lat_sz/2, _ala-lat_sz/2, _ala-lat_sz/2, None]
                        _abox_lons += [_alo-lon_sz/2, _alo-lon_sz/2, _alo+lon_sz/2, _alo+lon_sz/2, _alo-lon_sz/2, None]
                    fig.add_trace(go.Scattermapbox(
                        lat=_abox_lats, lon=_abox_lons,
                        mode="lines", fill="toself",
                        fillcolor=_hex_to_rgba(_act_col if _act_col.startswith("#") else color, 0.55),
                        line=dict(color=_act_col if _act_col.startswith("#") else color, width=1.5),
                        name=f"{sid} 手動追加メッシュ ({len(_actdf)}件)",
                        hoverinfo="skip", showlegend=True,
                    ))
                    fig.add_trace(go.Scattermapbox(
                        lat=_actdf["center_lat"].tolist(),
                        lon=_actdf["center_lon"].tolist(),
                        mode="markers",
                        marker=dict(size=14, color=_act_col if _act_col.startswith("#") else color,
                                    symbol="circle", opacity=0.8),
                        customdata=[[idx, row["mesh_code"], "c"] for _, row in _actdf.iterrows()],
                        name=f"{sid} 有効化済みクリック",
                        hovertemplate="<b>手動追加済み（再クリックで取消）</b><br>%{customdata[1]}<extra></extra>",
                        showlegend=False,
                    ))

        # サイトマーカー
        fig.add_trace(go.Scattermapbox(
            lat=[bb["latitude"]], lon=[bb["longitude"]],
            mode="markers",
            marker=dict(size=14, color=color, symbol="circle"),
            name=f"{sid} (r={bb.get('radius', DEFAULT_RADIUS_M):.0f}m)",
            hovertemplate=(f"<b>{sid}</b><br>半径: {bb.get('radius', DEFAULT_RADIUS_M):.0f}m<extra></extra>"),
        ))

    if focus_center:
        center_lat, center_lon = focus_center
    else:
        _lats = [bb["latitude"]  for bb in billboards]
        _lons = [bb["longitude"] for bb in billboards]
        center_lat = float(np.mean(_lats))
        center_lon = float(np.mean(_lons))
        if len(_lats) > 1:
            _span = max(max(_lats) - min(_lats), max(_lons) - min(_lons), 1e-6)
            focus_zoom = int(np.clip(np.log2(180 / _span), 4, 15))
    fig.update_layout(
        mapbox=dict(style="open-street-map",
                    center=dict(lat=center_lat, lon=center_lon), zoom=focus_zoom),
        height=680,
        margin=dict(r=0, t=0, l=0, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.88)"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="円形視認エリア解析システム", page_icon="⭕", layout="wide")


# ── パスワード認証 ──────────────────────────────────────────────────────────
def _check_password():
    if st.session_state.get("authenticated"):
        return
    st.title("🔒 ログインが必要です")
    st.caption("このアプリを利用するにはパスワードが必要です。")
    pwd = st.text_input("パスワード", type="password", key="pwd_input")
    if st.button("ログイン", type="primary"):
        _app_pwd = st.secrets.get("APP_PASSWORD", "")
        if not _app_pwd:
            st.error("⚠️ APP_PASSWORD が Secrets に設定されていません。管理者に設定してください。")
        elif pwd == _app_pwd:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("パスワードが違います。")
    st.stop()

_check_password()

st.title("⭕ 円形視認エリア解析システム")
st.caption("Plateau CityGML × 10次メッシュ LOS 解析（円形エリア版）")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ データ入力")
    st.divider()

    # ① サイトデータ
    st.subheader("① サイトデータ")
    bb_input_mode = st.radio(
        "入力方法",
        ["📂 CSVアップロード", "✏️ 手入力（1件のみ）"],
        key="bb_input_mode",
        horizontal=True,
    )

    bb_file   = None
    manual_bb = None

    if bb_input_mode == "📂 CSVアップロード":
        st.markdown("""
**必須列**: `site_id`, `latitude`, `longitude`, `radius`

- `radius`: 円形視認エリアの半径（メートル）
""")
        bb_file = st.file_uploader("CSVをアップロード", type=["csv"], key="bb_csv")
    else:
        st.caption("1件の情報を入力してください。")
        with st.form("manual_bb_form"):
            sid_raw = st.text_input("Site ID", placeholder="例: 000001")
            m_lat   = st.number_input("緯度 latitude",  value=35.4657, format="%.6f")
            m_lon   = st.number_input("経度 longitude", value=139.6223, format="%.6f")
            m_r     = st.number_input("半径 radius (m)", value=DEFAULT_RADIUS_M,
                                      min_value=10.0, max_value=2000.0, step=10.0)
            submitted = st.form_submit_button("✅ 設定を反映", use_container_width=True)

        if submitted:
            sid_clean = str(sid_raw).strip()
            if len(sid_clean) == 0:
                st.warning("⚠️ Site IDを入力してください。")
            else:
                st.session_state["manual_bb"] = {
                    "site_id":   sid_clean,
                    "latitude":  m_lat,
                    "longitude": m_lon,
                    "radius":    m_r,
                }

        if st.session_state.get("manual_bb"):
            manual_bb = st.session_state["manual_bb"]
            d = manual_bb
            st.success(
                f"✅ **{d['site_id']}** 設定済み  \n"
                f"緯度 {d['latitude']:.5f} / 経度 {d['longitude']:.5f}  \n"
                f"半径 {d['radius']:.0f}m"
            )

    st.divider()

    # 🔄 前回メッシュを再利用
    st.subheader("🔄 前回メッシュを再利用（任意）")
    st.caption(
        "前回ダウンロードしたメッシュZIPをアップロードすると、"
        "建物データ取得・計算をスキップして直接結果を表示します。"
    )
    reuse_zip = st.file_uploader(
        "メッシュコードZIP", type=["zip"], key="reuse_zip",
        help="「全 Site ID を一括ダウンロード」で取得した ZIP ファイルをアップロードしてください。",
    )

    gml_file  = None
    fetch_btn = False
    bldg_mode = "⛔ 使用しない"

    if reuse_zip is None:
        st.divider()
        st.subheader("② 建物データ（CityGML）")
        bldg_mode = st.radio(
            "取得方法",
            ["🚀 Plateau から自動取得", "📂 手動アップロード", "⛔ 使用しない"],
            help=(
                "自動取得: サイト位置から必要な建物データをネット経由で自動ダウンロード\n"
                "手動: .gml ファイルをアップロード\n"
                "使用しない: 建物遮蔽なし（円形全体を視認エリアとして計算）"
            ),
        )
        if bldg_mode == "📂 手動アップロード":
            gml_file = st.file_uploader(
                "CityGML (.gml) をアップロード", type=["gml", "xml"], key="gml"
            )
        elif bldg_mode == "🚀 Plateau から自動取得":
            st.caption(
                "CSV をアップロード後、ボタンで Plateau の建物データを自動ダウンロードします。"
                "インターネット接続が必要です。"
            )
            fetch_btn = st.button(
                "🏢 建物データを自動取得",
                disabled=(bb_file is None and not manual_bb),
                use_container_width=True,
                type="secondary",
            )
        else:
            st.caption("建物遮蔽なしで計算します。円形内のメッシュすべてが有効になります。")
    else:
        st.success("🔄 再利用モード有効: 建物データ取得・計算をスキップします。")

    st.divider()
    _has_input = (bb_file is not None) or bool(manual_bb)
    if reuse_zip is None:
        run_btn = st.button(
            "▶ 計算実行", type="primary", use_container_width=True,
            disabled=not _has_input,
        )
    else:
        run_btn = False

# ── Main ─────────────────────────────────────────────────────────────────────

_csv_mode = (bb_input_mode == "📂 CSVアップロード")

if (_csv_mode and bb_file is None) or (not _csv_mode and not manual_bb):
    st.info("👈 左のサイドバーからサイトデータを入力してください。")
    st.stop()

# bb_df の構築
if _csv_mode:
    try:
        bb_df = pd.read_csv(bb_file, dtype={"site_id": str})
        required = {"site_id", "latitude", "longitude", "radius"}
        missing  = required - set(bb_df.columns)
        if missing:
            st.error(f"CSV に必要な列がありません: {missing}")
            st.stop()
        if len(bb_df) > MAX_SITES:
            st.error(f"一度に処理できるサイトは最大 {MAX_SITES} 件です（現在 {len(bb_df)} 行）。CSVを分割してください。")
            st.stop()
        bb_df["radius"] = pd.to_numeric(bb_df["radius"], errors="coerce").fillna(DEFAULT_RADIUS_M)
    except Exception as e:
        st.error(f"CSV 読み込みエラー: {e}")
        st.stop()
else:
    bb_df = pd.DataFrame([manual_bb])

st.success(f"サイト {len(bb_df)} 件を読み込みました")

# ── 半径編集テーブル ──────────────────────────────────────────────────────────
st.subheader("📋 サイト一覧（半径を編集できます）")
st.caption("radius 列をクリックして半径（メートル）を変更できます。変更後に計算実行してください。")

_edit_df = bb_df[["site_id", "latitude", "longitude", "radius"]].copy()
_edited = st.data_editor(
    _edit_df,
    column_config={
        "site_id":   st.column_config.TextColumn("Site ID", disabled=True),
        "latitude":  st.column_config.NumberColumn("緯度", format="%.6f", disabled=True),
        "longitude": st.column_config.NumberColumn("経度", format="%.6f", disabled=True),
        "radius":    st.column_config.NumberColumn("半径 (m)", min_value=10, max_value=2000, step=10),
    },
    use_container_width=True,
    hide_index=True,
    key="radius_editor",
)
# 編集後の値を bb_df に反映
bb_df = bb_df.copy()
bb_df["radius"] = _edited["radius"].values

# ── 位置補正用: corrected_coords 初期化 ──────────────────────────────────────
_src_sig = bb_df[["site_id", "latitude", "longitude"]].to_csv(index=False)
if st.session_state.get("_corr_src") != _src_sig:
    st.session_state["_corr_src"] = _src_sig
    st.session_state["corrected_coords"] = {
        str(i): {"latitude": float(r["latitude"]), "longitude": float(r["longitude"])}
        for i, r in bb_df.iterrows()
    }
    if "finalized_master" in st.session_state:
        del st.session_state["finalized_master"]

_corr = st.session_state["corrected_coords"]
bb_df_w = bb_df.copy()
for _ci, _cr in bb_df_w.iterrows():
    _ckey = str(_ci)
    if _ckey in _corr:
        bb_df_w.at[_ci, "latitude"]  = _corr[_ckey]["latitude"]
        bb_df_w.at[_ci, "longitude"] = _corr[_ckey]["longitude"]

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

buildings_gdf = st.session_state.get("buildings_gdf") if bldg_mode != "⛔ 使用しない" else None
if bldg_mode == "🚀 Plateau から自動取得" and buildings_gdf is not None:
    st.info(f"🏢 取得済み建物データ: {len(buildings_gdf):,} 棟（高さ平均 {buildings_gdf['height'].mean():.1f}m）")

st.divider()

# ── プレビューマップ ──────────────────────────────────────────────────────────
st.subheader("📍 設定確認マップ")
prev_fig = go.Figure()
for idx, row in bb_df_w.iterrows():
    color  = COLORS[idx % len(COLORS)]
    circle = create_circle(row.latitude, row.longitude, float(row["radius"]))
    xs, ys = circle.exterior.xy
    prev_fig.add_trace(go.Scattermapbox(
        lat=list(ys), lon=list(xs), mode="lines", fill="toself",
        fillcolor="rgba(255,200,0,0.15)", line=dict(color=color, width=1.5),
        name=f"{row.site_id} 円形エリア",
        hoverinfo="skip",
    ))
    prev_fig.add_trace(go.Scattermapbox(
        lat=[row.latitude], lon=[row.longitude], mode="markers",
        marker=dict(size=13, color=color),
        name=str(row.site_id),
        hovertemplate=f"<b>{row.site_id}</b><br>半径: {row['radius']:.0f}m<extra></extra>",
    ))

_prev_focus_opts = ["全表示"] + [str(row.site_id) for _, row in bb_df_w.iterrows()]
_prev_focus_sel = st.selectbox(
    "🎯 フォーカス（選択するとマップがそのサイトへ移動）",
    _prev_focus_opts,
    key="prev_map_focus",
)
if _prev_focus_sel != "全表示":
    _prev_idx  = _prev_focus_opts.index(_prev_focus_sel) - 1
    _prev_row  = bb_df_w.iloc[_prev_idx]
    center_lat = float(_prev_row["latitude"])
    center_lon = float(_prev_row["longitude"])
    _prev_zoom = 17
else:
    center_lat = bb_df_w["latitude"].mean()
    center_lon = bb_df_w["longitude"].mean()
    if len(bb_df_w) > 1:
        _span = max(bb_df_w["latitude"].max()  - bb_df_w["latitude"].min(),
                    bb_df_w["longitude"].max() - bb_df_w["longitude"].min(), 1e-6)
        _prev_zoom = int(np.clip(np.log2(180 / _span), 4, 15))
    else:
        _prev_zoom = 16

prev_fig.update_layout(
    mapbox=dict(style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon), zoom=_prev_zoom),
    height=420, margin=dict(r=0, t=0, l=0, b=0),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                bgcolor="rgba(255,255,255,0.88)"),
)
st.plotly_chart(prev_fig, use_container_width=True)
st.caption("▲ 円形が視認エリアの範囲です。下の補正マップで位置を調整後に計算してください。")

# ── 位置補正マップ ────────────────────────────────────────────────────────────
st.divider()
st.subheader("✏️ 位置補正マップ")

if not _FOLIUM_OK:
    st.warning("folium / streamlit-folium が未インストールです。")
else:
    _sel_opts = {
        str(i): str(r["site_id"])
        for i, r in bb_df_w.iterrows()
    }

    _cc_map, _cc_ctrl = st.columns([3, 2])

    with _cc_ctrl:
        _sel = st.selectbox(
            "補正するサイトを選択",
            options=list(_sel_opts.keys()),
            format_func=lambda k: _sel_opts[k],
            key="corr_select",
            help="選択したサイトのマーカーが赤くなります。",
        )

    _sel_center = _corr.get(_sel, {})
    _fm_lat = _sel_center.get("latitude", bb_df_w["latitude"].mean())
    _fm_lon = _sel_center.get("longitude", bb_df_w["longitude"].mean())
    _fm = folium.Map(location=[_fm_lat, _fm_lon], zoom_start=17, tiles="OpenStreetMap")

    for _fi, _fr in bb_df_w.iterrows():
        _fi_key = str(_fi)
        _fsid   = str(_fr["site_id"])
        _flat   = _corr[_fi_key]["latitude"]
        _flon   = _corr[_fi_key]["longitude"]
        _fcolor = "red" if _fi_key == _sel else "blue"
        _fcircle = create_circle(_flat, _flon, float(_fr["radius"]))
        folium.Polygon(
            locations=[[p[1], p[0]] for p in _fcircle.exterior.coords],
            color=COLORS[_fi % len(COLORS)],
            fill=True, fill_opacity=0.15, weight=2,
            tooltip=f"{_fsid} 円形エリア (r={_fr['radius']:.0f}m)",
        ).add_to(_fm)
        folium.Marker(
            location=[_flat, _flon],
            popup=_fsid,
            tooltip=f"{_fsid}（{_flat:.5f}, {_flon:.5f}）",
            icon=folium.Icon(color=_fcolor, icon="flag"),
        ).add_to(_fm)

    with _cc_map:
        _map_res = st_folium(
            _fm, key="corr_folium", height=430,
            use_container_width=True,
            returned_objects=["last_clicked"],
        )

    with _cc_ctrl:
        _cur      = _corr[_sel]
        _orig_row = bb_df.loc[int(_sel)]
        _sel_sid  = str(_orig_row["site_id"])
        _is_moved = (
            abs(_cur["latitude"]  - float(_orig_row["latitude"]))  > 1e-7 or
            abs(_cur["longitude"] - float(_orig_row["longitude"])) > 1e-7
        )
        _status_icon = "✏️" if _is_moved else "📍"
        st.markdown(
            f"**{_status_icon} {_sel_sid} 現在値**  \n"
            f"緯度: `{_cur['latitude']:.6f}`  \n"
            f"経度: `{_cur['longitude']:.6f}`"
        )

        _clk = (_map_res or {}).get("last_clicked")
        if _clk:
            _clk_lat = round(_clk["lat"], 6)
            _clk_lon = round(_clk["lng"], 6)
            st.info(
                f"📍 クリック位置  \n"
                f"緯度: `{_clk_lat}`  \n"
                f"経度: `{_clk_lon}`"
            )
            if st.button(
                f"▶ {_sel_sid} をこの位置に移動",
                key="apply_corr", type="secondary", use_container_width=True,
            ):
                st.session_state["corrected_coords"][_sel]["latitude"]  = _clk_lat
                st.session_state["corrected_coords"][_sel]["longitude"] = _clk_lon
                if "finalized_master" in st.session_state:
                    del st.session_state["finalized_master"]
                st.rerun()
        else:
            st.caption("地図上をクリックすると新しい位置を指定できます")

        if _is_moved:
            if st.button(f"↩ {_sel_sid} の補正をリセット", key="reset_corr", use_container_width=True):
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
            _or = bb_df.loc[int(_ss)]
            _mv = (abs(_sp["latitude"]  - float(_or["latitude"]))  > 1e-7 or
                   abs(_sp["longitude"] - float(_or["longitude"])) > 1e-7)
            st.caption(
                f"{'✏️' if _mv else '📍'} **{str(_or['site_id'])}**: "
                f"{_sp['latitude']:.5f}, {_sp['longitude']:.5f}"
            )

# ── 最終確定 ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader("✅ 最終確定")
_fin_c1, _fin_c2 = st.columns([1, 1])

with _fin_c1:
    if st.button("✅ 位置を最終確定する", type="primary", key="finalize_btn", use_container_width=True):
        _fdf = bb_df.copy()
        for _fi2, _fr2 in _fdf.iterrows():
            _fc = st.session_state.get("corrected_coords", {}).get(str(_fi2))
            if _fc:
                _fdf.at[_fi2, "latitude"]  = _fc["latitude"]
                _fdf.at[_fi2, "longitude"] = _fc["longitude"]
        st.session_state["finalized_master"] = _fdf

if "finalized_master" in st.session_state:
    _fmdf    = st.session_state["finalized_master"]
    _out_cols = [c for c in ["site_id", "latitude", "longitude", "radius"] if c in _fmdf.columns]
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

# ── メッシュ再利用モード ──────────────────────────────────────────────────────
_reuse_mode = reuse_zip is not None

if _reuse_mode:
    _zip_bytes  = reuse_zip.getvalue()
    _reuse_hash = hashlib.md5(
        _zip_bytes + bb_df_w.to_csv(index=False).encode()
    ).hexdigest()

    if st.session_state.get("reuse_zip_hash") != _reuse_hash:
        _sid_meshes: dict = {}
        try:
            with zipfile.ZipFile(io.BytesIO(_zip_bytes)) as _rzf:
                for _rfname in _rzf.namelist():
                    _rm = re.match(r"No\.(.+)\.csv$", _rfname)
                    if _rm:
                        _rsid = _rm.group(1)
                        _rcontent = _rzf.read(_rfname).decode("utf-8")
                        _sid_meshes[_rsid] = [
                            ln.strip() for ln in _rcontent.splitlines() if ln.strip()
                        ]
        except Exception as _re:
            st.error(f"ZIPの解析に失敗しました: {_re}")
            st.stop()

        _bb_recs = bb_df_w.to_dict("records")
        _av, _as, _ac = reconstruct_from_meshes(_bb_recs, _sid_meshes)
        _rdf = (
            pd.concat(_av, ignore_index=True)
            if any(not v.empty for v in _av) else pd.DataFrame()
        )
        _total_meshes = sum(len(v) for v in _av if not v.empty)
        st.session_state["result_df"]       = _rdf
        st.session_state["all_visible"]     = _av
        st.session_state["all_sectors"]     = _as
        st.session_state["all_candidates"]  = _ac
        st.session_state["bb_list"]         = _bb_recs
        st.session_state["buildings_calc"]  = None
        st.session_state["buildings_orig"]  = None
        st.session_state["reuse_zip_hash"]  = _reuse_hash
        st.session_state.pop("excl_applied",       None)
        st.session_state.pop("manual_activated",   None)
        st.session_state.pop("manual_deactivated", None)
        st.success(f"🔄 メッシュを再構築しました（合計 {_total_meshes:,} メッシュ）")
        st.rerun()
    else:
        _total_meshes = sum(
            len(v) for v in st.session_state.get("all_visible", [])
            if v is not None and not v.empty
        )
        st.info(f"🔄 再利用メッシュ読み込み済み（合計 {_total_meshes:,} メッシュ）")

# ── 計算実行 ─────────────────────────────────────────────────────────────────
if not _reuse_mode and run_btn:
    _n_bb    = len(bb_df_w)
    prog_bar = st.progress(0, text="計算を開始しています...")
    all_visible    = [None] * _n_bb
    all_sectors    = [None] * _n_bb
    all_candidates = [None] * _n_bb
    _done = [0]

    def _calc_one(args):
        idx, bb = args
        return idx, compute_visibility(bb, buildings_gdf)

    with ThreadPoolExecutor(max_workers=min(_n_bb, 6)) as _ex:
        _futs = {_ex.submit(_calc_one, (idx, row.to_dict())): idx
                 for idx, (_, row) in enumerate(bb_df_w.iterrows())}
        for _fut in as_completed(_futs):
            idx, (vdf, cdf, sector, _) = _fut.result()
            all_visible[idx]    = vdf
            all_candidates[idx] = cdf
            all_sectors[idx]    = sector
            _done[0] += 1
            prog_bar.progress(_done[0] / _n_bb, text=f"{_done[0]}/{_n_bb} 件完了")

    prog_bar.progress(1.0, text="完了！")

    result_df = (
        pd.concat(all_visible, ignore_index=True)
        if any(not v.empty for v in all_visible)
        else pd.DataFrame()
    )
    st.session_state["result_df"]       = result_df
    st.session_state["all_visible"]     = all_visible
    st.session_state["all_sectors"]     = all_sectors
    st.session_state["all_candidates"]  = all_candidates
    st.session_state["bb_list"]         = bb_df_w.to_dict("records")
    st.session_state["buildings_calc"]  = buildings_gdf
    st.session_state["buildings_orig"]  = buildings_gdf
    st.session_state.pop("excl_applied",       None)
    st.session_state.pop("manual_activated",   None)
    st.session_state.pop("manual_deactivated", None)

# ── 結果表示 ─────────────────────────────────────────────────────────────────
if "result_df" in st.session_state:
    result_df      = st.session_state["result_df"]
    all_visible    = st.session_state["all_visible"]
    all_sectors    = st.session_state["all_sectors"]
    bb_list        = st.session_state["bb_list"]
    buildings_calc = st.session_state["buildings_calc"]
    _excl_applied  = st.session_state.get("excl_applied", frozenset())

    st.divider()
    st.subheader("📊 計算結果")

    cols = st.columns(min(len(bb_list), 4))
    for _ci, _bb in enumerate(bb_list):
        _vdf = all_visible[_ci] if all_visible[_ci] is not None else pd.DataFrame()
        with cols[_ci % min(len(bb_list), 4)]:
            st.metric(
                label=str(_bb["site_id"]),
                value=f"{len(_vdf):,} メッシュ",
                delta=f"半径 {_bb.get('radius', DEFAULT_RADIUS_M):.0f}m",
            )

    # 地図表示設定
    with st.expander("🎛️ 地図表示設定", expanded=True):
        _focus_opts = ["全表示"] + [str(_bb["site_id"]) for _bb in bb_list]
        _focus_sel = st.selectbox(
            "🎯 フォーカス（選択すると地図がそのサイトへ移動）",
            _focus_opts,
            key="map_focus",
        )

        _n_bb  = len(bb_list)
        _fcols = st.columns(min(_n_bb, 4))
        _show  = {}
        _mcols = {}
        for _i, _bb in enumerate(bb_list):
            _s = str(_bb["site_id"])
            with _fcols[_i % min(_n_bb, 4)]:
                _show[_i]  = st.checkbox(f"表示: {_s}", value=True, key=f"show_{_i}")
                _mcols[_i] = st.color_picker(
                    f"メッシュ色: {_s}", value=COLORS[_i % len(COLORS)], key=f"meshcol_{_i}",
                )

    _focus_center = None
    _focus_zoom   = 16
    if _focus_sel != "全表示":
        _focus_idx    = _focus_opts.index(_focus_sel) - 1
        _focus_bb     = bb_list[_focus_idx]
        _focus_center = (_focus_bb["latitude"], _focus_bb["longitude"])
        _focus_zoom   = 17

    _fbb   = [bb  for i, bb  in enumerate(bb_list)                          if _show.get(i, True)]
    _fvis  = [vdf for i, (bb, vdf) in enumerate(zip(bb_list, all_visible))  if _show.get(i, True)]
    _fsec  = [sec for i, (bb, sec) in enumerate(zip(bb_list, all_sectors))  if _show.get(i, True)]
    _fmcols = {new_i: _mcols[old_i]
               for new_i, old_i in enumerate(i for i, bb in enumerate(bb_list) if _show.get(i, True))}

    _bldgs_for_map = (
        buildings_calc[~buildings_calc.index.isin(_excl_applied)].copy()
        if buildings_calc is not None and _excl_applied
        else buildings_calc
    )
    with st.spinner("地図を生成中..."):
        fig = build_map(_fbb, _fsec, _fvis, _bldgs_for_map,
                        mesh_colors=_fmcols,
                        focus_center=_focus_center,
                        focus_zoom=_focus_zoom)
    st.plotly_chart(fig, use_container_width=True)

    # ── 手動メッシュ補正 ────────────────────────────────────────────────────────
    all_candidates = st.session_state.get("all_candidates")
    _manual_activated   = set(st.session_state.get("manual_activated",   set()))
    _manual_deactivated = set(st.session_state.get("manual_deactivated", set()))
    _fcat = None

    if all_candidates:
        _fcat = [all_candidates[old_i] if old_i < len(all_candidates) else None
                 for old_i in (i for i, _ in enumerate(bb_list) if _show.get(i, True))]
        _has_cands = any(c is not None and not c.empty for c in _fcat)
    else:
        _has_cands = False

    _has_visible = any(v is not None and not v.empty for v in all_visible)

    if _has_cands or _manual_activated or _manual_deactivated or _has_visible:
        st.divider()
        st.subheader("✏️ 手動メッシュ補正")
        if _has_cands:
            st.caption("自動メッシュをクリックで取り消し（グレー）→再クリックで復元。緑の候補メッシュをクリックで追加。FIX で確定。")
        else:
            st.caption("自動メッシュをクリックで取り消し（グレー）→再クリックで復元。FIX で確定。")

        with st.spinner("マップ生成中..."):
            _mfig = build_map(
                _fbb, _fsec, _fvis, _bldgs_for_map,
                mesh_colors=_fmcols,
                focus_center=_focus_center,
                focus_zoom=_focus_zoom,
                candidates_dfs=_fcat,
                activated_codes=_manual_activated,
                deactivated_codes=_manual_deactivated,
            )

        _mevent = st.plotly_chart(
            _mfig, key="manual_mesh_map",
            use_container_width=True,
            on_select="rerun",
            selection_mode=["points"],
        )

        if _mevent and _mevent.selection and _mevent.selection.points:
            for _pt in _mevent.selection.points:
                _cd = _pt.get("customdata")
                if _cd and len(_cd) >= 2:
                    _mc   = str(_cd[1])
                    _kind = str(_cd[2]) if len(_cd) >= 3 else "c"
                    if _kind == "v":
                        if _mc in _manual_deactivated: _manual_deactivated.discard(_mc)
                        else: _manual_deactivated.add(_mc)
                    else:
                        if _mc in _manual_activated: _manual_activated.discard(_mc)
                        else: _manual_activated.add(_mc)
            st.session_state["manual_activated"]   = _manual_activated
            st.session_state["manual_deactivated"] = _manual_deactivated
            st.rerun()

        if _manual_activated or _manual_deactivated:
            _mc1, _mc2, _mc3, _mc4 = st.columns([3, 1, 1.2, 1])
            with _mc1:
                _info_parts = []
                if _manual_activated:   _info_parts.append(f"追加: {len(_manual_activated)}件")
                if _manual_deactivated: _info_parts.append(f"取り消し: {len(_manual_deactivated)}件")
                st.info(" ／ ".join(_info_parts))
            with _mc4:
                if st.button("🔄 全リセット", key="manual_reset_btn"):
                    st.session_state["manual_activated"]   = set()
                    st.session_state["manual_deactivated"] = set()
                    st.rerun()
            with _mc3:
                if _focus_sel != "全表示":
                    if st.button("↩️ この面のみリセット", key="manual_reset_one_btn"):
                        _r_idx = _focus_opts.index(_focus_sel) - 1
                        _r_codes_act   = set()
                        _r_codes_deact = set()
                        if all_candidates and _r_idx < len(all_candidates):
                            _rcd = all_candidates[_r_idx]
                            if _rcd is not None and not _rcd.empty:
                                _r_codes_act = set(_rcd["mesh_code"].tolist())
                        _r_av = st.session_state.get("all_visible", [])
                        if _r_idx < len(_r_av) and _r_av[_r_idx] is not None and not _r_av[_r_idx].empty:
                            _r_codes_deact = set(_r_av[_r_idx]["mesh_code"].tolist())
                        _manual_activated   -= _r_codes_act
                        _manual_deactivated -= _r_codes_deact
                        st.session_state["manual_activated"]   = _manual_activated
                        st.session_state["manual_deactivated"] = _manual_deactivated
                        st.rerun()
            with _mc2:
                if st.button("✅ FIX（手動補正を確定）", type="primary", key="manual_fix_btn"):
                    _new_av = list(all_visible)
                    if _manual_deactivated:
                        for _bbi in range(len(_new_av)):
                            if _new_av[_bbi] is not None and not _new_av[_bbi].empty:
                                _new_av[_bbi] = _new_av[_bbi][
                                    ~_new_av[_bbi]["mesh_code"].isin(_manual_deactivated)
                                ]
                    for _bbi, _bb in enumerate(bb_list):
                        if all_candidates and _bbi < len(all_candidates):
                            _cdf = all_candidates[_bbi]
                            if _cdf is not None and not _cdf.empty:
                                _to_add = _cdf[_cdf["mesh_code"].isin(_manual_activated)]
                                if not _to_add.empty:
                                    _existing = _new_av[_bbi] if _new_av[_bbi] is not None else pd.DataFrame()
                                    _merged = pd.concat([_existing, _to_add], ignore_index=True).drop_duplicates("mesh_code")
                                    _new_av[_bbi] = _merged
                    st.session_state["all_visible"] = _new_av
                    st.session_state["result_df"] = (
                        pd.concat(_new_av, ignore_index=True)
                        if any(v is not None and not v.empty for v in _new_av)
                        else pd.DataFrame()
                    )
                    st.session_state["manual_activated"]   = set()
                    st.session_state["manual_deactivated"] = set()
                    st.session_state["all_candidates"]     = None
                    st.rerun()

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
            st.session_state.pop("_excl_last_clk", None)
            st.session_state.pop("_excl_focus_prev", None)
            st.rerun()
    if _excl_set:
        with _ec2:
            st.info(f"除外済み建物: {len(_excl_set)} 棟")
        with _ec3:
            if st.button("🗑️ 除外リストをクリア", key="clear_excl", use_container_width=True):
                st.session_state[_excl_key] = set()
                st.session_state.pop("excl_applied", None)
                st.rerun()

    _buildings_orig = st.session_state.get("buildings_orig", buildings_calc)

    if _excl_mode:
        if _buildings_orig is None or _buildings_orig.empty:
            st.warning("建物データがありません。建物データありで計算した場合のみ除外補正が使用できます。")
        elif not _FOLIUM_OK:
            st.warning("folium / streamlit-folium が未インストールです。")
        else:
            _bldg_idx_in_area: set = set()
            for _ebb in bb_list:
                _ec = create_circle(
                    _ebb["latitude"], _ebb["longitude"],
                    float(_ebb.get("radius", DEFAULT_RADIUS_M)),
                )
                _hits = _buildings_orig[
                    _buildings_orig.geometry.intersects(_ec.buffer(0.00005))
                ].index.tolist()
                _bldg_idx_in_area.update(_hits)
            _bldgs_in_area = _buildings_orig.loc[sorted(_bldg_idx_in_area)].copy()

            st.caption(
                f"対象エリア内の建物: {len(_bldgs_in_area):,} 棟　｜　"
                "🔴 赤 = 除外済み（クリックで復活）　🔵 青 = 計算対象（クリックで除外）"
            )

            _excl_focus_opts = ["全表示"] + [
                str(_ebb["site_id"]) for _ebb in bb_list
            ]
            _excl_focus_sel = st.selectbox("🎯 フォーカス", _excl_focus_opts, key="excl_map_focus")
            if _excl_focus_sel != "全表示":
                _excl_fi     = _excl_focus_opts.index(_excl_focus_sel) - 1
                _ecenter_lat = float(bb_list[_excl_fi]["latitude"])
                _ecenter_lon = float(bb_list[_excl_fi]["longitude"])
                _ezoom = 17
            else:
                _ecenter_lat = np.mean([_b["latitude"]  for _b in bb_list])
                _ecenter_lon = np.mean([_b["longitude"] for _b in bb_list])
                _ezoom = 16

            _efm_key = f"excl_folium_{_excl_focus_sel}"
            if st.session_state.get("_excl_focus_prev") != _excl_focus_sel:
                st.session_state["_excl_focus_prev"] = _excl_focus_sel
                st.session_state.pop("_excl_last_clk", None)

            _efm = folium.Map(location=[_ecenter_lat, _ecenter_lon],
                              zoom_start=_ezoom, tiles="OpenStreetMap")
            for _bidx, _brow in _bldgs_in_area.iterrows():
                _bgeom = _brow["geometry"]
                _bpolys = (list(_bgeom.geoms) if _bgeom.geom_type.startswith("Multi") else [_bgeom])
                for _bpoly in _bpolys:
                    if _bpoly.geom_type != "Polygon":
                        continue
                    _bcol = "red" if _bidx in _excl_set else "blue"
                    folium.Polygon(
                        locations=[[p[1], p[0]] for p in _bpoly.exterior.coords],
                        color=_bcol, fill=True, fill_opacity=0.5, weight=1,
                        tooltip=f"建物 #{_bidx} ({_brow.get('height', 0):.1f}m)",
                    ).add_to(_efm)

            _emap_res = st_folium(_efm, key=_efm_key, height=400,
                                  use_container_width=True, returned_objects=["last_clicked"])

            _eclk = (_emap_res or {}).get("last_clicked")
            if _eclk and _eclk != st.session_state.get("_excl_last_clk"):
                st.session_state["_excl_last_clk"] = _eclk
                _eclk_pt = Point(_eclk["lng"], _eclk["lat"])
                for _bidx, _brow in _bldgs_in_area.iterrows():
                    if _brow["geometry"].distance(_eclk_pt) < 0.0002:
                        if _bidx in _excl_set:
                            _excl_set.discard(_bidx)
                        else:
                            _excl_set.add(_bidx)
                        st.session_state[_excl_key] = _excl_set
                        break
                st.rerun()

            if _excl_set:
                if st.button("🔁 除外建物を反映して再計算", type="primary",
                             key="apply_excl", use_container_width=True):
                    _excl_bldgs = buildings_calc[~buildings_calc.index.isin(_excl_set)]
                    _rnew_av    = [None] * len(bb_list)
                    _rnew_cdf   = [None] * len(bb_list)
                    _rnew_sec   = [None] * len(bb_list)
                    _rlog       = st.empty()
                    _rlog.info("再計算中...")
                    with ThreadPoolExecutor(max_workers=min(len(bb_list), 6)) as _rex:
                        _rfuts = {_rex.submit(
                            lambda args: (args[0], compute_visibility(args[1], _excl_bldgs)),
                            (idx, bb)
                        ): idx for idx, bb in enumerate(bb_list)}
                        for _rfut in as_completed(_rfuts):
                            _ri, (_rvdf, _rcdf, _rsec, _) = _rfut.result()
                            _rnew_av[_ri]  = _rvdf
                            _rnew_cdf[_ri] = _rcdf
                            _rnew_sec[_ri] = _rsec
                    st.session_state["all_visible"]    = _rnew_av
                    st.session_state["all_candidates"] = _rnew_cdf
                    st.session_state["all_sectors"]    = _rnew_sec
                    st.session_state["result_df"] = (
                        pd.concat(_rnew_av, ignore_index=True)
                        if any(v is not None and not v.empty for v in _rnew_av)
                        else pd.DataFrame()
                    )
                    st.session_state["excl_applied"]      = frozenset(_excl_set)
                    st.session_state["buildings_calc"]     = _excl_bldgs
                    st.session_state.pop("manual_activated",   None)
                    st.session_state.pop("manual_deactivated", None)
                    _rlog.success("✅ 再計算完了")
                    st.rerun()

    # ── メッシュコード CSV ダウンロード ────────────────────────────────────────
    st.divider()
    st.subheader("⬇️ メッシュコード CSV ダウンロード（Site ID別）")
    st.caption("同一 Site ID の複数行を統合・重複除去・昇順ソート済み。ヘッダーなし。")

    _sid_meshes_dl: dict = {}
    for _bb, _vdf in zip(bb_list, all_visible):
        _sid = str(_bb["site_id"])
        if _vdf is not None and not _vdf.empty:
            _sid_meshes_dl.setdefault(_sid, []).append(_vdf["mesh_code"])

    _sid_merged: dict = {}
    for _sid, _parts in _sid_meshes_dl.items():
        _merged = pd.concat(_parts, ignore_index=True).drop_duplicates().sort_values().reset_index(drop=True)
        _sid_merged[_sid] = _merged

    if not _sid_merged:
        st.warning("有効メッシュが 0 件でした。設定を見直してください。")
    else:
        _unique_sids = list(_sid_merged.keys())
        _dl_cols = st.columns(min(len(_unique_sids), 4))
        for _ci, _sid in enumerate(_unique_sids):
            _codes    = _sid_merged[_sid]
            _mesh_csv = _codes.to_csv(index=False, header=False).encode("utf-8")
            with _dl_cols[_ci % min(len(_unique_sids), 4)]:
                st.download_button(
                    label=f"⬇️ No.{_sid}.csv ({len(_codes):,}件)",
                    data=_mesh_csv,
                    file_name=f"No.{_sid}.csv",
                    mime="text/csv",
                    key=f"dl_{_sid}",
                    type="primary",
                    use_container_width=True,
                )

        _zip_buf = io.BytesIO()
        with zipfile.ZipFile(_zip_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
            for _sid, _codes in _sid_merged.items():
                _zf.writestr(f"No.{_sid}.csv", _codes.to_csv(index=False, header=False))
        _zip_buf.seek(0)
        st.download_button(
            label="⬇️ 全 Site ID を一括ダウンロード (ZIP)",
            data=_zip_buf.getvalue(),
            file_name="mesh_codes_all.zip",
            mime="application/zip",
            key="dl_all_zip",
            type="secondary",
            use_container_width=False,
        )
