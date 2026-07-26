"""
広告面板 建物フットプリント視認エリア計算アプリ
Plateau CityGML × 10次メッシュ（建物フットプリント版）
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
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
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
VISIBLE_RATIO_THRESHOLD = 0.50   # 建物ポリゴンとのオーバーラップ率閾値
MAX_SITES               = 30
BLDG_FETCH_BUFFER_DEG   = 500.0 / 111320.0   # CityGML取得用バッファ半径（約500m）
BLDG_FIND_BUFFER_DEG    = 50.0  / 111320.0   # 建物検索バッファ（約50m）

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


def local_scale(lat: float):
    return 111320.0, 111320.0 * math.cos(math.radians(lat))


# ─────────────────────────────────────────────────────────────────────────────
# 再利用モード: メッシュコードから結果を再構築
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_from_meshes(bb_list: list, sid_to_meshes: dict) -> tuple:
    """
    ZIPから読み込んだメッシュコードリストで
    all_visible / all_polygons / all_candidates を再構築する。
    all_polygons: 可視メッシュセルのunion（近似建物ポリゴン）
    all_candidates: 可視メッシュ拡張バッファ内の未登録メッシュ
    """
    all_visible:    list = []
    all_polygons:   list = []
    all_candidates: list = []
    lat_sz, lon_sz = mesh10_cell_size()

    for bb in bb_list:
        sid    = str(bb["site_id"])
        lat    = float(bb["latitude"])
        lon    = float(bb["longitude"])
        lat_sc, lon_sc = local_scale(lat)

        visible_codes = set(sid_to_meshes.get(sid, []))

        vis_rows   = []
        mesh_cells = []
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
            lo = clon - lon_sz / 2
            la = clat - lat_sz / 2
            mesh_cells.append(box(lo, la, lo + lon_sz, la + lat_sz))

        all_visible.append(pd.DataFrame(vis_rows) if vis_rows else pd.DataFrame())

        # 近似ポリゴン: 可視メッシュセルのunion
        if mesh_cells:
            try:
                approx_poly = unary_union(mesh_cells)
            except Exception:
                approx_poly = None
        else:
            approx_poly = None
        all_polygons.append(approx_poly)

        # 候補メッシュ: 可視メッシュのバウンディングボックス+2セル内の未登録メッシュ
        cand_rows = []
        if vis_rows:
            min_la = min(r["center_lat"] for r in vis_rows) - lat_sz * 2
            max_la = max(r["center_lat"] for r in vis_rows) + lat_sz * 2
            min_lo = min(r["center_lon"] for r in vis_rows) - lon_sz * 2
            max_lo = max(r["center_lon"] for r in vis_rows) + lon_sz * 2
            for _la in np.arange(math.floor(min_la / lat_sz) * lat_sz, max_la + lat_sz, lat_sz):
                for _lo in np.arange(math.floor(min_lo / lon_sz) * lon_sz, max_lo + lon_sz, lon_sz):
                    _code = encode_mesh10(_la + lat_sz / 2, _lo + lon_sz / 2)
                    if _code in visible_codes:
                        continue
                    clat_ = _la + lat_sz / 2
                    clon_ = _lo + lon_sz / 2
                    dx_m  = (clon_ - lon) * lon_sc
                    dy_m  = (clat_ - lat) * lat_sc
                    cand_rows.append({
                        "billboard_id": sid,
                        "mesh_code":    _code,
                        "center_lat":   round(clat_, 8),
                        "center_lon":   round(clon_, 8),
                        "distance_m":   round(math.sqrt(dx_m ** 2 + dy_m ** 2), 1),
                        "area_ratio":   0.5,
                    })
        all_candidates.append(pd.DataFrame(cand_rows) if cand_rows else pd.DataFrame())

    return all_visible, all_polygons, all_candidates


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
    return [tuple(vals[i:i + dim]) for i in range(0, len(vals) - dim + 1, dim)]


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

        geom = polys[0]
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
    url     = "https://www.geospatial.jp/ckan/api/3/action/package_search?q=plateau&rows=1000"
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
            name   = res.get("name", "")
            rurl   = res.get("url", "")
            name_l = name.lower()
            if "citygml" in name_l and rurl.lower().endswith(".zip"):
                if "v4" in name_l:   v4_url       = v4_url       or rurl
                elif "v3" in name_l: v3_url        = v3_url       or rurl
                else:                fallback_url  = fallback_url  or rurl
        return v4_url or v3_url or fallback_url
    except Exception:
        return None


def _read_zip_cd(zip_url: str) -> dict:
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
        if cd_data[pos:pos + 4] != b"PK\x01\x02":
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
    """各サイト座標のバッファ（約500m）に必要な3次メッシュコード（8桁）セットを返す"""
    lat_sz_3 = (2.0 / 3.0) / 8 / 10
    lon_sz_3 = 1.0 / 8 / 10
    prefixes = set()
    for _, bb in billboards_df.iterrows():
        buf = Point(bb.longitude, bb.latitude).buffer(BLDG_FETCH_BUFFER_DEG)
        minlon, minlat, maxlon, maxlat = buf.bounds
        la = math.floor(minlat / lat_sz_3) * lat_sz_3
        while la <= maxlat:
            lo = math.floor(minlon / lon_sz_3) * lon_sz_3
            while lo <= maxlon:
                if buf.intersects(box(lo, la, lo + lon_sz_3, la + lat_sz_3)):
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

    log("📍 サイトの市区町村を特定中...")
    muni_cds = set()
    _unique_coords = list({(round(row.latitude, 3), round(row.longitude, 3))
                           for _, row in billboards_df.iterrows()})
    log(f"   {len(billboards_df)}件 → {len(_unique_coords)}ユニーク座標でジオコーディング")
    with ThreadPoolExecutor(max_workers=min(len(_unique_coords), 6)) as _ex:
        for muni_cd in _ex.map(lambda p: _gsi_reverse_geocode(*p), _unique_coords):
            if muni_cd:
                muni_cds.add(muni_cd)
    if not muni_cds:
        log("❌ 市区町村コードを取得できませんでした")
        return None
    log(f"✅ 市区町村コード: {', '.join(sorted(muni_cds))}")

    log("🗺️ 必要なメッシュタイルを計算中...")
    needed_prefixes = get_needed_3rd_mesh_prefixes(billboards_df)
    log(f"✅ 対象3次メッシュ: {', '.join(sorted(needed_prefixes))}（{len(needed_prefixes)} タイル）")

    all_gdfs = []
    for muni_cd in sorted(muni_cds):
        dataset_id = catalog.get(muni_cd) or catalog.get(muni_cd[:4] + "0")
        if not dataset_id:
            log(f"⚠️ 市区町村 {muni_cd} のPlateauデータが見つかりません")
            continue

        log(f"🔍 `{dataset_id}` のZIP URLを取得中...")
        zip_url = _get_plateau_zip_url(dataset_id)
        if not zip_url:
            log(f"⚠️ `{dataset_id}` のZIP URLが取得できませんでした")
            continue

        log("📦 ZIPインデックスを解析中...")
        try:
            cd = _read_zip_cd(zip_url)
        except Exception as e:
            log(f"❌ ZIP解析エラー: {e}")
            continue

        needed = {
            fname: info for fname, info in cd.items()
            if any(fname.startswith(p) for p in needed_prefixes)
        }
        if not needed:
            log("⚠️ 対象メッシュのGMLがZIP内に見つかりませんでした")
            continue

        log(f"⬇️ {len(needed)} 個のGMLファイルをダウンロード中（並列）...")

        def _fetch_one_gml(item):
            fname, (local_off, comp_size, method) = item
            gml_bytes = _extract_gml_from_zip(zip_url, local_off, comp_size, method)
            return fname, parse_citygml(gml_bytes)

        with ThreadPoolExecutor(max_workers=min(len(needed), 6)) as _gex:
            _gfuts = {_gex.submit(_fetch_one_gml, item): item[0]
                      for item in needed.items()}
            for _fut in as_completed(_gfuts):
                _fname = _gfuts[_fut]
                try:
                    _fn, gdf = _fut.result()
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

    # バッファ外の建物を除去
    _area_union = None
    for _, _bb in billboards_df.iterrows():
        _buf = Point(_bb.longitude, _bb.latitude).buffer(BLDG_FETCH_BUFFER_DEG)
        _area_union = _buf if _area_union is None else _area_union.union(_buf)
    if _area_union is not None:
        before = len(combined)
        combined = combined[
            combined.geometry.intersects(_area_union.buffer(0.0001))
        ].reset_index(drop=True)
        log(f"✂️ バッファ外を除去: {before:,} → {len(combined):,} 棟")

    log(f"\n✅ **取得完了: 建物 {len(combined):,} 棟**")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 建物検索
# ─────────────────────────────────────────────────────────────────────────────

def get_building_at_point(lat: float, lon: float,
                          buildings_gdf: gpd.GeoDataFrame):
    """
    指定座標を含む建物ポリゴンを返す。
    含む建物がなければ最近傍（50m以内）を返す。
    見つからない場合は None を返す。
    """
    pt = Point(lon, lat)

    # sindex があれば高速化
    try:
        sindex = buildings_gdf.sindex
        cands_idx = list(sindex.intersection(pt.buffer(BLDG_FIND_BUFFER_DEG).bounds))
        cands = buildings_gdf.iloc[cands_idx]
    except Exception:
        cands = buildings_gdf

    # 点を含む建物を優先
    contained = cands[cands.geometry.contains(pt)]
    if not contained.empty:
        return contained.iloc[0]["geometry"]

    # 含む建物がなければ最近傍
    dists = cands.geometry.distance(pt)
    if dists.empty:
        return None
    nearest_idx = dists.idxmin()
    if dists[nearest_idx] <= BLDG_FIND_BUFFER_DEG:
        return cands.loc[nearest_idx]["geometry"]

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 視認計算（建物フットプリント = 視認エリア）
# ─────────────────────────────────────────────────────────────────────────────

def compute_visibility(bb: dict,
                       buildings_gdf: Optional[gpd.GeoDataFrame]) -> tuple:
    """
    建物フットプリント視認エリア計算。
    - 視認エリア = 入力座標が所在する建物ポリゴン
    - メッシュ判定: 建物ポリゴンとのオーバーラップ率 >= VISIBLE_RATIO_THRESHOLD → 有効
    - LOS計算なし（建物自体が視認エリアのため）
    """
    lat = bb["latitude"]
    lon = bb["longitude"]
    sid = bb.get("site_id", "B001")
    lat_sc, lon_sc = local_scale(lat)

    if buildings_gdf is None or buildings_gdf.empty:
        return pd.DataFrame(), pd.DataFrame(), None, 0

    bldg_poly = get_building_at_point(lat, lon, buildings_gdf)
    if bldg_poly is None:
        return pd.DataFrame(), pd.DataFrame(), None, 0

    lat_sz, lon_sz = mesh10_cell_size()
    mesh_area = lat_sz * lon_sz
    minlon, minlat, maxlon, maxlat = bldg_poly.bounds

    visible_rows   = []
    candidate_rows = []
    total          = 0

    for la in np.arange(math.floor(minlat / lat_sz) * lat_sz,
                        maxlat + lat_sz, lat_sz):
        for lo in np.arange(math.floor(minlon / lon_sz) * lon_sz,
                            maxlon + lon_sz, lon_sz):
            mbox = box(lo, la, lo + lon_sz, la + lat_sz)
            if not bldg_poly.intersects(mbox):
                continue
            total += 1
            inter = bldg_poly.intersection(mbox)
            if inter.is_empty:
                continue
            ratio    = inter.area / mesh_area
            clat     = la + lat_sz / 2
            clon     = lo + lon_sz / 2
            dx_m     = (clon - lon) * lon_sc
            dy_m     = (clat - lat) * lat_sc
            dist_m   = math.sqrt(dx_m ** 2 + dy_m ** 2)
            code     = encode_mesh10(clat, clon)
            row_data = {
                "billboard_id": sid,
                "mesh_code":    code,
                "center_lat":   round(clat, 8),
                "center_lon":   round(clon, 8),
                "distance_m":   round(dist_m, 1),
                "area_ratio":   round(ratio, 3),
            }
            if ratio >= VISIBLE_RATIO_THRESHOLD:
                visible_rows.append(row_data)
            elif ratio >= 0.01:
                candidate_rows.append(row_data)

    return (pd.DataFrame(visible_rows), pd.DataFrame(candidate_rows),
            bldg_poly, total)


# ─────────────────────────────────────────────────────────────────────────────
# 地図生成
# ─────────────────────────────────────────────────────────────────────────────

def _draw_polygon_traces(fig, poly, color, name, fill_opacity=0.15):
    """Polygon または MultiPolygon を Scattermapbox に追加する"""
    geoms = list(poly.geoms) if poly.geom_type.startswith("Multi") else [poly]
    all_lons, all_lats = [], []
    for g in geoms:
        if g.geom_type != "Polygon":
            continue
        xs, ys = g.exterior.xy
        all_lons.extend(list(xs) + [None])
        all_lats.extend(list(ys) + [None])
    if all_lons:
        fig.add_trace(go.Scattermapbox(
            lat=all_lats, lon=all_lons,
            mode="lines", fill="toself",
            fillcolor=_hex_to_rgba(color.lstrip("#") and color, fill_opacity),
            line=dict(color=color, width=2),
            name=name,
            hoverinfo="skip",
        ))


def build_map(billboards: list,
              polygons: list,
              visible_dfs: list,
              buildings_gdf: Optional[gpd.GeoDataFrame],
              mesh_colors: Optional[dict] = None,
              focus_center: Optional[tuple] = None,
              focus_zoom: int = 17,
              candidates_dfs=None,
              activated_codes=None,
              deactivated_codes=None) -> go.Figure:
    fig = go.Figure()

    # 建物レイヤー（高さ別色分け）
    if buildings_gdf is not None and not buildings_gdf.empty:
        _area_union = None
        for _bb in billboards:
            _buf = Point(_bb["longitude"], _bb["latitude"]).buffer(BLDG_FETCH_BUFFER_DEG)
            _area_union = _buf if _area_union is None else _area_union.union(_buf)
        if _area_union is not None:
            buildings_gdf = buildings_gdf[
                buildings_gdf.geometry.intersects(_area_union.buffer(0.00005))
            ]

    if buildings_gdf is not None and not buildings_gdf.empty:
        max_h     = buildings_gdf["height"].quantile(0.95) or 1
        bins      = [0, max_h * 0.2, max_h * 0.4, max_h * 0.6, max_h * 0.8, float("inf")]
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
            all_lons_b, all_lats_b = [], []
            for geom in subset["geometry"]:
                polys_b = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
                for p in polys_b:
                    if p.geom_type != "Polygon":
                        continue
                    xs, ys = p.exterior.xy
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

    for idx, (bb, poly, vdf) in enumerate(zip(billboards, polygons, visible_dfs)):
        color = COLORS[idx % len(COLORS)]
        sid   = bb.get("site_id", f"B{idx+1}")

        # 建物ポリゴン描画
        if poly is not None:
            geoms = list(poly.geoms) if poly.geom_type.startswith("Multi") else [poly]
            all_plons, all_plats = [], []
            for g in geoms:
                if g.geom_type != "Polygon":
                    continue
                xs, ys = g.exterior.xy
                all_plons.extend(list(xs) + [None])
                all_plats.extend(list(ys) + [None])
            if all_plons:
                fig.add_trace(go.Scattermapbox(
                    lat=all_plats, lon=all_plons,
                    mode="lines", fill="toself",
                    fillcolor="rgba(255,220,0,0.12)",
                    line=dict(color=color, width=2.5),
                    name=f"{sid} 建物フットプリント",
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
                           f"建物内面積比: {row['area_ratio']*100:.1f}%")
                    box_lats.extend([la0, la0, la0 + lat_sz, la0 + lat_sz, la0, None])
                    box_lons.extend([lo0, lo0 + lon_sz, lo0 + lon_sz, lo0, lo0, None])
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
                    d_lats.extend([la0, la0, la0 + lat_sz, la0 + lat_sz, la0, None])
                    d_lons.extend([lo0, lo0 + lon_sz, lo0 + lon_sz, lo0, lo0, None])
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
                        _cbox_lats += [_cla - lat_sz/2, _cla + lat_sz/2, _cla + lat_sz/2,
                                       _cla - lat_sz/2, _cla - lat_sz/2, None]
                        _cbox_lons += [_clo - lon_sz/2, _clo - lon_sz/2, _clo + lon_sz/2,
                                       _clo + lon_sz/2, _clo - lon_sz/2, None]
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
                _actdf   = cdf[cdf["mesh_code"].isin(_act_set)]
                if not _actdf.empty:
                    _abox_lats, _abox_lons = [], []
                    _act_col = mesh_colors.get(idx, color) if mesh_colors else color
                    for _, _am in _actdf.iterrows():
                        _ala, _alo = _am["center_lat"], _am["center_lon"]
                        _abox_lats += [_ala - lat_sz/2, _ala + lat_sz/2, _ala + lat_sz/2,
                                       _ala - lat_sz/2, _ala - lat_sz/2, None]
                        _abox_lons += [_alo - lon_sz/2, _alo - lon_sz/2, _alo + lon_sz/2,
                                       _alo + lon_sz/2, _alo - lon_sz/2, None]
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
                        marker=dict(size=14,
                                    color=_act_col if _act_col.startswith("#") else color,
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
            name=str(sid),
            hovertemplate=f"<b>{sid}</b><extra></extra>",
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

st.set_page_config(page_title="建物フットプリント視認エリア解析システム",
                   page_icon="🏢", layout="wide")


def _check_password():
    if st.session_state.get("authenticated"):
        return
    st.title("🔒 ログインが必要です")
    st.caption("このアプリを利用するにはパスワードが必要です。")
    pwd = st.text_input("パスワード", type="password", key="pwd_input")
    if st.button("ログイン", type="primary"):
        _app_pwd = st.secrets.get("APP_PASSWORD", "")
        if not _app_pwd:
            st.error("⚠️ APP_PASSWORD が Secrets に設定されていません。")
        elif pwd == _app_pwd:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("パスワードが違います。")
    st.stop()


_check_password()

st.title("🏢 建物フットプリント視認エリア解析システム")
st.caption("Plateau CityGML × 10次メッシュ（入力座標が所在する建物フットプリントを視認エリアとして解析）")

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
        st.markdown("**必須列**: `site_id`, `latitude`, `longitude`")
        bb_file = st.file_uploader("CSVをアップロード", type=["csv"], key="bb_csv")
    else:
        st.caption("1件の情報を入力してください。")
        with st.form("manual_bb_form"):
            sid_raw = st.text_input("Site ID", placeholder="例: 000001")
            m_lat   = st.number_input("緯度 latitude",  value=35.6815, format="%.6f")
            m_lon   = st.number_input("経度 longitude", value=139.7670, format="%.6f")
            submitted = st.form_submit_button("✅ 設定を反映", use_container_width=True)

        if submitted:
            sid_clean = str(sid_raw).strip()
            if not sid_clean:
                st.warning("⚠️ Site IDを入力してください。")
            else:
                st.session_state["manual_bb"] = {
                    "site_id":   sid_clean,
                    "latitude":  m_lat,
                    "longitude": m_lon,
                }

        if st.session_state.get("manual_bb"):
            manual_bb = st.session_state["manual_bb"]
            d = manual_bb
            st.success(
                f"✅ **{d['site_id']}** 設定済み  \n"
                f"緯度 {d['latitude']:.5f} / 経度 {d['longitude']:.5f}"
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
        help="「全 Site ID を一括ダウンロード」で取得したZIPファイルをアップロードしてください。",
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
                "使用しない: 建物が特定できないため計算できません"
            ),
        )
        if bldg_mode == "📂 手動アップロード":
            gml_file = st.file_uploader(
                "CityGML (.gml) をアップロード", type=["gml", "xml"], key="gml"
            )
        elif bldg_mode == "🚀 Plateau から自動取得":
            st.caption("CSVをアップロード後、ボタンでPlateauの建物データを自動ダウンロードします。")
            fetch_btn = st.button(
                "🏢 建物データを自動取得",
                disabled=(bb_file is None and not manual_bb),
                use_container_width=True,
                type="secondary",
            )
        else:
            st.warning("⚠️ 建物データなしでは建物を特定できません。")
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

# bb_df 構築
if _csv_mode:
    try:
        bb_df   = pd.read_csv(bb_file, dtype={"site_id": str})
        required = {"site_id", "latitude", "longitude"}
        missing  = required - set(bb_df.columns)
        if missing:
            st.error(f"CSVに必要な列がありません: {missing}")
            st.stop()
        if len(bb_df) > MAX_SITES:
            st.error(f"一度に処理できるサイトは最大 {MAX_SITES} 件です（現在 {len(bb_df)} 行）。")
            st.stop()
    except Exception as e:
        st.error(f"CSV読み込みエラー: {e}")
        st.stop()
else:
    bb_df = pd.DataFrame([manual_bb])

st.success(f"サイト {len(bb_df)} 件を読み込みました")

# サイト一覧テーブル
st.subheader("📋 サイト一覧")
st.dataframe(
    bb_df[["site_id", "latitude", "longitude"]],
    use_container_width=True,
    hide_index=True,
)

# ── 位置補正用 corrected_coords 初期化 ────────────────────────────────────────
_src_sig = bb_df[["site_id", "latitude", "longitude"]].to_csv(index=False)
if st.session_state.get("_corr_src") != _src_sig:
    st.session_state["_corr_src"] = _src_sig
    st.session_state["corrected_coords"] = {
        str(i): {"latitude": float(r["latitude"]), "longitude": float(r["longitude"])}
        for i, r in bb_df.iterrows()
    }
    if "finalized_master" in st.session_state:
        del st.session_state["finalized_master"]

_corr    = st.session_state["corrected_coords"]
bb_df_w  = bb_df.copy()
for _ci, _cr in bb_df_w.iterrows():
    _ckey = str(_ci)
    if _ckey in _corr:
        bb_df_w.at[_ci, "latitude"]  = _corr[_ckey]["latitude"]
        bb_df_w.at[_ci, "longitude"] = _corr[_ckey]["longitude"]

# 建物データ（手動アップロード）
if gml_file is not None:
    with st.spinner("CityGMLを解析中..."):
        try:
            bldgs = parse_citygml(gml_file.read())
            if bldgs.empty:
                st.warning("CityGMLから建物データを抽出できませんでした。")
            else:
                st.success(f"建物 {len(bldgs):,} 棟を読み込みました（高さ平均 {bldgs['height'].mean():.1f}m）")
                st.session_state["buildings_gdf"] = bldgs
        except Exception as e:
            st.error(f"CityGML解析エラー: {e}")

# 建物データ（自動取得）
if fetch_btn:
    st.subheader("🏢 建物データ自動取得ログ")
    log_box = st.empty()
    with st.spinner("Plateauから建物データを取得中..."):
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
    color = COLORS[idx % len(COLORS)]
    prev_fig.add_trace(go.Scattermapbox(
        lat=[row.latitude], lon=[row.longitude], mode="markers",
        marker=dict(size=13, color=color),
        name=str(row.site_id),
        hovertemplate=f"<b>{row.site_id}</b><br>{row.latitude:.6f}, {row.longitude:.6f}<extra></extra>",
    ))

_prev_focus_opts = ["全表示"] + [str(row.site_id) for _, row in bb_df_w.iterrows()]
_prev_focus_sel  = st.selectbox(
    "🎯 フォーカス", _prev_focus_opts, key="prev_map_focus",
)
if _prev_focus_sel != "全表示":
    _prev_idx  = _prev_focus_opts.index(_prev_focus_sel) - 1
    _prev_row  = bb_df_w.iloc[_prev_idx]
    center_lat = float(_prev_row["latitude"])
    center_lon = float(_prev_row["longitude"])
    _prev_zoom = 18
else:
    center_lat = bb_df_w["latitude"].mean()
    center_lon = bb_df_w["longitude"].mean()
    if len(bb_df_w) > 1:
        _span = max(bb_df_w["latitude"].max()  - bb_df_w["latitude"].min(),
                    bb_df_w["longitude"].max() - bb_df_w["longitude"].min(), 1e-6)
        _prev_zoom = int(np.clip(np.log2(180 / _span), 4, 15))
    else:
        _prev_zoom = 18

prev_fig.update_layout(
    mapbox=dict(style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon), zoom=_prev_zoom),
    height=400, margin=dict(r=0, t=0, l=0, b=0),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                bgcolor="rgba(255,255,255,0.88)"),
)
st.plotly_chart(prev_fig, use_container_width=True)
st.caption("▲ 各マーカーが入力座標です。下の補正マップで位置を調整できます。")

# ── 位置補正マップ ────────────────────────────────────────────────────────────
st.divider()
st.subheader("✏️ 位置補正マップ")

if not _FOLIUM_OK:
    st.warning("folium / streamlit-folium が未インストールです。")
else:
    _sel_opts = {str(i): str(r["site_id"]) for i, r in bb_df_w.iterrows()}
    _cc_map, _cc_ctrl = st.columns([3, 2])

    with _cc_ctrl:
        _sel = st.selectbox(
            "補正するサイトを選択",
            options=list(_sel_opts.keys()),
            format_func=lambda k: _sel_opts[k],
            key="corr_select",
        )

    _sel_center = _corr.get(_sel, {})
    _fm_lat = _sel_center.get("latitude",  bb_df_w["latitude"].mean())
    _fm_lon = _sel_center.get("longitude", bb_df_w["longitude"].mean())
    _fm = folium.Map(location=[_fm_lat, _fm_lon], zoom_start=18, tiles="OpenStreetMap")

    for _fi, _fr in bb_df_w.iterrows():
        _fi_key = str(_fi)
        _fsid   = str(_fr["site_id"])
        _flat   = _corr[_fi_key]["latitude"]
        _flon   = _corr[_fi_key]["longitude"]
        _fcolor = "red" if _fi_key == _sel else "blue"
        folium.Marker(
            location=[_flat, _flon],
            popup=_fsid,
            tooltip=f"{_fsid}（{_flat:.6f}, {_flon:.6f}）",
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
            if st.button(f"↩ {_sel_sid} の補正をリセット", key="reset_corr",
                         use_container_width=True):
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
    if st.button("✅ 位置を最終確定する", type="primary",
                 key="finalize_btn", use_container_width=True):
        _fdf = bb_df.copy()
        for _fi2, _fr2 in _fdf.iterrows():
            _fc = st.session_state.get("corrected_coords", {}).get(str(_fi2))
            if _fc:
                _fdf.at[_fi2, "latitude"]  = _fc["latitude"]
                _fdf.at[_fi2, "longitude"] = _fc["longitude"]
        st.session_state["finalized_master"] = _fdf

if "finalized_master" in st.session_state:
    _fmdf    = st.session_state["finalized_master"]
    _out_cols = [c for c in ["site_id", "latitude", "longitude"] if c in _fmdf.columns]
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

        _bb_recs       = bb_df_w.to_dict("records")
        _av, _ap, _ac  = reconstruct_from_meshes(_bb_recs, _sid_meshes)
        _rdf = (
            pd.concat(_av, ignore_index=True)
            if any(not v.empty for v in _av) else pd.DataFrame()
        )
        _total_meshes = sum(len(v) for v in _av if not v.empty)
        st.session_state["result_df"]       = _rdf
        st.session_state["all_visible"]     = _av
        st.session_state["all_polygons"]    = _ap
        st.session_state["all_candidates"]  = _ac
        st.session_state["bb_list"]         = _bb_recs
        st.session_state["buildings_calc"]  = None
        st.session_state["reuse_zip_hash"]  = _reuse_hash
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
    if bldg_mode == "⛔ 使用しない" or buildings_gdf is None:
        st.error("❌ 建物データが必要です。建物データを取得または手動アップロードしてから計算してください。")
        st.stop()

    _n_bb    = len(bb_df_w)
    prog_bar = st.progress(0, text="計算を開始しています...")
    all_visible    = [None] * _n_bb
    all_polygons   = [None] * _n_bb
    all_candidates = [None] * _n_bb
    _done = [0]

    def _calc_one(args):
        idx, bb = args
        return idx, compute_visibility(bb, buildings_gdf)

    with ThreadPoolExecutor(max_workers=min(_n_bb, 6)) as _ex:
        _futs = {_ex.submit(_calc_one, (idx, row.to_dict())): idx
                 for idx, (_, row) in enumerate(bb_df_w.iterrows())}
        for _fut in as_completed(_futs):
            idx, (vdf, cdf, poly, _) = _fut.result()
            all_visible[idx]    = vdf
            all_candidates[idx] = cdf
            all_polygons[idx]   = poly
            _done[0] += 1
            prog_bar.progress(_done[0] / _n_bb, text=f"{_done[0]}/{_n_bb} 件完了")

    prog_bar.progress(1.0, text="完了！")

    # 建物が見つからなかったサイトを警告
    _no_bldg = [
        str(bb_df_w.iloc[i]["site_id"])
        for i, p in enumerate(all_polygons) if p is None
    ]
    if _no_bldg:
        st.warning(
            f"⚠️ 以下のサイトで建物が見つかりませんでした（座標を確認してください）: "
            f"{', '.join(_no_bldg)}"
        )

    result_df = (
        pd.concat(all_visible, ignore_index=True)
        if any(v is not None and not v.empty for v in all_visible)
        else pd.DataFrame()
    )
    st.session_state["result_df"]       = result_df
    st.session_state["all_visible"]     = all_visible
    st.session_state["all_polygons"]    = all_polygons
    st.session_state["all_candidates"]  = all_candidates
    st.session_state["bb_list"]         = bb_df_w.to_dict("records")
    st.session_state["buildings_calc"]  = buildings_gdf
    st.session_state.pop("manual_activated",   None)
    st.session_state.pop("manual_deactivated", None)

# ── 結果表示 ─────────────────────────────────────────────────────────────────
if "result_df" in st.session_state:
    result_df      = st.session_state["result_df"]
    all_visible    = st.session_state["all_visible"]
    all_polygons   = st.session_state["all_polygons"]
    bb_list        = st.session_state["bb_list"]
    buildings_calc = st.session_state["buildings_calc"]

    st.divider()
    st.subheader("📊 計算結果")

    cols = st.columns(min(len(bb_list), 4))
    for _ci, _bb in enumerate(bb_list):
        _vdf  = all_visible[_ci] if all_visible[_ci] is not None else pd.DataFrame()
        _poly = all_polygons[_ci]
        _area_m2 = _poly.area * (111320.0 ** 2) if _poly is not None else 0
        with cols[_ci % min(len(bb_list), 4)]:
            st.metric(
                label=str(_bb["site_id"]),
                value=f"{len(_vdf):,} メッシュ",
                delta=f"建物面積 約{_area_m2:.0f}m²" if _area_m2 > 0 else "建物未検出",
            )

    # 地図表示設定
    with st.expander("🎛️ 地図表示設定", expanded=True):
        _focus_opts = ["全表示"] + [str(_bb["site_id"]) for _bb in bb_list]
        _focus_sel = st.selectbox(
            "🎯 フォーカス",
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
    _focus_zoom   = 18
    if _focus_sel != "全表示":
        _focus_idx    = _focus_opts.index(_focus_sel) - 1
        _focus_bb     = bb_list[_focus_idx]
        _focus_center = (_focus_bb["latitude"], _focus_bb["longitude"])

    _fbb   = [bb  for i, bb  in enumerate(bb_list)                           if _show.get(i, True)]
    _fvis  = [vdf for i, (bb, vdf) in enumerate(zip(bb_list, all_visible))   if _show.get(i, True)]
    _fpoly = [p   for i, (bb, p)   in enumerate(zip(bb_list, all_polygons))  if _show.get(i, True)]
    _fmcols = {new_i: _mcols[old_i]
               for new_i, old_i in enumerate(i for i, _ in enumerate(bb_list) if _show.get(i, True))}

    with st.spinner("地図を生成中..."):
        fig = build_map(_fbb, _fpoly, _fvis, buildings_calc,
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
        _fcat      = [all_candidates[old_i] if old_i < len(all_candidates) else None
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
                _fbb, _fpoly, _fvis, buildings_calc,
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
                                _to_add  = _cdf[_cdf["mesh_code"].isin(_manual_activated)]
                                if not _to_add.empty:
                                    _existing = _new_av[_bbi] if _new_av[_bbi] is not None else pd.DataFrame()
                                    _merged   = pd.concat([_existing, _to_add], ignore_index=True).drop_duplicates("mesh_code")
                                    _new_av[_bbi] = _merged
                    st.session_state["all_visible"] = _new_av
                    st.session_state["result_df"]   = (
                        pd.concat(_new_av, ignore_index=True)
                        if any(v is not None and not v.empty for v in _new_av)
                        else pd.DataFrame()
                    )
                    st.session_state["manual_activated"]   = set()
                    st.session_state["manual_deactivated"] = set()
                    st.session_state["all_candidates"]     = None
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
        st.warning("有効メッシュが 0 件でした。座標が建物内にあるか確認してください。")
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
