# grouping.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

import folium
from folium.plugins import Search

from sklearn.cluster import MiniBatchKMeans
from shapely.geometry import MultiPoint


# -----------------------------
# Labels
# -----------------------------
def label_from_gidx(g: int) -> str:
    return f"R{int(g) + 1:02d}"


def parse_label_to_gidx(label: str) -> int:
    label = str(label).strip().upper()
    if not label.startswith("R"):
        raise ValueError("Format grup harus seperti R01, R02, dst.")
    return int(label.replace("R", "")) - 1


# -----------------------------
# Validation
# -----------------------------
def _to_float_series(s: pd.Series) -> pd.Series:
    # Accept comma decimal "106,88" -> "106.88"
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def validate_input_df(df: pd.DataFrame) -> Tuple[bool, str, Optional[pd.DataFrame], Dict[str, int]]:
    """
    Required columns (case-insensitive): nama_toko, lat, long
    Returns:
      ok, user_friendly_message, df_clean, stats
    df_clean columns: nama_toko, lat, long, _row_id
    """
    stats = {"rows_in": 0, "rows_valid": 0, "rows_dropped": 0}

    if df is None or df.empty:
        return False, (
            "File kamu kosong atau tidak ada datanya.\n\n"
            "✅ Pastikan ada kolom: **nama_toko**, **lat**, **long**."
        ), None, stats

    stats["rows_in"] = int(len(df))

    cols = {c.lower().strip(): c for c in df.columns}
    required = ["nama_toko", "lat", "long"]
    if not all(k in cols for k in required):
        return False, (
            "Kolom file kamu belum sesuai.\n\n"
            "✅ Wajib ada 3 kolom ini (huruf besar/kecil bebas):\n"
            "- **nama_toko**\n- **lat**\n- **long**\n\n"
            "💡 Cara cepat: download template di aplikasi, lalu copy-paste data kamu ke situ."
        ), None, stats

    df2 = df[[cols["nama_toko"], cols["lat"], cols["long"]]].copy()
    df2.columns = ["nama_toko", "lat", "long"]

    df2["nama_toko"] = df2["nama_toko"].astype(str).str.strip()
    df2["lat"] = _to_float_series(df2["lat"])
    df2["long"] = _to_float_series(df2["long"])

    before = len(df2)
    df2 = df2.dropna(subset=["lat", "long"]).copy()

    # Range checks
    out_lat = (df2["lat"].abs() > 90).sum()
    out_lon = (df2["long"].abs() > 180).sum()
    if out_lat > 0 or out_lon > 0:
        return False, (
            "Ada nilai **lat/long** yang tidak valid (di luar range).\n\n"
            "✅ Pastikan:\n"
            "- lat harus antara **-90 sampai 90**\n"
            "- long harus antara **-180 sampai 180**\n\n"
            "💡 Contoh benar:\n"
            "- lat: -6.12\n"
            "- long: 106.88"
        ), None, stats

    after = len(df2)
    stats["rows_valid"] = int(after)
    stats["rows_dropped"] = int(before - after)

    if after == 0:
        return False, (
            "Semua baris lat/long kamu kosong atau tidak terbaca.\n\n"
            "✅ Pastikan kolom lat & long berisi angka.\n"
            "💡 Kalau pakai koma (106,88) tidak masalah—sistem sudah handle."
        ), None, stats

    # Stable row id per file upload (after cleaning)
    df2 = df2.reset_index(drop=True)
    df2["_row_id"] = df2.index.astype(int).astype(str)

    msg = (
        f"Data terbaca: **{stats['rows_valid']:,}** titik."
        + (f" (**{stats['rows_dropped']:,}** baris diabaikan karena lat/long kosong.)" if stats["rows_dropped"] else "")
    )
    return True, msg, df2, stats


# -----------------------------
# Balanced assignment (cap-aware)
# -----------------------------
def _balanced_assign(X: np.ndarray, centers: np.ndarray, cap: int) -> np.ndarray:
    """
    Assign each point to nearest center with capacity preference.
    If all centers full, assign to currently smallest-count center (cap can be exceeded).
    """
    n = X.shape[0]
    K = centers.shape[0]
    d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)  # (n,K)

    # Hard points first (bigger gap between best & second best)
    best2 = np.partition(d, 1, axis=1)[:, :2]
    hardness = (best2[:, 1] - best2[:, 0])
    order = np.argsort(hardness)[::-1]

    labels = np.full(n, -1, dtype=int)
    counts = np.zeros(K, dtype=int)

    for idx in order:
        pref = np.argsort(d[idx])  # centers sorted by distance
        placed = False
        for g in pref:
            if counts[g] < cap:
                labels[idx] = int(g)
                counts[g] += 1
                placed = True
                break
        if not placed:
            # all full: put into smallest group (exceed allowed)
            g = int(np.argmin(counts))
            labels[idx] = g
            counts[g] += 1

    return labels


# -----------------------------
# Stable sweep ordering (space-filling curve)
# -----------------------------
def _normalize01(a: np.ndarray) -> np.ndarray:
    amin = float(np.min(a))
    amax = float(np.max(a))
    return (a - amin) / (amax - amin + 1e-12)


def _morton_code_16bit(x01: np.ndarray, y01: np.ndarray) -> np.ndarray:
    """
    Morton / Z-order code: locality-preserving ordering.
    Great for "sweep" grouping that looks stable & non-lempar.
    """
    xi = (x01 * 65535).astype(np.uint32)
    yi = (y01 * 65535).astype(np.uint32)

    def part1by1(n: np.ndarray) -> np.ndarray:
        n = n & 0x0000FFFF
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n

    return (part1by1(xi) | (part1by1(yi) << 1)).astype(np.uint64)


def _apply_corner_direction(x01: np.ndarray, y01: np.ndarray, direction: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    direction determines which corner is "start".
    """
    direction = str(direction).strip()

    # NW_to_SE: start at northwest (lat high, long low)
    if direction == "NW_to_SE":
        return x01, 1.0 - y01
    # SW_to_NE: start at southwest (lat low, long low)
    if direction == "SW_to_NE":
        return x01, y01
    # NE_to_SW: start at northeast (lat high, long high)
    if direction == "NE_to_SW":
        return 1.0 - x01, 1.0 - y01
    # SE_to_NW: start at southeast (lat low, long high)
    if direction == "SE_to_NW":
        return 1.0 - x01, y01

    # default
    return x01, 1.0 - y01


# -----------------------------
# Grouping methods
# -----------------------------
@dataclass
class GroupMeta:
    K: int
    cap: int
    n_points: int
    cap_impossible: bool
    method: str
    direction: str = ""


def initial_grouping_kmeans_cap(df: pd.DataFrame, K: int, cap: int, seed: int = 42) -> Tuple[pd.DataFrame, GroupMeta]:
    """
    KMeans center + balanced assignment (cap-aware).
    Can produce "kelempar" when cap forces far moves (known behavior).
    """
    df = df.copy()
    X = df[["lat", "long"]].to_numpy(dtype=float)

    km = MiniBatchKMeans(n_clusters=int(K), random_state=int(seed), n_init=10)
    km.fit(X)
    centers = km.cluster_centers_

    labels = _balanced_assign(X, centers, cap=int(cap))
    df["_gidx"] = labels.astype(int)

    meta = GroupMeta(
        K=int(K),
        cap=int(cap),
        n_points=int(len(df)),
        cap_impossible=bool(len(df) > int(K) * int(cap)),
        method="kmeans_cap",
    )
    return df, meta


def initial_grouping_sweep(df: pd.DataFrame, K: int, cap: int, direction: str = "NW_to_SE") -> Tuple[pd.DataFrame, GroupMeta]:
    """
    Stable sweep grouping:
    - Make locality-preserving ordering (Morton / Z-order)
    - Chunk sequentially into groups by cap
    Result: groups tend to be contiguous and visually stable.
    """
    df = df.copy()

    x = df["long"].to_numpy(dtype=float)
    y = df["lat"].to_numpy(dtype=float)

    x01 = _normalize01(x)
    y01 = _normalize01(y)

    x_use, y_use = _apply_corner_direction(x01, y01, direction)
    key = _morton_code_16bit(x_use, y_use)

    df["_order_key"] = key
    df = df.sort_values("_order_key").reset_index(drop=True)

    n = len(df)
    labels = (np.arange(n) // max(1, int(cap))).astype(int)
    labels = np.minimum(labels, int(K) - 1)  # compress into max K groups

    df["_gidx"] = labels.astype(int)
    df = df.drop(columns=["_order_key"], errors="ignore")

    meta = GroupMeta(
        K=int(K),
        cap=int(cap),
        n_points=int(n),
        cap_impossible=bool(n > int(K) * int(cap)),
        method="sweep_morton",
        direction=direction,
    )
    return df, meta


# -----------------------------
# Refine (manual, lock-aware, cap-aware)
# -----------------------------
def refine_from_current(
    df: pd.DataFrame,
    K: int,
    cap: int,
    refine_iter: int,
    seed: int,
    override_map: Dict[str, int],
) -> pd.DataFrame:
    """
    Manual refine:
    - recompute centers from current labels
    - for each unlocked point: move to nearest center if target not full
    - locked points (overrides) never moved
    """
    df = df.copy()
    X = df[["lat", "long"]].to_numpy(dtype=float)
    labels = df["_gidx"].to_numpy(dtype=int)

    locked = np.zeros(len(df), dtype=bool)
    if override_map:
        id_to_idx: Dict[str, List[int]] = {}
        for i, rid in enumerate(df["_row_id"].astype(str).tolist()):
            id_to_idx.setdefault(rid, []).append(i)

        for rid, tgt in override_map.items():
            if rid in id_to_idx:
                for i in id_to_idx[rid]:
                    locked[i] = True
                    labels[i] = int(tgt)

    rng = np.random.default_rng(int(seed))

    for _ in range(int(refine_iter)):
        centers = np.zeros((int(K), 2), dtype=float)
        for g in range(int(K)):
            pts = X[labels == g]
            centers[g] = pts.mean(axis=0) if len(pts) else X.mean(axis=0)

        counts = np.bincount(labels, minlength=int(K)).astype(int)

        order = np.arange(len(df))
        rng.shuffle(order)

        for i in order:
            if locked[i]:
                continue

            cur = int(labels[i])
            d = np.linalg.norm(centers - X[i], axis=1)
            pref = np.argsort(d)

            for g in pref:
                g = int(g)
                if g == cur:
                    break
                if counts[g] >= int(cap):
                    continue
                counts[cur] -= 1
                counts[g] += 1
                labels[i] = g
                break

    df["_gidx"] = labels.astype(int)
    return df


# -----------------------------
# Apply overrides (anti rollback + delete support)
# -----------------------------
def apply_overrides_to_current(
    df_base: pd.DataFrame,
    df_current: pd.DataFrame,
    override_map: Dict[str, int],
    K: int,
    mode: str,
) -> pd.DataFrame:
    """
    mode:
      - 'current_only': apply overrides on current result (no rollback)
      - 'rebuild_from_base_then_apply': start from df_base then apply overrides
        (useful after deleting overrides so removed ones truly disappear)
    """
    df = df_base.copy() if mode == "rebuild_from_base_then_apply" else df_current.copy()

    if not override_map:
        return df

    for rid, tgt in override_map.items():
        tgt = int(tgt)
        if not (0 <= tgt < int(K)):
            continue
        m = df["_row_id"].astype(str) == str(rid)
        if m.any():
            df.loc[m, "_gidx"] = tgt

    return df


# -----------------------------
# Colors
# -----------------------------
def _palette_hex(K: int) -> List[str]:
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
        "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
        "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]
    if K <= len(base):
        return base[:K]

    import colorsys
    cols = base[:]
    for i in range(len(base), K):
        h = (i - len(base)) / max(1, (K - len(base)))
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        cols.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return cols


# -----------------------------
# Map (colors, boundary WAJIB, global search)
# -----------------------------
def build_map(df: pd.DataFrame, K: int) -> folium.Map:
    df = df.copy()

    center = [float(df["lat"].mean()), float(df["long"].mean())]
    m = folium.Map(location=center, zoom_start=12)

    palette = _palette_hex(int(K))

    fg_search = folium.FeatureGroup(name="(Search)", show=False)
    m.add_child(fg_search)

    for g in range(int(K)):
        color = palette[g % len(palette)]
        fg = folium.FeatureGroup(name=label_from_gidx(g), show=True)

        sub = df[df["_gidx"] == g]

        # points
        for _, r in sub.iterrows():
            lat = float(r["lat"])
            lon = float(r["long"])
            name = str(r["nama_toko"])

            popup_html = f"""
            <div style="font-size:13px">
              <b>{name}</b><br/>
              Grup: <b>{label_from_gidx(g)}</b><br/>
              Lat: {lat}<br/>Long: {lon}<br/>
              <a href="https://www.google.com/maps?q={lat},{lon}" target="_blank" rel="noopener noreferrer">
                🧭 Buka di Google Maps
              </a>
            </div>
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.85,
                tooltip=name,
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(fg)

            # invisible marker for global search
            folium.Marker(
                location=[lat, lon],
                tooltip=name,
                icon=folium.DivIcon(html=""),
            ).add_to(fg_search)

        # ===== Boundary / Hull WAJIB =====
        pts = list(zip(sub["long"].astype(float).tolist(), sub["lat"].astype(float).tolist()))

        if len(pts) >= 3:
            hull = MultiPoint(pts).convex_hull
            if hasattr(hull, "exterior"):
                coords = [(y, x) for x, y in hull.exterior.coords]
                folium.Polygon(
                    locations=coords,
                    color=color,
                    fill=True,
                    fill_opacity=0.15,
                    weight=2,
                ).add_to(fg)
            else:
                coords = [(y, x) for x, y in hull.coords]
                folium.PolyLine(coords, color=color, weight=4).add_to(fg)

        elif len(pts) == 2:
            coords = [(pts[0][1], pts[0][0]), (pts[1][1], pts[1][0])]
            folium.PolyLine(coords, color=color, weight=4, opacity=0.8).add_to(fg)

        elif len(pts) == 1:
            folium.Circle(
                location=(pts[0][1], pts[0][0]),
                radius=80,
                color=color,
                fill=True,
                fill_opacity=0.15,
                weight=2,
            ).add_to(fg)

        m.add_child(fg)

    try:
        Search(
            layer=fg_search,
            search_label="tooltip",
            placeholder="Cari nama toko...",
            collapsed=False,
        ).add_to(m)
    except Exception:
        pass

    folium.LayerControl(collapsed=False).add_to(m)
    return m
