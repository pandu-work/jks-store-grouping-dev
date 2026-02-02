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
    # Accept "R01", "R1" etc
    label = str(label).strip().upper()
    if not label.startswith("R"):
        raise ValueError("Label group harus format 'Rxx', contoh: R01.")
    return int(label.replace("R", "")) - 1


# -----------------------------
# Validation helpers
# -----------------------------
def _to_float_series(s: pd.Series) -> pd.Series:
    # Accept comma decimal "106,88" -> "106.88"
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def validate_input_df(df: pd.DataFrame) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Required columns (case-insensitive): nama_toko, lat, long
    Output: df_clean with columns: nama_toko, lat, long, _row_id
    """
    if df is None or df.empty:
        return False, "File kosong / tidak ada data.", None

    df = df.copy()
    cols = {c.lower().strip(): c for c in df.columns}

    required = ["nama_toko", "lat", "long"]
    if not all(k in cols for k in required):
        return False, "Kolom wajib: nama_toko, lat, long (case-insensitive).", None

    df2 = df[[cols["nama_toko"], cols["lat"], cols["long"]]].copy()
    df2.columns = ["nama_toko", "lat", "long"]

    df2["nama_toko"] = df2["nama_toko"].astype(str).str.strip()
    df2["lat"] = _to_float_series(df2["lat"])
    df2["long"] = _to_float_series(df2["long"])

    # drop missing coordinates
    before = len(df2)
    df2 = df2.dropna(subset=["lat", "long"]).copy()
    after = len(df2)

    if after == 0:
        return False, "Semua baris lat/long kosong atau tidak valid.", None

    # range check
    if (df2["lat"].abs() > 90).any() or (df2["long"].abs() > 180).any():
        return False, "Lat/Long tidak valid (out of range). Pastikan lat [-90..90], long [-180..180].", None

    # create stable row id per uploaded file (by row order after cleaning)
    df2 = df2.reset_index(drop=True)
    df2["_row_id"] = df2.index.astype(int).astype(str)

    dropped = before - after
    msg = ""
    if dropped > 0:
        msg = f"Info: {dropped:,} baris di-drop karena lat/long tidak valid."

    return True, msg, df2


# -----------------------------
# Balanced assignment (cap-aware)
# -----------------------------
def _balanced_assign(X: np.ndarray, centers: np.ndarray, cap: int) -> np.ndarray:
    """
    Assign each point to nearest center with a capacity limit (soft).
    - Try fill nearest center while count < cap
    - If all centers full, assign to smallest-count center (cap can be exceeded)
    """
    n = X.shape[0]
    K = centers.shape[0]

    # distance matrix (n, K)
    d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)

    # hard points first: larger gap between best & second best
    # safer than random ordering
    best2 = np.partition(d, 1, axis=1)[:, :2]
    hardness = (best2[:, 1] - best2[:, 0])
    order = np.argsort(hardness)[::-1]

    labels = np.full(n, -1, dtype=int)
    counts = np.zeros(K, dtype=int)

    for idx in order:
        pref = np.argsort(d[idx])  # nearest center first
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
# Initial grouping (NO auto refine)
# -----------------------------
@dataclass
class GroupMeta:
    K: int
    cap: int
    n_points: int
    cap_impossible: bool


def initial_grouping(df: pd.DataFrame, K: int, cap: int, seed: int = 42) -> Tuple[pd.DataFrame, GroupMeta]:
    """
    - Fit MiniBatchKMeans to get centers
    - Balanced greedy assignment with cap preference
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
    - Recompute centers from current labels
    - For each unlocked point: move to nearest center if target not full
    - Locked points (overrides) are never moved
    """
    df = df.copy()
    X = df[["lat", "long"]].to_numpy(dtype=float)
    labels = df["_gidx"].to_numpy(dtype=int)

    # lock points based on override_map
    locked = np.zeros(len(df), dtype=bool)
    if override_map:
        # _row_id can be duplicated? In our cleaning it's unique, but keep safe.
        id_to_idxs: Dict[str, List[int]] = {}
        for i, rid in enumerate(df["_row_id"].astype(str).tolist()):
            id_to_idxs.setdefault(rid, []).append(i)

        for rid, tgt in override_map.items():
            if rid in id_to_idxs:
                for i in id_to_idxs[rid]:
                    locked[i] = True
                    labels[i] = int(tgt)

    rng = np.random.default_rng(int(seed))

    for _ in range(int(refine_iter)):
        # recompute centers
        centers = np.zeros((int(K), 2), dtype=float)
        for g in range(int(K)):
            pts = X[labels == g]
            centers[g] = pts.mean(axis=0) if len(pts) else X.mean(axis=0)

        counts = np.bincount(labels, minlength=int(K)).astype(int)

        # optional shuffle for tie-breaking stability (still deterministic via seed)
        order = np.arange(len(df))
        rng.shuffle(order)

        for i in order:
            if locked[i]:
                continue

            cur = int(labels[i])
            d = np.linalg.norm(centers - X[i], axis=1)
            pref = np.argsort(d)

            moved = False
            for g in pref:
                g = int(g)
                if g == cur:
                    moved = True
                    break
                if counts[g] >= int(cap):
                    continue
                # move
                counts[cur] -= 1
                counts[g] += 1
                labels[i] = g
                moved = True
                break

            if not moved:
                labels[i] = cur

    df["_gidx"] = labels.astype(int)
    return df


# -----------------------------
# Apply overrides to current (anti rollback + delete support)
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
    if mode == "rebuild_from_base_then_apply":
        df = df_base.copy()
    else:
        df = df_current.copy()

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
    """
    A nicer palette that supports K up to 50 without confusing repeats.
    Uses a fixed set + HSV fallback.
    """
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
        "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
        "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]
    if K <= len(base):
        return base[:K]

    # HSV fallback (deterministic)
    cols = base[:]
    for i in range(len(base), K):
        h = (i - len(base)) / max(1, (K - len(base)))
        # simple hsv -> rgb
        import colorsys
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

    # GLOBAL search index layer (invisible markers)
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
              Group: <b>{label_from_gidx(g)}</b><br/>
              Lat: {lat}<br/>Long: {lon}<br/>
              <a href="https://www.google.com/maps?q={lat},{lon}" target="_blank" rel="noopener noreferrer">
                🧭 Go to Google Maps
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

    # Search global
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
