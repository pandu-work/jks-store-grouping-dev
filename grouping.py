import numpy as np
import pandas as pd

import folium
from folium.plugins import Search

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from shapely.geometry import MultiPoint


# -----------------------------
# Labels
# -----------------------------
def label_from_gidx(g: int) -> str:
    return f"R{int(g) + 1:02d}"

def parse_label_to_gidx(label: str) -> int:
    return int(label.replace("R", "")) - 1


# -----------------------------
# Validation
# -----------------------------
def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def validate_input_df(df: pd.DataFrame):
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}

    required = ["nama_toko", "lat", "long"]
    if not all(k in cols for k in required):
        return False, "Kolom wajib: nama_toko, lat, long (case-insensitive).", None

    df2 = df[[cols["nama_toko"], cols["lat"], cols["long"]]].copy()
    df2.columns = ["nama_toko", "lat", "long"]

    df2["lat"] = _to_float_series(df2["lat"])
    df2["long"] = _to_float_series(df2["long"])
    df2 = df2.dropna(subset=["lat", "long"])

    if (df2["lat"].abs() > 90).any() or (df2["long"].abs() > 180).any():
        return False, "Lat/Long tidak valid (out of range).", None

    df2 = df2.reset_index(drop=True)
    df2["_row_id"] = df2.index.astype(int).astype(str)  # immutable per file
    return True, "", df2


# -----------------------------
# Balanced assignment (cap-aware)
# -----------------------------
def _balanced_assign(X: np.ndarray, centers: np.ndarray, cap: int) -> np.ndarray:
    """
    Assign each point to nearest center with capacity limit.
    If all centers full, assign to currently smallest-count center (allow exceed).
    """
    n = X.shape[0]
    K = centers.shape[0]
    d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)  # (n,K)

    # Hard points first (bigger gap between best & second best)
    order = np.argsort((np.partition(d, 1, axis=1)[:, 1] - np.partition(d, 1, axis=1)[:, 0]))[::-1]

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
# Initial grouping (NO auto refine)
# -----------------------------
def initial_grouping(df: pd.DataFrame, K: int, cap: int, seed: int = 42):
    df = df.copy()
    X = df[["lat", "long"]].to_numpy(dtype=float)

    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=10)
    km.fit(X)
    centers = km.cluster_centers_

    labels = _balanced_assign(X, centers, cap=cap)

    df["_gidx"] = labels.astype(int)

    meta = {
        "K": int(K),
        "cap": int(cap),
        "n_points": int(len(df)),
        "cap_impossible": bool(len(df) > K * cap),
    }
    return df, meta


# -----------------------------
# Refine (manual, lock-aware, cap-aware)
# -----------------------------
def refine_from_current(
    df: pd.DataFrame,
    K: int,
    cap: int,
    refine_iter: int,
    neighbor_k: int,
    seed: int,
    override_map: dict,
):
    df = df.copy()
    X = df[["lat", "long"]].to_numpy(dtype=float)
    labels = df["_gidx"].to_numpy(dtype=int)

    # lock points based on override_map
    locked = np.zeros(len(df), dtype=bool)
    if override_map:
        id_to_idx = {}
        for i, rid in enumerate(df["_row_id"].astype(str).tolist()):
            id_to_idx.setdefault(rid, []).append(i)

        for rid, tgt in override_map.items():
            if rid in id_to_idx:
                for i in id_to_idx[rid]:
                    locked[i] = True
                    labels[i] = int(tgt)

    for _ in range(int(refine_iter)):
        # recompute centers (include locked points to keep geometry stable)
        centers = np.zeros((K, 2), dtype=float)
        for g in range(K):
            pts = X[labels == g]
            if len(pts) == 0:
                centers[g] = X.mean(axis=0)
            else:
                centers[g] = pts.mean(axis=0)

        # cap counts baseline (locked already included)
        counts = np.bincount(labels, minlength=K).astype(int)

        # neighborhood info (optional smoothing)
        nn = NearestNeighbors(n_neighbors=min(int(neighbor_k), len(df))).fit(X)
        _, neigh = nn.kneighbors(X)

        # iterate points
        for i in range(len(df)):
            if locked[i]:
                continue

            # preferred group by nearest center
            d = np.linalg.norm(centers - X[i], axis=1)
            pref = np.argsort(d)

            # try move with cap check
            cur = int(labels[i])
            moved = False
            for g in pref:
                g = int(g)
                if g == cur:
                    moved = True
                    break
                if counts[g] >= cap:
                    continue
                # move
                counts[cur] -= 1
                counts[g] += 1
                labels[i] = g
                moved = True
                break

            if not moved:
                # if all full, keep current (cap respected)
                labels[i] = cur

    df["_gidx"] = labels.astype(int)
    return df


# -----------------------------
# Apply overrides to df_current (anti rollback + delete support)
# -----------------------------
def apply_overrides_to_current(df_base, df_current, override_map, K: int, cap: int, mode: str):
    """
    mode:
      - 'current_only': apply overrides on current result (no rollback, no rebuild)
      - 'rebuild_from_base_then_apply': rebuild baseline from df_base then apply override_map
        (useful after deleting overrides so removed ones truly disappear)
    """
    if mode == "rebuild_from_base_then_apply":
        df = df_base.copy()
    else:
        df = df_current.copy()

    if not override_map:
        return df

    # apply (override wins, cap can be exceeded)
    for rid, tgt in override_map.items():
        m = df["_row_id"].astype(str) == str(rid)
        if m.any() and 0 <= int(tgt) < int(K):
            df.loc[m, "_gidx"] = int(tgt)

    return df


# -----------------------------
# Map (colors, boundary WAJIB, global search)
# -----------------------------
def build_map(df: pd.DataFrame, K: int):
    df = df.copy()
    center = [df["lat"].mean(), df["long"].mean()]
    m = folium.Map(location=center, zoom_start=12)

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    ]

    # GLOBAL search index layer
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
            # hull could be Polygon or LineString in degenerate cases; handle safely
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
                # fallback line
                coords = [(y, x) for x, y in hull.coords]
                folium.PolyLine(coords, color=color, weight=4).add_to(fg)

        elif len(pts) == 2:
            # ✅ requested fallback: polyline for 2 points
            coords = [(pts[0][1], pts[0][0]), (pts[1][1], pts[1][0])]
            folium.PolyLine(coords, color=color, weight=4, opacity=0.8).add_to(fg)

        elif len(pts) == 1:
            # single point fallback: small circle
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
