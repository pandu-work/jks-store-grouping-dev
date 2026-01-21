# grouping.py
import math
import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import BallTree

import folium
from folium.plugins import Search


# =========================================================
# Helpers
# =========================================================
def _to_float_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s2, errors="coerce")


def label_from_gidx(gidx: int) -> str:
    return f"R{int(gidx) + 1:02d}"


def parse_label_to_gidx(label: str) -> int:
    if label is None:
        raise ValueError("Label kosong.")
    t = str(label).strip().upper()
    if t.startswith("R"):
        t = t[1:]
    if not t.isdigit():
        raise ValueError(f"Label group tidak valid: {label}")
    return int(t) - 1


def _palette(n: int):
    # set warna stabil (repeat-safe)
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
        "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363",
    ]
    if n <= len(base):
        return base[:n]
    out = []
    for i in range(n):
        out.append(base[i % len(base)])
    return out


# =========================================================
# Input validation
# =========================================================
def validate_input_df(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return False, "File kosong.", None

    cols = {str(c).strip().lower(): c for c in df.columns}

    def pick(cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    col_name = pick(["nama_toko", "nama toko", "toko", "store", "name", "nama"])
    col_lat = pick(["lat", "latitude", "y"])
    col_lon = pick(["long", "lon", "lng", "longitude", "x"])

    if col_name is None or col_lat is None or col_lon is None:
        return (
            False,
            "Kolom wajib tidak lengkap. Minimal harus ada: nama_toko | lat | long.\n"
            f"Kolom ditemukan: {list(df.columns)}",
            None,
        )

    df2 = df.copy()
    df2 = df2.rename(columns={col_name: "nama_toko", col_lat: "lat", col_lon: "long"})

    df2["nama_toko"] = df2["nama_toko"].astype(str).str.strip()
    df2["lat"] = _to_float_series(df2["lat"])
    df2["long"] = _to_float_series(df2["long"])

    df2 = df2.dropna(subset=["lat", "long"])
    df2 = df2[df2["nama_toko"].str.len() > 0]

    if len(df2) == 0:
        return False, "Semua baris invalid setelah parsing. Cek lat/long & nama_toko.", None

    # range check Indonesia-ish (optional)
    if (df2["lat"].abs() > 90).any() or (df2["long"].abs() > 180).any():
        return False, "Ada lat/long di luar range valid (lat ±90, long ±180).", None

    df2 = df2.reset_index(drop=True)
    return True, "OK", df2


# =========================================================
# Geometry: convex hull (monotonic chain)
# =========================================================
def _convex_hull(points_xy: np.ndarray) -> np.ndarray:
    # points_xy: (n,2) [lon,lat]
    pts = np.unique(points_xy, axis=0)
    if len(pts) < 3:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull


# =========================================================
# Core grouping
# =========================================================
def _initial_centroids_kmeans(coords_latlon: np.ndarray, K: int, seed: int = 42):
    # KMeans in lat/lon space (OK for seeding)
    km = MiniBatchKMeans(
        n_clusters=K,
        random_state=seed,
        n_init="auto",
        batch_size=min(4096, max(256, len(coords_latlon))),
        max_iter=200,
    )
    km.fit(coords_latlon)
    return km.cluster_centers_


def _balanced_assign(coords: np.ndarray, centroids: np.ndarray, K: int, cap: int):
    """
    Greedy balanced assignment:
    - compute distance to each centroid
    - assign each point to nearest available centroid (capacity cap)
    """
    n = coords.shape[0]
    # distances (n,K)
    d = np.sqrt(((coords[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
    order = np.argsort(d, axis=1)

    counts = np.zeros(K, dtype=int)
    labels = np.full(n, -1, dtype=int)

    # assign in order of "certainty" (smaller nearest distance first)
    nearest_dist = d[np.arange(n), order[:, 0]]
    idxs = np.argsort(nearest_dist)

    for i in idxs:
        for g in order[i]:
            if counts[g] < cap:
                labels[i] = g
                counts[g] += 1
                break

    # fallback: if still -1 (cap too small), force assign to least loaded
    if (labels < 0).any():
        for i in np.where(labels < 0)[0]:
            g = int(np.argmin(counts))
            labels[i] = g
            counts[g] += 1

    return labels


def refine_from_current(
    dfw: pd.DataFrame,
    K: int,
    cap: int,
    refine_iter: int = 2,
    neighbor_k: int = 12,
    seed: int = 42,
    override_map: dict | None = None,
):
    """
    Local refinement:
    - each point tries to move to a neighbor's group if closer (and capacity allows)
    - respects override_map (locked)
    """
    rng = np.random.default_rng(seed)
    override_map = override_map or {}

    coords = dfw[["lat", "long"]].to_numpy(dtype=float)
    tree = BallTree(np.radians(coords), metric="haversine")  # stable for neighbor query
    # but for comparison, we still compute euclid in lat/lon for simplicity
    coords_ll = coords

    labels = dfw["_gidx"].to_numpy(dtype=int)
    n = len(labels)

    locked = np.zeros(n, dtype=bool)
    # lock by unique row id if exists, else by nama_toko
    if "_row_id" in dfw.columns:
        key_series = dfw["_row_id"].astype(str)
    else:
        key_series = dfw["nama_toko"].astype(str)

    key_to_idx = {}
    for i, k in enumerate(key_series):
        key_to_idx.setdefault(k, []).append(i)

    for k, target in override_map.items():
        if k in key_to_idx:
            for i in key_to_idx[k]:
                locked[i] = True
                labels[i] = int(target)

    for _ in range(int(refine_iter)):
        # recompute centroids
        centroids = np.zeros((K, 2), dtype=float)
        counts = np.zeros(K, dtype=int)
        for g in range(K):
            mask = labels == g
            counts[g] = int(mask.sum())
            if counts[g] > 0:
                centroids[g] = coords_ll[mask].mean(axis=0)
            else:
                # empty cluster -> random point
                centroids[g] = coords_ll[int(rng.integers(0, n))]

        # query neighbors
        # haversine query returns indices; we use them to propose group moves
        _, ind = tree.query(np.radians(coords), k=min(neighbor_k, n))

        # iterate points in random order (avoid bias)
        order = np.arange(n)
        rng.shuffle(order)

        for i in order:
            if locked[i]:
                continue

            gi = labels[i]
            # if current group already over cap, allow moving out
            # but if under cap, still allow move if improves compactness
            best_g = gi
            best_dist = float(np.linalg.norm(coords_ll[i] - centroids[gi]))

            # propose groups from neighbors
            neighbor_groups = labels[ind[i]]
            for g in np.unique(neighbor_groups):
                g = int(g)
                if g == gi:
                    continue
                if counts[g] >= cap:
                    continue
                dist_g = float(np.linalg.norm(coords_ll[i] - centroids[g]))
                if dist_g < best_dist:
                    best_dist = dist_g
                    best_g = g

            if best_g != gi:
                # apply move
                labels[i] = best_g
                counts[gi] -= 1
                counts[best_g] += 1

    dfw = dfw.copy()
    dfw["_gidx"] = labels.astype(int)
    return dfw


def apply_overrides(dfw: pd.DataFrame, override_map: dict, K: int, cap: int):
    """
    Apply forced group assignments.
    Does NOT run refine. Only replaces labels and keeps capacity best-effort.
    """
    if not override_map:
        return dfw, 0, 0

    dfw2 = dfw.copy()
    applied = 0
    skipped = 0

    # counting current load
    counts = dfw2["_gidx"].value_counts().to_dict()

    # helper get indices
    if "_row_id" in dfw2.columns:
        key_series = dfw2["_row_id"].astype(str)
    else:
        key_series = dfw2["nama_toko"].astype(str)

    key_to_idx = {}
    for i, k in enumerate(key_series):
        key_to_idx.setdefault(k, []).append(i)

    for key, target in override_map.items():
        if key not in key_to_idx:
            skipped += 1
            continue
        g = int(target)
        if g < 0 or g >= K:
            skipped += 1
            continue

        for i in key_to_idx[key]:
            old = int(dfw2.at[i, "_gidx"])
            if old == g:
                continue

            # best-effort cap: allow exceeding cap if needed (because user forced)
            counts[old] = int(counts.get(old, 0)) - 1
            counts[g] = int(counts.get(g, 0)) + 1
            dfw2.at[i, "_gidx"] = g
            applied += 1

    return dfw2, applied, skipped


# =========================================================
# Map building
# =========================================================
def build_map(dfw: pd.DataFrame, K: int):
    if len(dfw) == 0:
        return folium.Map(location=[-6.2, 106.8], zoom_start=11)

    center = [dfw["lat"].mean(), dfw["long"].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

    colors = _palette(K)

    # feature groups
    fg_hull = folium.FeatureGroup(name="Batas Group (Hull)", show=True)
    m.add_child(fg_hull)

    fg_points = []
    for g in range(K):
        fg = folium.FeatureGroup(name=f"{label_from_gidx(g)} (toko)", show=True)
        fg_points.append(fg)
        m.add_child(fg)

    # add hull polygons
    for g in range(K):
        sub = dfw[dfw["_gidx"] == g]
        if len(sub) < 3:
            continue
        pts = sub[["long", "lat"]].to_numpy(dtype=float)
        hull = _convex_hull(pts)
        if len(hull) < 3:
            continue
        poly_latlon = [[float(y), float(x)] for x, y in hull]  # folium uses [lat,lon]
        folium.Polygon(
            locations=poly_latlon,
            color=colors[g],
            weight=3,
            fill=True,
            fill_opacity=0.10,
            tooltip=f"{label_from_gidx(g)} hull",
        ).add_to(fg_hull)

    # add markers with gmaps button
    for idx, r in dfw.iterrows():
        g = int(r["_gidx"])
        lat = float(r["lat"])
        lon = float(r["long"])
        name = str(r["nama_toko"])

        gmaps = f"https://www.google.com/maps?q={lat},{lon}"
        html = f"""
        <div style="font-size:13px">
          <b>{name}</b><br/>
          Group: <b>{label_from_gidx(g)}</b><br/>
          Lat: {lat}<br/>
          Long: {lon}<br/>
          <a href="{gmaps}" target="_blank" rel="noopener noreferrer">
            🧭 Go to Google Maps
          </a>
        </div>
        """

        marker = folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=colors[g],
            fill=True,
            fill_opacity=0.85,
            popup=folium.Popup(html, max_width=320),
            tooltip=f"{name} ({label_from_gidx(g)})",
        )
        marker.add_to(fg_points[g])

    # Search by tooltip text (works decently)
    try:
        Search(
            layer=fg_points[0],
            search_label="tooltip",
            placeholder="Cari nama toko ...",
            collapsed=False,
        ).add_to(m)
    except Exception:
        pass

    folium.LayerControl(collapsed=True).add_to(m)
    return m


# =========================================================
# Pipeline entry
# =========================================================
def process_excel(
    df_clean: pd.DataFrame,
    K: int,
    hard_cap: int,
    seed: int = 42,
    refine_on: bool = True,
    refine_iter: int = 2,
    neighbor_k: int = 12,
    override_map: dict | None = None,
):
    """
    Returns: (df_result, folium_map)
    df_result includes: nama_toko, lat, long, _gidx, kategori
    """
    dfw = df_clean.copy()

    # stable row id (avoid duplicate name confusion). If you already have unique id, replace this.
    if "_row_id" not in dfw.columns:
        dfw["_row_id"] = np.arange(len(dfw)).astype(int).astype(str)

    coords = dfw[["lat", "long"]].to_numpy(dtype=float)

    # seed centroids
    centroids = _initial_centroids_kmeans(coords, K=K, seed=seed)

    # balanced assignment
    labels = _balanced_assign(coords, centroids, K=K, cap=int(hard_cap))
    dfw["_gidx"] = labels.astype(int)

    # apply overrides (initial)
    override_map = override_map or {}
    dfw, applied, skipped = apply_overrides(dfw, override_map, K=K, cap=int(hard_cap))

    # refine (optional)
    if refine_on and int(refine_iter) > 0:
        dfw = refine_from_current(
            dfw,
            K=K,
            cap=int(hard_cap),
            refine_iter=int(refine_iter),
            neighbor_k=int(neighbor_k),
            seed=seed,
            override_map=override_map,
        )

    dfw["kategori"] = dfw["_gidx"].apply(label_from_gidx)

    m = build_map(dfw, K=K)
    return dfw, m, applied, skipped


# =========================================================
# Compatibility wrappers (biar app.py bebas mau panggil yang mana)
# =========================================================
def initial_grouping(df_clean: pd.DataFrame, K: int, cap: int = None, hard_cap: int = None, **kwargs):
    """
    Accept both cap or hard_cap to avoid TypeError.
    """
    if hard_cap is None and cap is None:
        raise TypeError("initial_grouping butuh cap atau hard_cap.")
    hc = int(hard_cap if hard_cap is not None else cap)
    refine_on = bool(kwargs.get("refine_on", True))
    refine_iter = int(kwargs.get("refine_iter", 2))
    neighbor_k = int(kwargs.get("neighbor_k", 12))
    seed = int(kwargs.get("seed", 42))

    dfw, m, applied, skipped = process_excel(
        df_clean=df_clean,
        K=int(K),
        hard_cap=hc,
        seed=seed,
        refine_on=refine_on,
        refine_iter=refine_iter,
        neighbor_k=neighbor_k,
        override_map=kwargs.get("override_map", None),
    )
    return dfw, m, {"applied": applied, "skipped": skipped}


def run_grouping_pipeline(df_clean: pd.DataFrame, K: int, cap: int = None, hard_cap: int = None, **kwargs):
    return initial_grouping(df_clean=df_clean, K=K, cap=cap, hard_cap=hard_cap, **kwargs)
