import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
import folium
from folium.plugins import Search
from shapely.geometry import MultiPoint

# =============================
# LABEL
# =============================
def label_from_gidx(g):
    return f"R{int(g)+1:02d}"

def parse_label_to_gidx(label):
    return int(label.replace("R", "")) - 1

# =============================
# VALIDATION
# =============================
def _to_float_series(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def validate_input_df(df):
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}

    try:
        df2 = df[[cols["nama_toko"], cols["lat"], cols["long"]]].copy()
        df2.columns = ["nama_toko", "lat", "long"]
    except Exception:
        return False, "Kolom wajib: nama_toko, lat, long", None

    df2["lat"] = _to_float_series(df2["lat"])
    df2["long"] = _to_float_series(df2["long"])
    df2 = df2.dropna(subset=["lat", "long"])

    if (df2["lat"].abs() > 90).any() or (df2["long"].abs() > 180).any():
        return False, "Lat/Long tidak valid", None

    df2 = df2.reset_index(drop=True)
    df2["_row_id"] = df2.index.astype(str)
    return True, "", df2

# =============================
# INITIAL GROUPING
# =============================
def initial_grouping(df, K, cap, seed=42):
    X = df[["lat", "long"]].values
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    df["_gidx"] = labels
    return df, None, None

# =============================
# REFINE
# =============================
def refine_from_current(df, K, cap, refine_iter, neighbor_k, seed, override_map):
    df = df.copy()
    X = df[["lat", "long"]].values
    labels = df["_gidx"].values

    locked = np.zeros(len(df), dtype=bool)
    for rid, tgt in override_map.items():
        locked[df["_row_id"] == rid] = True
        labels[df["_row_id"] == rid] = int(tgt)

    for _ in range(refine_iter):
        centers = np.vstack([X[labels == i].mean(axis=0) for i in range(K)])
        nn = NearestNeighbors(n_neighbors=neighbor_k).fit(X)
        _, neigh = nn.kneighbors(X)

        for i in range(len(df)):
            if locked[i]:
                continue
            dists = np.linalg.norm(centers - X[i], axis=1)
            labels[i] = int(np.argmin(dists))

    df["_gidx"] = labels
    return df

# =============================
# OVERRIDE
# =============================
def apply_overrides(df, override_map, K, cap):
    df = df.copy()
    applied = skipped = 0
    for rid, tgt in override_map.items():
        m = df["_row_id"] == rid
        if not m.any() or tgt < 0 or tgt >= K:
            skipped += 1
            continue
        df.loc[m, "_gidx"] = int(tgt)
        applied += 1
    return df, applied, skipped

# =============================
# MAP
# =============================
def build_map(df, K):
    center = [df["lat"].mean(), df["long"].mean()]
    m = folium.Map(location=center, zoom_start=12)

    palette = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728",
        "#9467bd","#8c564b","#e377c2","#7f7f7f",
        "#bcbd22","#17becf","#aec7e8","#ffbb78",
    ]

    fg_search = folium.FeatureGroup(name="search", show=False)
    m.add_child(fg_search)

    for g in range(K):
        sub = df[df["_gidx"] == g]
        color = palette[g % len(palette)]

        fg = folium.FeatureGroup(name=label_from_gidx(g))
        for _, r in sub.iterrows():
            folium.CircleMarker(
                [r["lat"], r["long"]],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.8,
                tooltip=r["nama_toko"],
            ).add_to(fg)

            folium.Marker(
                [r["lat"], r["long"]],
                tooltip=r["nama_toko"],
                icon=folium.DivIcon(html=""),
            ).add_to(fg_search)

        # ===== Boundary (WAJIB) =====
        pts = list(zip(sub["long"], sub["lat"]))
        if len(pts) >= 3:
            hull = MultiPoint(pts).convex_hull
            folium.Polygon(
                locations=[(y, x) for x, y in hull.exterior.coords],
                color=color,
                fill=True,
                fill_opacity=0.15,
            ).add_to(fg)
        elif len(pts) == 2:
            folium.PolyLine(
                locations=[(p[1], p[0]) for p in pts],
                color=color,
                weight=4,
            ).add_to(fg)
        elif len(pts) == 1:
            folium.Circle(
                location=(pts[0][1], pts[0][0]),
                radius=80,
                color=color,
                fill=True,
                fill_opacity=0.15,
            ).add_to(fg)

        m.add_child(fg)

    Search(layer=fg_search, search_label="tooltip").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m
