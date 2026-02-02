# grouping.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

import folium
from folium.plugins import Search

from shapely.geometry import MultiPoint
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

import heapq


# =========================
# Labels
# =========================
def label_from_gidx(g: int) -> str:
    return f"R{int(g) + 1:02d}"


def parse_label_to_gidx(label: str) -> int:
    label = str(label).strip().upper()
    if not label.startswith("R"):
        raise ValueError("Format grup harus seperti R01, R02, dst.")
    return int(label.replace("R", "")) - 1


# =========================
# Validation (user-friendly)
# =========================
def _to_float_series(s: pd.Series) -> pd.Series:
    # Accept comma decimals: "106,88" -> "106.88"
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def validate_input_df(df: pd.DataFrame) -> Tuple[bool, str, Optional[pd.DataFrame], Dict[str, int]]:
    """
    Required columns (case-insensitive): nama_toko, lat, long
    Returns: ok, message, df_clean, stats
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

    out_lat = int((df2["lat"].abs() > 90).sum())
    out_lon = int((df2["long"].abs() > 180).sum())
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

    df2 = df2.reset_index(drop=True)
    df2["_row_id"] = df2.index.astype(int).astype(str)

    msg = (
        f"Data terbaca: **{stats['rows_valid']:,}** titik."
        + (f" (**{stats['rows_dropped']:,}** baris diabaikan karena lat/long kosong.)" if stats["rows_dropped"] else "")
    )
    return True, msg, df2, stats


# =========================
# Meta
# =========================
@dataclass
class GroupMeta:
    K: int
    cap: int
    n_points: int
    cap_impossible: bool
    method: str
    target_min: int
    target_max: int


def _desired_sizes(n: int, K: int) -> List[int]:
    base = n // K
    rem = n % K
    return [base + (1 if g < rem else 0) for g in range(K)]


# =========================
# Method A (FINAL DEFAULT): Region Growing (anti campur)
# =========================
def initial_grouping_region_grow(
    df: pd.DataFrame,
    K: int,
    cap: int,
    seed: int = 42,
    k_neighbors: int = 12,
) -> Tuple[pd.DataFrame, GroupMeta]:
    """
    Territory / contiguity clustering (anti-campur):
    - Balanced sizes first (±1)
    - Cap as hard max (best-effort if impossible)
    - Each group grows via nearest neighbors (kNN graph) => contiguous “wilayah”
    """
    df = df.copy()
    X = df[["lat", "long"]].to_numpy(dtype=float)
    n = len(df)
    K = int(K)
    cap = int(cap)

    desired = np.array(_desired_sizes(n, K), dtype=int)
    target_min = int(desired.min())
    target_max = int(desired.max())

    cap_impossible = bool(n > K * cap) or bool(np.any(desired > cap))

    # hard limits per group during growing
    limits = np.minimum(desired, cap)

    # Build kNN graph
    k_neighbors = int(max(3, min(k_neighbors, n - 1)))
    nn = NearestNeighbors(n_neighbors=k_neighbors, algorithm="ball_tree")
    nn.fit(X)
    dists, neigh = nn.kneighbors(X)

    # Choose seeds using farthest-point sampling (spread out)
    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(0, n))
    seeds = [first]
    dist_to_seeds = np.linalg.norm(X - X[first], axis=1)

    for _ in range(1, K):
        nxt = int(np.argmax(dist_to_seeds))
        seeds.append(nxt)
        dist_to_seeds = np.minimum(dist_to_seeds, np.linalg.norm(X - X[nxt], axis=1))

    # Init assignments
    labels = np.full(n, -1, dtype=int)
    counts = np.zeros(K, dtype=int)

    for g, si in enumerate(seeds):
        labels[si] = g
        counts[g] += 1

    # Frontier priority queue: (edge_dist, node, group)
    pq = []
    for g, si in enumerate(seeds):
        for j, nb in enumerate(neigh[si]):
            if nb == si:
                continue
            heapq.heappush(pq, (float(dists[si][j]), int(nb), int(g)))

    # Region grow
    while pq:
        cost, i, g = heapq.heappop(pq)

        if labels[i] != -1:
            continue
        if counts[g] >= limits[g]:
            continue

        labels[i] = g
        counts[g] += 1

        for j, nb in enumerate(neigh[i]):
            if labels[nb] == -1:
                heapq.heappush(pq, (float(dists[i][j]), int(nb), int(g)))

        if np.all(counts >= limits):
            break

    # Assign remaining points (best-effort): nearest group with room (cap), else smallest group
    if (labels == -1).any():
        centers = np.zeros((K, 2), dtype=float)
        for g in range(K):
            pts = X[labels == g]
            centers[g] = pts.mean(axis=0) if len(pts) else X.mean(axis=0)

        for i in np.where(labels == -1)[0]:
            d = np.linalg.norm(centers - X[i], axis=1)
            pref = np.argsort(d)
            placed = False
            for g in pref:
                g = int(g)
                if counts[g] < cap:
                    labels[i] = g
                    counts[g] += 1
                    placed = True
                    break
            if not placed:
                g = int(np.argmin(counts))
                labels[i] = g
                counts[g] += 1

    df["_gidx"] = labels.astype(int)

    meta = GroupMeta(
        K=K,
        cap=cap,
        n_points=n,
        cap_impossible=cap_impossible,
        method="region_grow",
        target_min=target_min,
        target_max=target_max,
    )
    return df, meta


# =========================
# Method B (optional): Balanced Compact (distance + balanced)
# =========================
def _balanced_assign_with_limits(X: np.ndarray, centers: np.ndarray, limits: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    K = centers.shape[0]
    d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)

    best2 = np.partition(d, 1, axis=1)[:, :2]
    hardness = (best2[:, 1] - best2[:, 0])
    order = np.argsort(hardness)[::-1]

    labels = np.full(n, -1, dtype=int)
    counts = np.zeros(K, dtype=int)

    for i in order:
        pref = np.argsort(d[i])
        for g in pref:
            g = int(g)
            if counts[g] < limits[g]:
                labels[i] = g
                counts[g] += 1
                break
        if labels[i] == -1:
            g = int(np.argmin(counts))
            labels[i] = g
            counts[g] += 1

    return labels


def initial_grouping_balanced_compact(
    df: pd.DataFrame,
    K: int,
    cap: int,
    seed: int = 42,
    refine_iter: int = 12,
) -> Tuple[pd.DataFrame, GroupMeta]:
    """
    Balanced size first, then optimize closeness.
    WARNING: can still mix in dense center (no contiguity constraint).
    """
    df = df.copy()
    X = df[["lat", "long"]].to_numpy(dtype=float)
    n = len(df)
    K = int(K)
    cap = int(cap)

    desired = np.array(_desired_sizes(n, K), dtype=int)
    target_min = int(desired.min())
    target_max = int(desired.max())

    cap_impossible = bool(n > K * cap) or bool(np.any(desired > cap))
    limits = desired.copy()
    if cap_impossible:
        limits = np.minimum(desired, cap)

    km = MiniBatchKMeans(n_clusters=K, random_state=int(seed), n_init=10)
    km.fit(X)
    centers = km.cluster_centers_

    if (not cap_impossible) and int(limits.sum()) == n:
        labels = _balanced_assign_with_limits(X, centers, limits)
    else:
        labels = km.predict(X).astype(int)

    rng = np.random.default_rng(int(seed))

    def recompute_centers(labels_: np.ndarray) -> np.ndarray:
        c = np.zeros((K, 2), dtype=float)
        for g in range(K):
            pts = X[labels_ == g]
            c[g] = pts.mean(axis=0) if len(pts) else X.mean(axis=0)
        return c

    def counts_of(labels_: np.ndarray) -> np.ndarray:
        return np.bincount(labels_, minlength=K).astype(int)

    def repair(labels_: np.ndarray) -> np.ndarray:
        cnt = counts_of(labels_)
        c = recompute_centers(labels_)
        for g in range(K):
            while cnt[g] > cap:
                idxs = np.where(labels_ == g)[0]
                d_g = np.linalg.norm(X[idxs] - c[g], axis=1)
                worst_i = int(idxs[int(np.argmax(d_g))])

                d_all = np.linalg.norm(c - X[worst_i], axis=1)
                pref = np.argsort(d_all)
                moved = False
                for h in pref:
                    h = int(h)
                    if h == g:
                        continue
                    if cnt[h] < cap:
                        labels_[worst_i] = h
                        cnt[g] -= 1
                        cnt[h] += 1
                        moved = True
                        break
                if not moved:
                    break
        return labels_

    labels = repair(labels)

    min_size = target_min
    max_size = min(target_max, cap) if not cap_impossible else cap

    for _ in range(int(refine_iter)):
        centers = recompute_centers(labels)
        cnt = counts_of(labels)

        order = np.arange(n)
        rng.shuffle(order)

        improved = 0
        for i in order:
            cur = int(labels[i])
            d = np.linalg.norm(centers - X[i], axis=1)
            pref = np.argsort(d)

            best_gain = 0.0
            best_tgt = cur

            for tgt in pref:
                tgt = int(tgt)
                if tgt == cur:
                    break
                if cnt[tgt] >= cap:
                    continue

                if not cap_impossible:
                    if cnt[tgt] >= max_size:
                        continue
                    if cnt[cur] <= min_size:
                        continue

                gain = float(np.linalg.norm(X[i] - centers[cur]) - np.linalg.norm(X[i] - centers[tgt]))
                if gain > best_gain:
                    best_gain = gain
                    best_tgt = tgt

            if best_tgt != cur:
                cnt[cur] -= 1
                cnt[best_tgt] += 1
                labels[i] = best_tgt
                improved += 1

        labels = repair(labels)
        if improved == 0:
            break

    df["_gidx"] = labels.astype(int)
    meta = GroupMeta(
        K=K,
        cap=cap,
        n_points=n,
        cap_impossible=cap_impossible,
        method="balanced_compact",
        target_min=target_min,
        target_max=target_max,
    )
    return df, meta


# =========================
# Method C (optional): KMeans + Cap (legacy)
# =========================
def _balanced_assign_cap(X: np.ndarray, centers: np.ndarray, cap: int) -> np.ndarray:
    n = X.shape[0]
    K = centers.shape[0]
    d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)

    best2 = np.partition(d, 1, axis=1)[:, :2]
    hardness = (best2[:, 1] - best2[:, 0])
    order = np.argsort(hardness)[::-1]

    labels = np.full(n, -1, dtype=int)
    counts = np.zeros(K, dtype=int)

    for idx in order:
        pref = np.argsort(d[idx])
        placed = False
        for g in pref:
            g = int(g)
            if counts[g] < cap:
                labels[idx] = g
                counts[g] += 1
                placed = True
                break
        if not placed:
            g = int(np.argmin(counts))
            labels[idx] = g
            counts[g] += 1

    return labels


def initial_grouping_kmeans_cap(df: pd.DataFrame, K: int, cap: int, seed: int = 42) -> Tuple[pd.DataFrame, GroupMeta]:
    df = df.copy()
    X = df[["lat", "long"]].to_numpy(dtype=float)
    n = len(df)
    K = int(K)
    cap = int(cap)

    desired = np.array(_desired_sizes(n, K), dtype=int)
    target_min = int(desired.min())
    target_max = int(desired.max())

    cap_impossible = bool(n > K * cap) or bool(np.any(desired > cap))

    km = MiniBatchKMeans(n_clusters=K, random_state=int(seed), n_init=10)
    km.fit(X)
    centers = km.cluster_centers_

    labels = _balanced_assign_cap(X, centers, cap=cap)
    df["_gidx"] = labels.astype(int)

    meta = GroupMeta(
        K=K,
        cap=cap,
        n_points=n,
        cap_impossible=cap_impossible,
        method="kmeans_cap",
        target_min=target_min,
        target_max=target_max,
    )
    return df, meta


# =========================
# Refine Manual (override-safe)
# =========================
def refine_from_current(
    df: pd.DataFrame,
    K: int,
    cap: int,
    refine_iter: int,
    seed: int,
    override_map: Dict[str, int],
) -> pd.DataFrame:
    """
    Manual refine (centroid-based) with locked overrides.
    WARNING: can reintroduce mixing if overused in dense areas.
    """
    df = df.copy()
    X = df[["lat", "long"]].to_numpy(dtype=float)
    labels = df["_gidx"].to_numpy(dtype=int)
    K = int(K)
    cap = int(cap)

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
        centers = np.zeros((K, 2), dtype=float)
        for g in range(K):
            pts = X[labels == g]
            centers[g] = pts.mean(axis=0) if len(pts) else X.mean(axis=0)

        counts = np.bincount(labels, minlength=K).astype(int)

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
                if counts[g] >= cap:
                    continue
                counts[cur] -= 1
                counts[g] += 1
                labels[i] = g
                break

    df["_gidx"] = labels.astype(int)
    return df


# =========================
# Overrides
# =========================
def apply_overrides_to_current(
    df_base: pd.DataFrame,
    df_current: pd.DataFrame,
    override_map: Dict[str, int],
    K: int,
    mode: str,
) -> pd.DataFrame:
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


# =========================
# Colors
# =========================
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


# =========================
# Map (global search + popup lengkap)
# =========================
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

        for _, r in sub.iterrows():
            lat = float(r["lat"])
            lon = float(r["long"])
            name = str(r["nama_toko"])
            grp = label_from_gidx(g)

            popup_html = f"""
            <div style="font-size:13px">
              <b>{name}</b><br/>
              Grup: <b>{grp}</b><br/>
              Lat: {lat}<br/>Long: {lon}<br/>
              <a href="https://www.google.com/maps?q={lat},{lon}" target="_blank" rel="noopener noreferrer">
                🧭 Buka di Google Maps
              </a>
            </div>
            """

            # Visible point
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.85,
                tooltip=f"{name} | {grp}",
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(fg)

            # Search marker MUST also have popup
            folium.Marker(
                location=[lat, lon],
                tooltip=f"{name} | {grp}",
                popup=folium.Popup(popup_html, max_width=320),
                icon=folium.DivIcon(html=""),
            ).add_to(fg_search)

        # Boundary / hull (visual only)
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
