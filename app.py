# app.py
from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Dict, Tuple, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from grouping import (
    validate_input_df,
    initial_grouping,
    refine_from_current,
    apply_overrides_to_current,
    build_map,
    label_from_gidx,
    parse_label_to_gidx,
)

st.set_page_config(page_title="JKS Store Grouping", layout="wide")

DEFAULT_K = 12
DEFAULT_CAP = 25
DEFAULT_SEED = 42


# ---------- helpers ----------
def file_hash(uploaded_file) -> str:
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="grouping")
    return bio.getvalue()


def init_state():
    st.session_state.file_hash = None
    st.session_state.df_clean = None
    st.session_state.df_base = None
    st.session_state.df_current = None
    st.session_state.override_map = {}


if "df_current" not in st.session_state:
    init_state()


# ---------- caching ----------
@st.cache_data(show_spinner=False)
def cached_read_excel(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def cached_validate(df_raw: pd.DataFrame) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    return validate_input_df(df_raw)


@st.cache_data(show_spinner=False)
def cached_initial_grouping(df_clean: pd.DataFrame, K: int, cap: int, seed: int) -> Tuple[pd.DataFrame, dict]:
    df_base, meta = initial_grouping(df_clean.copy(), K=K, cap=cap, seed=seed)
    df_base["kategori"] = df_base["_gidx"].apply(label_from_gidx)
    meta_dict = {
        "K": meta.K,
        "cap": meta.cap,
        "n_points": meta.n_points,
        "cap_impossible": meta.cap_impossible,
    }
    return df_base, meta_dict


# ---------- title ----------
st.title("🧭 JKS Store Grouping")

st.markdown(
"""
**Kontrak logic (final):**
- Default **12 group (R01–R12)**, cap default **25**
- **Refine manual saja** (tidak ada auto-refine)
- Override menempel ke hasil terakhir (`df_current`) dan **tidak auto-refine**
- Override bisa **hapus sebagian / hapus semua**
- Boundary/hull **WAJIB** + warna per group **WAJIB**
- Search di map **GLOBAL**
"""
)

# ---------- dummy template ----------
st.subheader("📄 Contoh Format Excel (Template)")
dummy = pd.DataFrame({
    "nama_toko": ["Toko A", "Toko B", "Toko C"],
    "lat": [-6.1201, -6.1212, -6.1223],
    "long": [106.8801, 106.8792, 106.8783],
})
st.dataframe(dummy, use_container_width=True)
st.download_button(
    "⬇️ Download Template Excel",
    data=df_to_excel_bytes(dummy),
    file_name="template_jks_grouping.xlsx",
)

# ---------- sidebar ----------
st.sidebar.header("⚙️ Konfigurasi")
K = st.sidebar.number_input("Jumlah Group", min_value=1, max_value=50, value=DEFAULT_K, step=1)
CAP = st.sidebar.number_input("Cap per Group", min_value=1, max_value=500, value=DEFAULT_CAP, step=1)
refine_iter = st.sidebar.slider("Refine Iterasi", 1, 30, 3)

# ---------- upload ----------
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.stop()

file_bytes = uploaded.getvalue()
fh = hashlib.md5(file_bytes).hexdigest()

# Reset state only when file changes
if st.session_state.file_hash != fh:
    init_state()
    st.session_state.file_hash = fh

# ---------- load & validate ----------
if st.session_state.df_clean is None:
    df_raw = cached_read_excel(file_bytes)
    ok, msg, df_clean = cached_validate(df_raw)

    if not ok or df_clean is None:
        st.error(msg)
        st.stop()

    if msg:
        st.info(msg)

    st.session_state.df_clean = df_clean

# ---------- initial grouping (NO auto refine) ----------
if st.session_state.df_base is None:
    df_base, meta = cached_initial_grouping(
        st.session_state.df_clean.copy(),
        K=int(K),
        cap=int(CAP),
        seed=int(DEFAULT_SEED),
    )

    st.session_state.df_base = df_base.copy()
    st.session_state.df_current = df_base.copy()

    if meta.get("cap_impossible", False):
        st.warning(
            f"Total titik = {meta['n_points']:,} lebih besar dari K×cap = {meta['K']*meta['cap']:,}. "
            "Secara matematis tidak mungkin semua group ≤ cap. Sistem akan best-effort."
        )

df_current = st.session_state.df_current

# ---------- ringkasan ----------
st.subheader("📊 Ringkasan Group")
counts = df_current["_gidx"].value_counts().sort_index()
summary = pd.DataFrame({
    "kategori": [label_from_gidx(i) for i in range(int(K))],
    "jumlah": [int(counts.get(i, 0)) for i in range(int(K))],
})
summary["cap"] = int(CAP)
summary["over_cap"] = summary["jumlah"] > summary["cap"]

# show warning if any over cap
if summary["over_cap"].any():
    over = summary[summary["over_cap"]]
    st.warning("Ada group yang melebihi cap. (Ini bisa terjadi karena cap impossible atau override).")

st.dataframe(summary, use_container_width=True)

# ---------- override apply ----------
st.subheader("🧷 Override Manual")

df_opt = df_current[["_row_id", "nama_toko", "_gidx"]].copy()
df_opt["kategori"] = df_opt["_gidx"].apply(label_from_gidx)

# safer label formatting (avoid split bugs)
df_opt["label"] = (
    df_opt["nama_toko"].astype(str)
    + " | "
    + df_opt["kategori"].astype(str)
    + " | id="
    + df_opt["_row_id"].astype(str)
)

with st.form("override_form"):
    selected = st.multiselect("Pilih toko:", options=df_opt["label"].tolist())
    target_label = st.selectbox(
        "Pindahkan ke group:",
        options=[label_from_gidx(i) for i in range(int(K))]
    )
    apply_btn = st.form_submit_button("Apply Override")

if apply_btn:
    if not selected:
        st.warning("Belum pilih toko.")
    else:
        tgt = parse_label_to_gidx(target_label)
        for s in selected:
            rid = s.split("id=")[-1].strip()
            st.session_state.override_map[rid] = int(tgt)

        # apply override to df_current (no rebuild / no rollback)
        df_current = apply_overrides_to_current(
            df_base=st.session_state.df_base,
            df_current=st.session_state.df_current,
            override_map=st.session_state.override_map,
            K=int(K),
            mode="current_only",
        )
        df_current["kategori"] = df_current["_gidx"].apply(label_from_gidx)
        st.session_state.df_current = df_current
        st.success("Override diterapkan (tanpa auto-refine).")

# ---------- remove override ----------
st.markdown("### 🗑️ Hapus Override (WAJIB)")
if st.session_state.override_map:
    override_df = pd.DataFrame(
        [{"_row_id": k, "forced_group": label_from_gidx(v)} for k, v in st.session_state.override_map.items()]
    )
    st.dataframe(override_df, use_container_width=True)

    to_remove = st.multiselect(
        "Pilih override yang mau dihapus (_row_id):",
        options=list(st.session_state.override_map.keys())
    )

    col1, col2 = st.columns(2)

    if col1.button("Hapus Selected"):
        for rid in to_remove:
            st.session_state.override_map.pop(rid, None)

        # rebuild from base then apply remaining overrides
        df_current = apply_overrides_to_current(
            df_base=st.session_state.df_base,
            df_current=st.session_state.df_current,
            override_map=st.session_state.override_map,
            K=int(K),
            mode="rebuild_from_base_then_apply",
        )
        df_current["kategori"] = df_current["_gidx"].apply(label_from_gidx)
        st.session_state.df_current = df_current
        st.success("Override selected dihapus (tanpa auto-refine).")

    if col2.button("Hapus SEMUA Override"):
        st.session_state.override_map = {}
        df_current = st.session_state.df_base.copy()
        df_current["kategori"] = df_current["_gidx"].apply(label_from_gidx)
        st.session_state.df_current = df_current
        st.success("Semua override dihapus (kembali ke baseline initial grouping, tanpa refine).")
else:
    st.caption("Belum ada override aktif.")

# ---------- refine manual only ----------
st.subheader("🧪 Refine Manual (HANYA jika tombol diklik)")
if st.button("Refine Sekarang"):
    df_ref = refine_from_current(
        st.session_state.df_current.copy(),
        K=int(K),
        cap=int(CAP),
        refine_iter=int(refine_iter),
        seed=int(DEFAULT_SEED),
        override_map=st.session_state.override_map,
    )
    df_ref["kategori"] = df_ref["_gidx"].apply(label_from_gidx)
    st.session_state.df_current = df_ref
    df_current = df_ref
    st.success("Refine selesai (manual).")

# ---------- map ----------
st.subheader("🗺️ Peta Grouping")
m = build_map(st.session_state.df_current, K=int(K))
components.html(m._repr_html_(), height=720, scrolling=True)

# ---------- download ----------
st.subheader("⬇️ Download")
st.download_button(
    "Download Excel Hasil (df_current)",
    data=df_to_excel_bytes(st.session_state.df_current),
    file_name="hasil_grouping.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
