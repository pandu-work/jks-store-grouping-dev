# app.py
from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Tuple, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from grouping import (
    validate_input_df,
    initial_grouping_balanced_compact,
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
    st.session_state.grouping_params = None


if "df_current" not in st.session_state:
    init_state()


@st.cache_data(show_spinner=False)
def cached_read_excel(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def cached_validate(df_raw: pd.DataFrame):
    return validate_input_df(df_raw)


st.title("🧭 JKS Store Grouping")

st.markdown(
"""
**Cara pakai (mudah):**
1) Upload Excel  
2) Klik **Buat Grouping**  
3) (Opsional) Pindahkan toko pakai **Override**  
4) Download hasil
"""
)

with st.expander("📄 Template Excel (kalau format kamu belum sesuai)"):
    dummy = pd.DataFrame({
        "nama_toko": ["Toko A", "Toko B", "Toko C"],
        "lat": [-6.1201, -6.1212, -6.1223],
        "long": [106.8801, 106.8792, 106.8783],
    })
    st.dataframe(dummy, use_container_width=True)
    st.download_button(
        "⬇️ Download Template",
        data=df_to_excel_bytes(dummy),
        file_name="template_jks_grouping.xlsx",
        use_container_width=True,
    )

# Sidebar (simple)
st.sidebar.header("⚙️ Pengaturan")
K = st.sidebar.number_input("Jumlah Grup (R01–Rxx)", min_value=1, max_value=50, value=DEFAULT_K, step=1)
CAP = st.sidebar.number_input("Batas Maks Toko per Grup (Cap)", min_value=1, max_value=500, value=DEFAULT_CAP, step=1)

st.sidebar.caption("Catatan: Kalau total toko > (Jumlah Grup × Cap), cap tidak mungkin terpenuhi 100%.")

# Refine manual at bottom sidebar (advanced)
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Refine Manual (Opsional)")
st.sidebar.caption("Gunakan hanya kalau kamu ingin merapikan lagi. Override tidak akan hilang.")
refine_iter = st.sidebar.slider("Kekuatan Refine (Iterasi)", 1, 30, 8)
refine_now = st.sidebar.button("Refine Sekarang", use_container_width=True)

# Upload
st.subheader("1) Upload Excel")
uploaded = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.info("Silakan upload Excel untuk mulai.")
    st.stop()

file_bytes = uploaded.getvalue()
fh = hashlib.md5(file_bytes).hexdigest()

# reset if file changed
if st.session_state.file_hash != fh:
    init_state()
    st.session_state.file_hash = fh

# validate
if st.session_state.df_clean is None:
    df_raw = cached_read_excel(file_bytes)
    ok, msg, df_clean, _stats = cached_validate(df_raw)
    if not ok or df_clean is None:
        st.error(msg)
        st.stop()
    st.success(msg)
    st.session_state.df_clean = df_clean

# Build grouping button (explicit)
st.subheader("2) Buat Grouping")
colA, colB = st.columns([1, 2])
with colA:
    build_btn = st.button("✅ Buat Grouping", use_container_width=True)
with colB:
    st.caption(
        "Klik tombol ini setelah upload. Sistem akan membagi toko **rata dulu**, lalu merapikan agar tiap grup saling dekat."
    )

params_key = (int(K), int(CAP))
need_build = (st.session_state.df_base is None) or (st.session_state.grouping_params != params_key)

if build_btn or need_build:
    df_base, meta = initial_grouping_balanced_compact(
        st.session_state.df_clean.copy(),
        K=int(K),
        cap=int(CAP),
        seed=int(DEFAULT_SEED),
        refine_iter=14,
    )
    df_base["kategori"] = df_base["_gidx"].apply(label_from_gidx)

    st.session_state.df_base = df_base.copy()
    st.session_state.df_current = df_base.copy()
    st.session_state.override_map = {}
    st.session_state.grouping_params = params_key

    if meta.cap_impossible:
        st.warning(
            f"Total titik = {meta.n_points:,} lebih besar dari (Jumlah Grup × Cap) = {meta.K * meta.cap:,}. "
            "Artinya cap tidak bisa terpenuhi 100%. Sistem tetap membuat grup sedekat mungkin (best-effort)."
        )

df_current = st.session_state.df_current

# Summary
st.subheader("📊 Ringkasan Grup")
counts = df_current["_gidx"].value_counts().sort_index()
summary = pd.DataFrame({
    "Grup": [label_from_gidx(i) for i in range(int(K))],
    "Jumlah Toko": [int(counts.get(i, 0)) for i in range(int(K))],
})
summary["Cap"] = int(CAP)
summary["Melebihi Cap?"] = summary["Jumlah Toko"] > summary["Cap"]
st.dataframe(summary, use_container_width=True)

# Override
st.subheader("3) Pindahkan Toko (Override)")
st.caption("Pilih toko → pilih grup tujuan → klik Terapkan. Override akan menempel sampai kamu hapus.")

df_opt = df_current[["_row_id", "nama_toko", "_gidx"]].copy()
df_opt["Grup Saat Ini"] = df_opt["_gidx"].apply(label_from_gidx)
df_opt["Pilihan"] = (
    df_opt["nama_toko"].astype(str) + " | "
    + df_opt["Grup Saat Ini"].astype(str) + " | id="
    + df_opt["_row_id"].astype(str)
)

with st.form("override_form"):
    selected = st.multiselect("Pilih toko:", options=df_opt["Pilihan"].tolist())
    target_label = st.selectbox("Pindahkan ke grup:", options=[label_from_gidx(i) for i in range(int(K))])
    apply_btn = st.form_submit_button("✅ Terapkan Override")

if apply_btn:
    if not selected:
        st.warning("Kamu belum memilih toko.")
    else:
        tgt = parse_label_to_gidx(target_label)
        for s in selected:
            rid = s.split("id=")[-1].strip()
            st.session_state.override_map[rid] = int(tgt)

        df_current = apply_overrides_to_current(
            df_base=st.session_state.df_base,
            df_current=st.session_state.df_current,
            override_map=st.session_state.override_map,
            K=int(K),
            mode="current_only",
        )
        df_current["kategori"] = df_current["_gidx"].apply(label_from_gidx)
        st.session_state.df_current = df_current
        st.success("Override diterapkan.")

with st.expander("🗑️ Kelola Override (hapus override)", expanded=False):
    if st.session_state.override_map:
        override_df = pd.DataFrame(
            [{"_row_id": k, "forced_group": label_from_gidx(v)} for k, v in st.session_state.override_map.items()]
        )
        st.dataframe(override_df, use_container_width=True)

        to_remove = st.multiselect("Pilih override yang mau dihapus (_row_id):", options=list(st.session_state.override_map.keys()))
        col1, col2 = st.columns(2)

        if col1.button("Hapus Selected", use_container_width=True):
            for rid in to_remove:
                st.session_state.override_map.pop(rid, None)

            df_current = apply_overrides_to_current(
                df_base=st.session_state.df_base,
                df_current=st.session_state.df_current,
                override_map=st.session_state.override_map,
                K=int(K),
                mode="rebuild_from_base_then_apply",
            )
            df_current["kategori"] = df_current["_gidx"].apply(label_from_gidx)
            st.session_state.df_current = df_current
            st.success("Override selected dihapus.")

        if col2.button("Hapus SEMUA Override", use_container_width=True):
            st.session_state.override_map = {}
            df_current = st.session_state.df_base.copy()
            df_current["kategori"] = df_current["_gidx"].apply(label_from_gidx)
            st.session_state.df_current = df_current
            st.success("Semua override dihapus.")
    else:
        st.info("Belum ada override yang aktif.")

# Refine manual (sidebar bottom)
if refine_now:
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
    st.success("Refine selesai.")

# Map
st.subheader("🗺️ Peta Grup")
st.caption("Klik titik (atau cari nama toko) untuk melihat detail dan membuka Google Maps.")

m = build_map(st.session_state.df_current, K=int(K))
components.html(m._repr_html_(), height=720, scrolling=True)

# Download
st.subheader("4) Download Hasil")
st.download_button(
    "⬇️ Download Excel Hasil",
    data=df_to_excel_bytes(st.session_state.df_current),
    file_name="hasil_grouping.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)
