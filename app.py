# app.py
import hashlib
from io import BytesIO

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from grouping import (
    validate_input_df,
    initial_grouping,
    apply_overrides,
    refine_from_current,
    build_map,
    label_from_gidx,
    parse_label_to_gidx,
)

st.set_page_config(page_title="JKS Grouping", layout="wide")
st.markdown("### 🧪 DEV VERSION (TEST ONLY) — DO NOT SHARE TO USERS", unsafe_allow_html=True)
st.title("🧭 JKS Grouping — Web-based Excel to Grouping Map")

# -------------------------
# Sidebar config
# -------------------------
st.sidebar.header("⚙️ Konfigurasi")
K = st.sidebar.number_input("Jumlah Group (K)", min_value=1, max_value=200, value=8, step=1)
CAP = st.sidebar.number_input("Cap max toko per group", min_value=1, max_value=5000, value=120, step=1)
refine_on_init = st.sidebar.checkbox("Refine otomatis saat initial grouping", value=True)
refine_iter_init = st.sidebar.slider("Refine iter (initial)", 0, 30, 3)
neighbor_k = st.sidebar.slider("Neighbor-k (refine)", 5, 60, 12)

st.sidebar.divider()
show_preview = st.sidebar.checkbox("Tampilkan preview tabel", value=True)
preview_rows = st.sidebar.slider("Jumlah baris preview", 5, 200, 30)

# -------------------------
# Helper: export
# -------------------------
def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="grouped")
    return bio.getvalue()

def hash_uploaded_file(uploaded_file) -> str:
    b = uploaded_file.getvalue()
    return hashlib.md5(b).hexdigest()

def init_session():
    st.session_state.file_hash = None
    st.session_state.df_clean = None     # cleaned input (nama_toko/lat/long + _row_id)
    st.session_state.df_work = None      # SINGLE source of truth (refine/override committed)
    st.session_state.override_map = {}   # row_id(str) -> gidx(int)

if "df_work" not in st.session_state:
    init_session()

# -------------------------
# Front guidance BEFORE upload
# -------------------------
st.markdown("### 📌 Format file Excel yang harus di-upload")
st.markdown(
    """
- File: **.xlsx**
- Minimal kolom: **nama_toko | lat | long**
- **Lat/Long boleh pakai titik atau koma** (contoh: `-6.12345` atau `-6,12345`)
- Kamu boleh refine berkali-kali sampai hasil pas, **baru** lakukan override.
"""
)

uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

if uploaded_file is None:
    st.info("Upload Excel dulu untuk mulai.")
    st.stop()

# -------------------------
# Reset state if file content changes (NOT by filename)
# -------------------------
current_hash = hash_uploaded_file(uploaded_file)
if st.session_state.file_hash != current_hash:
    init_session()
    st.session_state.file_hash = current_hash

# -------------------------
# Load & validate
# -------------------------
if st.session_state.df_clean is None:
    try:
        df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Gagal baca Excel: {e}")
        st.stop()

    ok, msg, df_clean = validate_input_df(df_raw)
    if not ok:
        st.error(msg)
        st.stop()

    st.session_state.df_clean = df_clean

# -------------------------
# Initial grouping (only once per file / reset)
# -------------------------
if st.session_state.df_work is None:
    with st.spinner("Membuat grouping awal..."):
        df_base, _ = initial_grouping(
            st.session_state.df_clean.copy(),
            K=int(K),
            cap=int(CAP),
            seed=42,
            refine_on=bool(refine_on_init),
            refine_iter=int(refine_iter_init),
            neighbor_k=int(neighbor_k),
            override_map={},  # no overrides at init
        )
        df_base["kategori"] = df_base["_gidx"].apply(label_from_gidx)
        st.session_state.df_work = df_base.copy()

df_work = st.session_state.df_work

# -------------------------
# Preview cleaned input
# -------------------------
if show_preview:
    st.subheader("🔎 Preview data input (cleaned)")
    st.dataframe(st.session_state.df_clean.head(preview_rows), width="stretch")

# -------------------------
# Summary
# -------------------------
st.subheader("✅ Ringkasan")
st.write(f"Total titik diproses: **{len(df_work):,}**")
st.dataframe(
    df_work["kategori"].value_counts().sort_index().rename_axis("kategori").to_frame("jumlah"),
    width="stretch",
)

with st.expander("🧠 Rules yang dipakai sistem (biar nggak 'balik ke versi lama')", expanded=False):
    st.markdown(
        """
**Source of truth = `df_work` (hasil terakhir yang sudah di-commit).**  
- Klik **Refine** → hasil refine akan **menggantikan** `df_work` (commit).
- Klik **Apply override** → override akan diterapkan ke **hasil terakhir** (`df_work`), **bukan** ke hasil awal.
- Setelah override, refine **tidak otomatis** jalan. Kalau kamu refine lagi, toko yang sudah di-override dianggap **LOCKED** (tidak boleh pindah).
- Upload file baru (konten beda) → semua state reset (refine/override hilang).
"""
    )

# -------------------------
# Override UI
# -------------------------
st.subheader("🧷 Override Manual (Opsional)")
st.caption(
    "Urutan ideal: **Refine beberapa kali sampai oke → baru override**. "
    "Override diterapkan ke hasil **terakhir** (tidak akan rollback)."
)

df_show = df_work[["_row_id", "nama_toko", "lat", "long", "_gidx", "kategori"]].copy()
df_show["pilihan"] = (
    df_show["nama_toko"].astype(str)
    + " | "
    + df_show["kategori"].astype(str)
    + " | id="
    + df_show["_row_id"].astype(str)
)

with st.form("override_form", clear_on_submit=False):
    col1, col2 = st.columns([2, 1])
    with col1:
        selected = st.multiselect(
            "Pilih toko yang mau di-override (pakai search di box ini):",
            options=df_show["pilihan"].tolist(),
        )
    with col2:
        target_label = st.selectbox(
            "Pindahkan ke group:",
            options=[label_from_gidx(i) for i in range(int(K))],
            index=0,
        )

    apply_btn = st.form_submit_button("Apply override")

if apply_btn:
    if not selected:
        st.warning("Belum pilih toko untuk di-override.")
    else:
        target_g = parse_label_to_gidx(target_label)

        for s in selected:
            rid = s.split("id=")[-1].strip()
            st.session_state.override_map[str(rid)] = int(target_g)

        # ✅ Apply override to LATEST committed result (df_work), not base
        df_over, applied, skipped = apply_overrides(
            st.session_state.df_work.copy(),
            override_map=st.session_state.override_map,
            K=int(K),
            cap=int(CAP),
        )
        df_over["kategori"] = df_over["_gidx"].apply(label_from_gidx)

        st.session_state.df_work = df_over
        df_work = df_over

        st.success(
            f"Override applied: {applied} perubahan. Skipped: {skipped}. "
            "Sistem tidak refine otomatis. (Kalau refine lagi, override dianggap LOCKED)"
        )

with st.expander("Lihat daftar override aktif"):
    if not st.session_state.override_map:
        st.write("- belum ada override -")
    else:
        tmp = [{"_row_id": rid, "forced_group": label_from_gidx(g)} for rid, g in st.session_state.override_map.items()]
        st.dataframe(pd.DataFrame(tmp), width="stretch")

# -------------------------
# Refine section (manual)
# -------------------------
st.subheader("🧪 Refine Manual")
colA, colB, colC = st.columns([1, 1, 2])
with colA:
    refine_btn = st.button("Refine sekarang")
with colB:
    refine_manual_iter = st.slider("Refine iter (manual)", 0, 30, 3)
with colC:
    st.caption(
        "Refine akan berjalan di atas hasil **terakhir**. "
        "Toko yang sudah di-override dianggap **LOCKED** dan tidak boleh pindah."
    )

if refine_btn:
    with st.spinner("Refining dari hasil terakhir (df_work)..."):
        df_ref = refine_from_current(
            st.session_state.df_work.copy(),
            K=int(K),
            cap=int(CAP),
            refine_iter=int(refine_manual_iter),
            neighbor_k=int(neighbor_k),
            seed=42,
            override_map=st.session_state.override_map,
        )
        df_ref["kategori"] = df_ref["_gidx"].apply(label_from_gidx)
        st.session_state.df_work = df_ref
        df_work = df_ref
    st.success("Refine manual selesai (committed).")

# -------------------------
# Map
# -------------------------
st.subheader("🗺️ Peta Grouping")
folium_map = build_map(df_work, K=int(K))
components.html(folium_map._repr_html_(), height=720, scrolling=True)

# -------------------------
# Download
# -------------------------
st.subheader("⬇️ Download")
st.download_button(
    label="Download Excel hasil grouping (termasuk override)",
    data=df_to_excel_bytes(df_work),
    file_name="hasil_grouping.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

with st.expander("Lihat tabel hasil (sample 200 baris)"):
    st.dataframe(df_work.head(200), width="stretch")
