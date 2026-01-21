# app.py
import streamlit as st
import pandas as pd
from io import BytesIO
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

st.set_page_config(page_title="JKS Store Grouping", layout="wide")
st.title("📍 JKS Store Grouping")


# -------------------------
# Session state init
# -------------------------
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "df_base" not in st.session_state:
    st.session_state.df_base = None     # hasil initial (+ refine awal jika ON)
if "df_current" not in st.session_state:
    st.session_state.df_current = None  # hasil setelah override + (opsional) refine manual
if "override_map" not in st.session_state:
    st.session_state.override_map = {}  # key: _row_id (string) -> gidx(int)
if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None


# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("⚙️ Pengaturan")

K = st.sidebar.number_input("Jumlah group (R01–Rxx)", min_value=2, max_value=40, value=12, step=1)
CAP = st.sidebar.number_input("Max toko per group (cap)", min_value=5, max_value=80, value=25, step=1)

st.sidebar.divider()
refine_on = st.sidebar.checkbox("Refine otomatis saat initial", value=True)
refine_iter = st.sidebar.slider("Refine iter (initial)", 0, 10, 2)
neighbor_k = st.sidebar.slider("Neighbor-k (refine)", 5, 40, 12)

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


# -------------------------
# Front guidance BEFORE upload
# -------------------------
st.markdown("### 📌 Format file Excel yang harus di-upload")
st.markdown(
    """
- File: **.xlsx**
- Minimal kolom: **nama_toko | lat | long**
- **Lat/Long boleh pakai titik atau koma** (contoh: `-6.12345` atau `-6,12345`)
"""
)

dummy = pd.DataFrame(
    [
        {"nama_toko": "Toko A", "lat": -6.175392, "long": 106.827153},
        {"nama_toko": "Toko B", "lat": -6.176100, "long": 106.828900},
        {"nama_toko": "Toko C", "lat": -6.180200, "long": 106.820100},
    ]
)
st.markdown("**Contoh isi (dummy):**")
st.dataframe(dummy, width="stretch")

st.download_button(
    label="⬇️ Download template dummy (.xlsx)",
    data=df_to_excel_bytes(dummy),
    file_name="template_jks_grouping.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.divider()


# -------------------------
# Upload
# -------------------------
uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

if not uploaded_file:
    st.info("Upload file Excel dulu untuk mulai.")
    st.stop()

# reset session if file changed
if st.session_state.last_file_name != uploaded_file.name:
    st.session_state.last_file_name = uploaded_file.name
    st.session_state.df_clean = None
    st.session_state.df_base = None
    st.session_state.df_current = None
    st.session_state.override_map = {}

# Read excel
try:
    df_raw = pd.read_excel(uploaded_file, sheet_name=0)
except Exception as e:
    st.error(f"Gagal baca Excel. Pastikan file .xlsx valid.\n\nDetail: {e}")
    st.stop()

# Preview
st.subheader("📄 Preview Data")
if show_preview:
    st.dataframe(df_raw.head(preview_rows), width="stretch")
else:
    st.caption("Preview dimatikan (aktifkan di sidebar jika perlu).")

# Validate
ok, msg, df_clean = validate_input_df(df_raw)
if not ok:
    st.error(msg)
    st.stop()

st.success("Format file OK. Siap diproses.")
st.session_state.df_clean = df_clean


# -------------------------
# Initial grouping (cached in session)
# -------------------------
if st.session_state.df_base is None:
    with st.spinner("Initial grouping (dan refine jika ON)..."):
        df_base, folium_map, meta = initial_grouping(
            df_clean=st.session_state.df_clean,
            K=int(K),
            cap=int(CAP),              # <--- IMPORTANT: app pakai cap
            refine_on=bool(refine_on),
            refine_iter=int(refine_iter),
            neighbor_k=int(neighbor_k),
            seed=42,
            override_map={},           # override kosong di awal
        )
    st.session_state.df_base = df_base
    st.session_state.df_current = df_base.copy()

df_current = st.session_state.df_current


# -------------------------
# Summary
# -------------------------
st.subheader("✅ Ringkasan")
st.write(f"Total titik diproses: **{len(df_current):,}**")
st.dataframe(
    df_current["kategori"].value_counts().sort_index().rename_axis("kategori").to_frame("jumlah"),
    width="stretch",
)


# -------------------------
# Override UI (OPS A)
# - IMPORTANT: inside form to avoid auto rerun while typing/selecting
# -------------------------
st.subheader("🧷 Override Manual (Opsional)")
st.caption(
    "Override tidak akan mengubah hasil sampai kamu klik **Apply override**. "
    "Setelah override, sistem **TIDAK refine ulang otomatis**."
)

# Build selection list
# use _row_id as key to avoid duplicate nama_toko
df_show = df_current[["_row_id", "nama_toko", "lat", "long", "_gidx", "kategori"]].copy()
df_show["pilihan"] = df_show["nama_toko"].astype(str) + " | " + df_show["kategori"].astype(str) + " | id=" + df_show["_row_id"].astype(str)

with st.form("override_form", clear_on_submit=False):
    col1, col2 = st.columns([2, 1])

    with col1:
        selected = st.multiselect(
            "Pilih toko yang mau dipindahkan (bisa >1)",
            options=df_show["pilihan"].tolist(),
        )
    with col2:
        target_label = st.selectbox(
            "Pindahkan ke group:",
            options=[label_from_gidx(i) for i in range(int(K))],
            index=0,
        )

    apply_btn = st.form_submit_button("✅ Apply override")

if apply_btn:
    if len(selected) == 0:
        st.warning("Pilih minimal 1 toko untuk override.")
    else:
        target_g = parse_label_to_gidx(target_label)
        # build override_map by row_id
        for s in selected:
            rid = s.split("id=")[-1].strip()
            st.session_state.override_map[str(rid)] = int(target_g)

        # apply override ONLY (no refine)
        df_over, applied, skipped = apply_overrides(
            st.session_state.df_base.copy(),
            override_map=st.session_state.override_map,
            K=int(K),
            cap=int(CAP),
        )
        df_over["kategori"] = df_over["_gidx"].apply(label_from_gidx)

        st.session_state.df_current = df_over
        df_current = df_over

        st.success(f"Override applied: {applied} perubahan. Skipped: {skipped}. (Tidak refine ulang otomatis)")


# show current overrides
with st.expander("Lihat daftar override aktif"):
    if not st.session_state.override_map:
        st.write("- belum ada override -")
    else:
        tmp = []
        for rid, g in st.session_state.override_map.items():
            tmp.append({"_row_id": rid, "forced_group": label_from_gidx(g)})
        st.dataframe(pd.DataFrame(tmp), width="stretch")


# -------------------------
# Optional: Manual refine button (explicit)
# -------------------------
st.subheader("🔁 Refine Ulang (Manual)")
st.caption("Klik ini kalau kamu memang ingin rapatkan cluster lagi setelah override.")
colA, colB = st.columns([1, 3])
with colA:
    refine_btn = st.button("Refine sekarang")
with colB:
    refine_manual_iter = st.slider("Refine iter (manual)", 0, 20, 3)

if refine_btn:
    with st.spinner("Refining dari hasil current..."):
        df_ref = refine_from_current(
            st.session_state.df_current.copy(),
            K=int(K),
            cap=int(CAP),
            refine_iter=int(refine_manual_iter),
            neighbor_k=int(neighbor_k),
            seed=42,
            override_map=st.session_state.override_map,  # lock forced points
        )
        df_ref["kategori"] = df_ref["_gidx"].apply(label_from_gidx)
        st.session_state.df_current = df_ref
        df_current = df_ref
    st.success("Refine manual selesai.")


# -------------------------
# Map
# -------------------------
st.subheader("🗺️ Peta Grouping")
folium_map = build_map(df_current, K=int(K))
components.html(folium_map._repr_html_(), height=720, scrolling=True)


# -------------------------
# Download
# -------------------------
st.subheader("⬇️ Download")
st.download_button(
    label="Download Excel hasil grouping (termasuk override)",
    data=df_to_excel_bytes(df_current),
    file_name="hasil_grouping.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

with st.expander("Lihat tabel hasil (sample 200 baris)"):
    st.dataframe(df_current.head(200), width="stretch")
