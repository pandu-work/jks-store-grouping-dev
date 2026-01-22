import hashlib
from io import BytesIO

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from grouping import (
    validate_input_df,
    initial_grouping,
    refine_from_current,
    apply_overrides,
    build_map,
    label_from_gidx,
    parse_label_to_gidx,
)

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="JKS Store Grouping", layout="wide")

DEFAULT_K = 12
DEFAULT_CAP = 25

# =============================
# HELPERS
# =============================
def file_hash(uploaded_file) -> str:
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="grouping")
    return bio.getvalue()

def init_state():
    st.session_state.file_hash = None
    st.session_state.df_base = None
    st.session_state.df_current = None
    st.session_state.override_map = {}

if "df_current" not in st.session_state:
    init_state()

# =============================
# TITLE
# =============================
st.title("🧭 JKS Store Grouping")

st.markdown(
"""
**Default sistem**
- 12 group (R01–R12)
- Kapasitas default: 25 toko / group  
- Refine **manual saja**
- Override **tidak auto-refine**
"""
)

# =============================
# DUMMY TEMPLATE (WAJIB)
# =============================
st.subheader("📄 Contoh Format Excel")
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

# =============================
# SIDEBAR
# =============================
st.sidebar.header("⚙️ Konfigurasi")
K = st.sidebar.number_input("Jumlah Group", 1, 50, DEFAULT_K)
CAP = st.sidebar.number_input("Cap per Group", 1, 200, DEFAULT_CAP)
refine_iter = st.sidebar.slider("Refine Iterasi", 1, 30, 3)
neighbor_k = st.sidebar.slider("Neighbor-k", 5, 50, 12)

# =============================
# UPLOAD
# =============================
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.stop()

fh = file_hash(uploaded)
if st.session_state.file_hash != fh:
    init_state()
    st.session_state.file_hash = fh

# =============================
# LOAD & VALIDATE
# =============================
if st.session_state.df_base is None:
    df_raw = pd.read_excel(uploaded)
    ok, msg, df_clean = validate_input_df(df_raw)
    if not ok:
        st.error(msg)
        st.stop()

    df_base, _, _ = initial_grouping(
        df_clean,
        K=int(K),
        cap=int(CAP),
        seed=42,
    )
    df_base["kategori"] = df_base["_gidx"].apply(label_from_gidx)

    st.session_state.df_base = df_base.copy()
    st.session_state.df_current = df_base.copy()

df_current = st.session_state.df_current

# =============================
# SUMMARY
# =============================
st.subheader("📊 Ringkasan Group")
st.dataframe(
    df_current["kategori"].value_counts().sort_index().to_frame("jumlah"),
    use_container_width=True,
)

# =============================
# OVERRIDE UI
# =============================
st.subheader("🧷 Override Manual")

df_opt = df_current[["_row_id", "nama_toko", "kategori"]].copy()
df_opt["label"] = (
    df_opt["nama_toko"].astype(str)
    + " | "
    + df_opt["kategori"]
    + " | id="
    + df_opt["_row_id"].astype(str)
)

with st.form("override_form"):
    selected = st.multiselect(
        "Pilih toko:",
        options=df_opt["label"].tolist(),
    )
    target_label = st.selectbox(
        "Pindahkan ke group:",
        options=[label_from_gidx(i) for i in range(int(K))],
    )
    apply_btn = st.form_submit_button("Apply Override")

if apply_btn and selected:
    tgt = parse_label_to_gidx(target_label)
    for s in selected:
        rid = s.split("id=")[-1]
        st.session_state.override_map[rid] = tgt

    df_current, _, _ = apply_overrides(
        df_current,
        st.session_state.override_map,
        K=int(K),
        cap=int(CAP),
    )
    df_current["kategori"] = df_current["_gidx"].apply(label_from_gidx)
    st.session_state.df_current = df_current

# =============================
# REMOVE OVERRIDE (WAJIB)
# =============================
st.markdown("### 🗑️ Hapus Override")
if st.session_state.override_map:
    rid_to_remove = st.multiselect(
        "Pilih override yang ingin dihapus:",
        options=list(st.session_state.override_map.keys()),
    )

    col1, col2 = st.columns(2)
    if col1.button("Hapus Selected"):
        for r in rid_to_remove:
            st.session_state.override_map.pop(r, None)
        df_current, _, _ = apply_overrides(
            st.session_state.df_current,
            st.session_state.override_map,
            K=int(K),
            cap=int(CAP),
        )
        df_current["kategori"] = df_current["_gidx"].apply(label_from_gidx)
        st.session_state.df_current = df_current

    if col2.button("Hapus SEMUA Override"):
        st.session_state.override_map = {}
else:
    st.caption("Belum ada override aktif")

# =============================
# REFINE (MANUAL ONLY)
# =============================
st.subheader("🧪 Refine Manual")
if st.button("Refine Sekarang"):
    df_ref = refine_from_current(
        st.session_state.df_current,
        K=int(K),
        cap=int(CAP),
        refine_iter=int(refine_iter),
        neighbor_k=int(neighbor_k),
        seed=42,
        override_map=st.session_state.override_map,
    )
    df_ref["kategori"] = df_ref["_gidx"].apply(label_from_gidx)
    st.session_state.df_current = df_ref
    df_current = df_ref

# =============================
# MAP
# =============================
st.subheader("🗺️ Peta Grouping")
m = build_map(df_current, K=int(K))
components.html(m._repr_html_(), height=700)

# =============================
# EXPORT
# =============================
st.subheader("⬇️ Download")
st.download_button(
    "Download Excel Hasil",
    data=df_to_excel_bytes(df_current),
    file_name="hasil_grouping.xlsx",
)
