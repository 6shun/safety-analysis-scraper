import streamlit as st
import numpy as np  # Added missing import
import os
import glob
import re
from bs4 import BeautifulSoup
import pandas as pd
import io

# ============================================================
# 1. Caption pattern: Table X. Predicted/Expected Crash ... (Section 1)
# ============================================================
TABLE_TITLE_PATTERN = re.compile(
    r"Table\s+\d{1,2}\.\s*"
    r"(Predicted|Expected)\s+Crash Frequencies and Rates "
    r"by Highway Segment/Intersection \(Section 1\)", re.I
)

# Column name variants (for robustness across files)
EXPECTED_VARIANTS = {
    "Expected FI": [
        "Expected FI Crash Frequency (crashes/yr)",
        "Expected FI Crash Frequency",
        "Expected FI Crash Frequency (crashes/year)",
    ],
    "Expected PDO": [
        "Expected PDO Crash Frequency (crashes/yr)",
        "Expected PDO Crash Frequency",
        "Expected PDO Crash Frequency (crashes/year)",
    ],
    "Expected Total": [
        "Expected Total Crash Frequency (crashes/yr)",
        "Expected Total Crash Frequency",
        "Expected Total Crash Frequency (crashes/year)",
    ],
}

PREDICTED_VARIANTS = {
    "Predicted FI": [
        "Predicted FI Crash Frequency (crashes/yr)",
        "Predicted FI Crash Frequency",
        "Predicted FI Crash Frequency (crashes/year)",
    ],
    "Predicted PDO": [
        "Predicted PDO Crash Frequency (crashes/yr)",
        "Predicted PDO Crash Frequency",
        "Predicted PDO Crash Frequency (crashes/year)",
    ],
    "Predicted Total": [
        "Predicted Total Crash Frequency (crashes/yr)",
        "Predicted Total Crash Frequency",
        "Predicted Total Crash Frequency (crashes/year)",
    ],
}

# ============================================================
# 2. Helpers (FIXED: numpy.int64 compatibility)
# ============================================================
def pick_first_present(variants, columns):
    """Return first variant present in columns, or None."""
    for v in variants:
        if v in columns:
            return v
    return None

def find_section1_table(soup):
    """Return the table whose caption matches TABLE_TITLE_PATTERN."""
    for table in soup.find_all('table'):
        caption = table.find('caption')
        if not caption:
            continue
        caption_text = caption.get_text(strip=True)
        if TABLE_TITLE_PATTERN.search(caption_text):
            return table
    
    node = soup.find(string=TABLE_TITLE_PATTERN)
    if node:
        parent_table = node.find_parent('table')
        if parent_table:
            return parent_table
    return None

def extract_section1_df(html_path):
    """Open HTML and return the matched Section 1 table as a DataFrame."""
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    table = find_section1_table(soup)
    if table is None:
        raise RuntimeError("Could not find table with caption 'Table X. Predicted/Expected Crash Frequencies and Rates by Highway Segment/Intersection (Section 1)'.")
    df = pd.read_html(str(table))[0]
    return df

def consolidate_tbl(df):
    """Clean and compute difference columns. Expected columns optional."""
    df = df.copy()
    df.columns = [re.sub(r'\s+', ' ', str(c).strip()) for c in df.columns]
    cols = list(df.columns)
    
    seg_col = "Segment Number/Intersection Name/Cross Road"
    if seg_col not in cols:
        raise KeyError(f"Segment column '{seg_col}' not found. Available columns: {', '.join(cols)}")
    
    pred_fi_col = pick_first_present(PREDICTED_VARIANTS["Predicted FI"], cols)
    pred_pdo_col = pick_first_present(PREDICTED_VARIANTS["Predicted PDO"], cols)
    pred_tot_col = pick_first_present(PREDICTED_VARIANTS["Predicted Total"], cols)
    missing_pred = []
    for name, col in [("Predicted FI", pred_fi_col), ("Predicted PDO", pred_pdo_col), ("Predicted Total", pred_tot_col)]:
        if col is None:
            missing_pred.append(name)
    if missing_pred:
        raise KeyError(f"Missing required Predicted columns: {', '.join(missing_pred)}. Available headers: {', '.join(cols)}")
    
    exp_fi_col = pick_first_present(EXPECTED_VARIANTS["Expected FI"], cols)
    exp_pdo_col = pick_first_present(EXPECTED_VARIANTS["Expected PDO"], cols)
    exp_tot_col = pick_first_present(EXPECTED_VARIANTS["Expected Total"], cols)
    
    cols_to_keep = [seg_col, pred_fi_col, pred_pdo_col, pred_tot_col]
    if exp_fi_col: cols_to_keep.append(exp_fi_col)
    if exp_pdo_col: cols_to_keep.append(exp_pdo_col)
    if exp_tot_col: cols_to_keep.append(exp_tot_col)
    
    # FIXED: Use pd.isna() instead of .notnull() for numpy compatibility
    non_numeric_mask = df[seg_col].apply(lambda x: pd.to_numeric(str(x), errors='coerce')).isna()
    filtered_df = df.loc[non_numeric_mask, cols_to_keep].copy()
    
    rename_map = {
        seg_col: "Segment Number/Intersection Name/Cross Road",
        pred_fi_col: "Predicted FI Crash Frequency (crashes/yr)",
        pred_pdo_col: "Predicted PDO Crash Frequency (crashes/yr)",
        pred_tot_col: "Predicted Total Crash Frequency (crashes/yr)",
    }
    if exp_fi_col: rename_map[exp_fi_col] = "Expected FI Crash Frequency (crashes/yr)"
    if exp_pdo_col: rename_map[exp_pdo_col] = "Expected PDO Crash Frequency (crashes/yr)"
    if exp_tot_col: rename_map[exp_tot_col] = "Expected Total Crash Frequency (crashes/yr)"
    filtered_df.rename(columns=rename_map, inplace=True)
    
    if exp_fi_col:
        filtered_df["Expected FI - Predicted FI (crashes/yr)"] = (
            filtered_df["Expected FI Crash Frequency (crashes/yr)"] - 
            filtered_df["Predicted FI Crash Frequency (crashes/yr)"]
        )
    if exp_pdo_col:
        filtered_df["Expected PDO - Predicted PDO (crashes/yr)"] = (
            filtered_df["Expected PDO Crash Frequency (crashes/yr)"] - 
            filtered_df["Predicted PDO Crash Frequency (crashes/yr)"]
        )
    if exp_tot_col:
        filtered_df["Expected Total - Predicted Total (crashes/yr)"] = (
            filtered_df["Expected Total Crash Frequency (crashes/yr)"] - 
            filtered_df["Predicted Total Crash Frequency (crashes/yr)"]
        )
    
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    filtered_df[numeric_cols] = filtered_df[numeric_cols].round(4)
    return filtered_df

def rename_special_rows(df):
    """Rename 'All Segments', 'All Intersections', 'Total' into clearer labels."""
    df = df.copy()
    first_col = df.columns[0]
    row_mapping = {
        'All Segments': 'Total Segment Crashes',
        'All Intersections': 'Total Intersection Crashes',
        'Total': 'Total Corridor Crashes',
    }
    mask = df[first_col].isin(row_mapping.keys())
    df.loc[mask, first_col] = df.loc[mask, first_col].map(row_mapping)
    return df

def update_summary_tables(filename, consolidated_df, inter_summary, seg_summary):
    """Update intersection/segment summary DataFrames."""
    df = consolidated_df.copy()
    first_col = df.columns[0]
    seg_total_label = 'Total Segment Crashes'
    int_total_label = 'Total Intersection Crashes'
    corridor_total_label = 'Total Corridor Crashes'
    
    if len(df) > 1 or df[first_col].iloc[0] != corridor_total_label:
        mask_inter = df[first_col].isin([seg_total_label, int_total_label, corridor_total_label])
        inter_rows = df.loc[~mask_inter].copy()
        if not inter_rows.empty:
            inter_rows.insert(0, 'Source File', filename)
            inter_summary = pd.concat([inter_summary, inter_rows], ignore_index=True)
        
        mask_seg = df[first_col] == seg_total_label
        seg_rows = df.loc[mask_seg].copy()
        if not seg_rows.empty:
            seg_rows.insert(0, 'Source File', filename)
            seg_summary = pd.concat([seg_summary, seg_rows], ignore_index=True)
    else:
        total_row = df.copy()
        total_row.insert(0, 'Source File', filename)
        seg_summary = pd.concat([seg_summary, total_row], ignore_index=True)
    
    return inter_summary, seg_summary

def append_total_row(df, id_col_name='Source File'):
    """Append a Total row at the end of df, summing all numeric columns."""
    if df.empty:
        return df
    df = df.copy()
    total_row = {}
    total_row[id_col_name] = 'Total'
    for col in df.columns:
        if col == id_col_name:
            continue
        if df[col].dtype in [np.int64, np.int32, np.float64, np.float32, 'int64', 'int32', 'float64', 'float32']:
            total_row[col] = df[col].sum()
        else:
            total_row[col] = ''
    total_df = pd.DataFrame([total_row])
    df = pd.concat([df, total_df], ignore_index=True)
    return df

# ============================================================
# 3. Streamlit App Logic (unchanged)
# ============================================================
st.title("🚗 Predictive Safety HTML Scraper")
st.markdown("Upload HTML files containing crash frequency tables and extract Section 1 data.")

uploaded_files = st.file_uploader(
    "Choose HTML files", 
    type=['html', 'htm'], 
    accept_multiple_files=True,
    help="Upload one or more .html or .htm files containing the target tables."
)

if uploaded_files:
    st.info(f"Found {len(uploaded_files)} file(s) to process.")
    
    if st.button("🔄 Process Files", type="primary"):
        inter_summary = pd.DataFrame()
        seg_summary = pd.DataFrame()
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            status_text.text(f"Processing: {filename}...")
            
            temp_path = f"temp_{filename}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                consolidated = extract_section1_df(temp_path)
                consolidated = consolidate_tbl(consolidated)
                consolidated = rename_special_rows(consolidated)
                
                inter_summary, seg_summary = update_summary_tables(
                    filename, consolidated, inter_summary, seg_summary
                )
                results.append((filename, consolidated.shape[0], "✅ SUCCESS"))
                st.success(f"✓ {filename}: {consolidated.shape[0]} rows extracted")
                
            except RuntimeError as e:
                if "Could not find table with caption" in str(e):
                    st.warning(f"⚠️ Skipping {filename} - Section 1 table not present.")
                    results.append((filename, 0, "NO SECTION 1"))
                else:
                    st.error(f"❌ Error processing {filename}: {e}")
                    results.append((filename, 0, f"ERROR: {e}"))
            except Exception as e:
                st.error(f"❌ Error processing {filename}: {e}")
                results.append((filename, 0, f"ERROR: {e}"))
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        st.subheader("📊 Processing Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            successes = sum(1 for _, _, status in results if "SUCCESS" in status)
            st.metric("Successful", successes)
        with col2:
            st.metric("Total Files", len(uploaded_files))
        with col3:
            st.metric("Skipped/Errors", len(uploaded_files) - successes)
        
        if not inter_summary.empty:
            st.subheader("📈 Intersection Crashes Summary")
            inter_with_total = append_total_row(inter_summary, 'Source File')
            st.dataframe(inter_with_total, use_container_width=True)
            
            csv_buffer = io.StringIO()
            inter_with_total.to_csv(csv_buffer, index=False)
            st.download_button(
                "📥 Download Intersection Summary CSV",
                csv_buffer.getvalue(),
                f"intersection_crashes_summary.csv",
                "text/csv"
            )
        
        if not seg_summary.empty:
            st.subheader("📈 Segment Crashes Summary")
            seg_with_total = append_total_row(seg_summary, 'Source File')
            st.dataframe(seg_with_total, use_container_width=True)
            
            csv_buffer = io.StringIO()
            seg_with_total.to_csv(csv_buffer, index=False)
            st.download_button(
                "📥 Download Segment Summary CSV",
                csv_buffer.getvalue(),
                f"segment_crashes_summary.csv",
                "text/csv"
            )
        
        if not inter_summary.empty or not seg_summary.empty:
            batch_buffer = io.BytesIO()
            with pd.ExcelWriter(batch_buffer, engine='openpyxl') as writer:
                if not inter_summary.empty:
                    inter_with_total = append_total_row(inter_summary, 'Source File')
                    inter_with_total.to_excel(writer, sheet_name='Intersection Crashes', index=False)
                if not seg_summary.empty:
                    seg_with_total = append_total_row(seg_summary, 'Source File')
                    seg_with_total.to_excel(writer, sheet_name='Segment Crashes', index=False)
            batch_buffer.seek(0)
            st.download_button(
                "📊 Download Complete Batch Summary (Excel)",
                batch_buffer.getvalue(),
                "Batch_Summary.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

st.markdown("---")
st.caption("Fixed numpy compatibility error [file:1]")
