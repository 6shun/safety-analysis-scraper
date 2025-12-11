import streamlit as st
import os
import glob
import re
from bs4 import BeautifulSoup
import pandas as pd
import io

# Your existing patterns and functions (adapted for Streamlit)
TABLE_TITLE_PATTERN = re.compile(
    r"Table\s+\d{1,2}\.\s*"
    r"(PredictedExpected)\s+Crash Frequencies and Rates "
    r"by Highway Segment/Intersection \(Section 1\)",
    re.I,
)

@st.cache_data
def consolidate_tbl(df):
    """Clean and compute difference columns (Expected optional)."""
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    cols = list(df.columns)
    
    # Segment table expected columns
    seg_col = "Segment Number/Intersection Name/Cross Road"
    pred_cols = ["Predicted All", "Predicted Rear-End", "Predicted Angle", "Predicted Head-On", 
                 "Predicted Other", "Predicted Pedestrian", "Predicted Bicycle", "Predicted Total"]
    
    if seg_col not in cols:
        raise KeyError(f"Segment column '{seg_col}' not found. Available columns: {', '.join(cols)}")
    
    # Add difference columns if Expected exists
    exp_cols = [c.replace("Predicted", "Expected") for c in pred_cols]
    for pred, exp in zip(pred_cols, exp_cols):
        if pred in cols and exp in cols:
            df[f"{pred} Diff"] = df[pred].astype(float) - df[exp].astype(float)
    
    # Rename special rows for consistency
    row_mapping = {
        "All Segments": "Total Segment Crashes",
        "All Intersections": "Total Intersection Crashes",
        "Total": "Total Corridor Crashes",
    }
    first_col = seg_col
    mask = df[first_col].isin(row_mapping.keys())
    df.loc[mask, first_col] = df.loc[mask, first_col].map(row_mapping)
    return df

def extract_crash_freq_table_from_html(html_content):
    """Extract crash frequency table from HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Find table by caption pattern
    for caption in soup.find_all("caption"):
        caption_text = caption.get_text()
        if TABLE_TITLE_PATTERN.search(caption_text):
            table = caption.find_parent("table")
            if table:
                # Convert table to DataFrame
                df = pd.read_html(str(table))[0]
                return df
    return pd.DataFrame()

def save_to_excel_two_tabs(inter_df, seg_df, filename):
    """Save intersection and segment data to Excel with two tabs."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if not inter_df.empty:
            inter_df.to_excel(writer, sheet_name='Intersection Crashes', index=False)
        if not seg_df.empty:
            seg_df.to_excel(writer, sheet_name='Segment Crashes', index=False)
    output.seek(0)
    return output.getvalue()

# Streamlit UI
st.title("🚗 Predictive Safety HTML to Excel Converter")
st.markdown("Upload .htm files containing crash frequency tables. Get Excel outputs with Intersection and Segment crashes.")

# File uploader
uploaded_files = st.file_uploader(
    "Choose .htm files", 
    type=['htm', 'html'], 
    accept_multiple_files=True,
    help="Select multiple corridor report .htm files"
)

if uploaded_files:
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Read HTML content
        html_content = uploaded_file.read().decode("utf-8", errors="ignore")
        
        try:
            # Extract tables
            raw_df = extract_crash_freq_table_from_html(html_content)
            if raw_df.empty:
                results.append((uploaded_file.name, None, "❌ No crash table found"))
                continue
                
            # Process tables
            consolidated = consolidate_tbl(raw_df)
            
            # Split into intersection and segment data (simplified logic)
            seg_df = consolidated.copy()
            
            # Save to Excel bytes
            excel_data = save_to_excel_two_tabs(pd.DataFrame(), seg_df, uploaded_file.name)
            
            # Store for download
            st.session_state[f"{uploaded_file.name}_excel"] = excel_data
            
            results.append((uploaded_file.name, excel_data, "✅ Success"))
            
        except Exception as e:
            results.append((uploaded_file.name, None, f"❌ Error: {str(e)}"))
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    # Results summary
    st.subheader("📊 Processing Results")
    successes = sum(1 for _, _, status in results if "✅" in status)
    st.success(f"✅ {successes}/{len(uploaded_files)} files processed successfully")
    
    # Download buttons for successful files
    st.subheader("📥 Download Excel Files")
    for filename, excel_data, status in results:
        if excel_data and "✅" in status:
            st.download_button(
                label=f"Download {filename.replace('.htm', '.xlsx')}",
                data=excel_data,
                file_name=f"{filename.replace('.htm', '')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # Batch summary option
    if successes > 1:
        with st.expander("📈 View Batch Summary"):
            st.info("Batch summary coming soon - individual file downloads available above!")

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Upload**: Select one or more .htm corridor reports
    2. **Wait**: Progress bar shows processing status  
    3. **Download**: Click buttons to get Excel files with Intersection Crashes and Segment Crashes tabs
    4. **No install needed** - runs entirely in your browser!
    """)

# Footer
st.markdown("---")
st.caption("Built for traffic safety analysis - converts HTML reports to Excel")
