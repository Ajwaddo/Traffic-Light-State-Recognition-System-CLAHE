import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="YOLOv8 Baseline Training Dashboard",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title and Description ---
st.title("🔵 YOLOv8 Baseline Model Training Dashboard")
st.markdown("""
This dashboard visualizes the results for the **baseline YOLOv8n model**, trained on the original,
unprocessed LISA dataset. Use the charts and metrics below to analyze the model's performance.
""")

# --- Load Data ---
# Function to load data with caching to improve performance
@st.cache_data
def load_data(file_path):
    """
    Loads the results.csv file, cleans the column names, and returns a DataFrame.
    """
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path)
        # Clean up column names by stripping leading/trailing whitespace
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading or parsing the CSV file: {e}")
        return None

# Path to the results file
RESULTS_FILE = 'yolov8_base_results.csv'

# Load the dataframe
df = load_data(RESULTS_FILE)

# --- Main Dashboard ---
if df is not None:
    st.sidebar.header("Dashboard Options")

    # --- Sidebar for Filtering ---
    # Allow user to select which epochs to display
    epoch_slider = st.sidebar.slider(
        'Select Epoch Range',
        min_value=int(df['epoch'].min()),
        max_value=int(df['epoch'].max()),
        value=(int(df['epoch'].min()), int(df['epoch'].max()))
    )
    
    # Filter dataframe based on slider
    df_filtered = df[(df['epoch'] >= epoch_slider[0]) & (df['epoch'] <= epoch_slider[1])]

    # --- Key Metrics Display ---
    st.header("Key Performance Indicators (Final Epoch)")
    
    # Get the last row of the dataframe for the final metrics
    final_metrics = df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    # Ensure correct column names from yolov8_base_results.csv are used
    col1.metric("mAP@0.5", f"{final_metrics['metrics/mAP50(B)']:.3f}")
    col2.metric("mAP@0.5:0.95", f"{final_metrics['metrics/mAP50-95(B)']:.3f}")
    col3.metric("Precision", f"{final_metrics['metrics/precision(B)']:.3f}")
    col4.metric("Recall", f"{final_metrics['metrics/recall(B)']:.3f}")

    # --- Charts Section ---
    st.header("Performance Over Epochs")

    # 1. mAP (mean Average Precision) Chart
    st.subheader("Mean Average Precision (mAP)")
    fig_map = px.line(
        df_filtered, 
        x='epoch', 
        y=['metrics/mAP50(B)', 'metrics/mAP50-95(B)'],
        title='Validation mAP vs. Epoch',
        labels={'value': 'mAP Score', 'epoch': 'Epoch'},
        markers=True
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # 2. Precision and Recall Chart
    st.subheader("Precision and Recall")
    fig_pr = px.line(
        df_filtered, 
        x='epoch', 
        y=['metrics/precision(B)', 'metrics/recall(B)'],
        title='Validation Precision and Recall vs. Epoch',
        labels={'value': 'Score', 'epoch': 'Epoch'},
        markers=True
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    # 3. Loss Functions Chart
    st.subheader("Loss Functions")
    # Separate charts for training and validation loss for clarity
    col_train, col_val = st.columns(2)

    with col_train:
        fig_loss_train = px.line(
            df_filtered, 
            x='epoch', 
            y=['train/box_loss', 'train/cls_loss', 'train/dfl_loss'],
            title='Training Loss vs. Epoch',
            labels={'value': 'Loss', 'epoch': 'Epoch'}
        )
        st.plotly_chart(fig_loss_train, use_container_width=True)

    with col_val:
        fig_loss_val = px.line(
            df_filtered, 
            x='epoch', 
            y=['val/box_loss', 'val/cls_loss', 'val/dfl_loss'],
            title='Validation Loss vs. Epoch',
            labels={'value': 'Loss', 'epoch': 'Epoch'}
        )
        st.plotly_chart(fig_loss_val, use_container_width=True)


    # --- Raw Data Display ---
    st.header("Raw Training Data")
    if st.checkbox('Show raw data from yolov8_base_results.csv', False):
        st.dataframe(df)

else:
    st.error(
        f"Could not find the results file: '{RESULTS_FILE}'. "
        "Please make sure the `yolov8_base_results.csv` file is in the same directory as this script."
    )