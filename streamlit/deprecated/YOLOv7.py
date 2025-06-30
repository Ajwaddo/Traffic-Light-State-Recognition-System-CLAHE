import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="YOLOv7 Training Performance Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title and Description ---
st.title("🚀 YOLOv7 Training Performance Dashboard")
st.markdown("""
This dashboard visualizes the training and validation results from the `yolov7_results.csv` file.
Use the interactive charts below to analyze the model's performance metrics as it trained.
""")

# --- Load Data ---
# Function to load data with caching for better performance
@st.cache_data
def load_data(file_path):
    """
    Loads the yolov7_results.csv file, cleans the column names, and returns a DataFrame.
    """
    if not os.path.exists(file_path):
        return None
    try:
        # The YOLOv7 results file can sometimes have extra whitespace in headers
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading or parsing the CSV file: {e}")
        return None

# Path to your results file
RESULTS_FILE = 'yolov7_results.csv'

# Load the dataframe
df = load_data(RESULTS_FILE)

# --- Main Dashboard ---
if df is not None:
    st.sidebar.header("Dashboard Options")

    # --- Sidebar for Filtering ---
    # Slider to select the range of epochs to display on the charts
    epoch_slider = st.sidebar.slider(
        'Select Epoch Range',
        min_value=int(df['Epoch'].min()),
        max_value=int(df['Epoch'].max()),
        value=(int(df['Epoch'].min()), int(df['Epoch'].max()))
    )
    
    # Apply the filter to the dataframe
    df_filtered = df[(df['Epoch'] >= epoch_slider[0]) & (df['Epoch'] <= epoch_slider[1])]

    # --- Key Metrics Display for the Final Epoch---
    st.header("Key Performance Indicators (Final Epoch)")
    
    # Get the last row of the dataframe for the most recent metrics
    final_metrics = df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("mAP@0.5", f"{final_metrics['mAP@.5']:.4f}")
    col2.metric("mAP@0.5:0.95", f"{final_metrics['mAP@.5:.95']:.4f}")
    col3.metric("Precision", f"{final_metrics['P']:.4f}")
    col4.metric("Recall", f"{final_metrics['R']:.4f}")

    # --- Charts Section ---
    st.header("Performance Over Epochs")

    # 1. mAP (mean Average Precision) Chart
    st.subheader("Mean Average Precision (mAP)")
    fig_map = px.line(
        df_filtered, 
        x='Epoch', 
        y=['mAP@.5', 'mAP@.5:.95'],
        title='Validation mAP vs. Epoch',
        labels={'value': 'mAP Score', 'Epoch': 'Epoch'},
        markers=True
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # 2. Precision and Recall Chart
    st.subheader("Precision and Recall")
    fig_pr = px.line(
        df_filtered, 
        x='Epoch', 
        y=['P', 'R'],
        title='Validation Precision and Recall vs. Epoch',
        labels={'value': 'Score', 'Epoch': 'Epoch'},
        markers=True
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    # 3. Loss Functions Chart
    st.subheader("Loss Functions")
    # Using columns to display training and validation loss side-by-side
    col_train, col_val = st.columns(2)

    with col_train:
        fig_loss_train = px.line(
            df_filtered, 
            x='Epoch', 
            y=['box_loss', 'obj_loss', 'cls_loss', 'total_loss'],
            title='Training Loss vs. Epoch',
            labels={'value': 'Loss', 'Epoch': 'Epoch'}
        )
        st.plotly_chart(fig_loss_train, use_container_width=True)

    with col_val:
        fig_loss_val = px.line(
            df_filtered, 
            x='Epoch', 
            y=['val_box_loss', 'val_obj_loss', 'val_cls_loss'],
            title='Validation Loss vs. Epoch',
            labels={'value': 'Loss', 'Epoch': 'Epoch'}
        )
        st.plotly_chart(fig_loss_val, use_container_width=True)


    # --- Raw Data Display ---
    st.header("Raw Training Data")
    if st.checkbox('Show raw data from yolov7_results.csv', False):
        st.dataframe(df)

else:
    st.error(
        f"Could not find the results file: '{RESULTS_FILE}'. "
        "Please make sure the `yolov7_results.csv` file is in the same directory as this script."
    )