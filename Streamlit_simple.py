import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Simple Streamlit App",
    page_icon="🚀",
    layout="wide"
)

# Main title
st.title("🚀 Simple Streamlit Application")
st.markdown("Welcome to your first Streamlit app!")

# Sidebar
st.sidebar.header("Controls")
name = st.sidebar.text_input("Enter your name:", "World")
number = st.sidebar.slider("Pick a number", 1, 100, 50)
option = st.sidebar.selectbox(
    "Choose an option:",
    ["Option A", "Option B", "Option C"]
)

# Main content
st.header(f"Hello, {name}! 👋")

# Two columns layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Selections")
    st.write(f"**Name:** {name}")
    st.write(f"**Number:** {number}")
    st.write(f"**Option:** {option}")
    
    # Interactive button
    if st.button("Click me!"):
        st.balloons()
        st.success("🎉 Button clicked!")

with col2:
    st.subheader("Random Data Visualization")
    
    # Generate sample data
    np.random.seed(number)  # Use the slider value as seed
    data = np.random.randn(50, 2)
    df = pd.DataFrame(data, columns=['x', 'y'])
    
    # Create a scatter plot
    fig, ax = plt.subplots()
    ax.scatter(df['x'], df['y'], alpha=0.6)
    ax.set_xlabel('X values')
    ax.set_ylabel('Y values')
    ax.set_title(f'Random Scatter Plot (seed: {number})')
    st.pyplot(fig)

# Data display
st.header("📊 Sample Data")
st.write("Here's the data used in the chart above:")
st.dataframe(df, use_container_width=True)

# File upload example
st.header("📁 File Upload")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("File contents:")
        st.dataframe(df_uploaded.head())
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")