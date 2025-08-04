import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.markdown(
    """
    <style>
    /* Apply gradient background */
    .stApp {
        background: linear-gradient(to bottom,   #FFD580, #CC5500);
        background-attachment: fixed;
    }

    /* Heading & text color for readability */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #4E342E !important;
    }

    /* Section divider style */
    hr {
        border: 1px solid #D3D3D3;
        width: 80%;
        margin: auto;
    }

    /* Chart & card background styling */
    .stPlotlyChart, .stAltairChart, .stVegaLiteChart, .stMarkdown {
        background-color: #FBE9E7 !important;
        color: #3E3E3E !important;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #D84315 !important;
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 8px 15px;
        font-size: 16px;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #BF360C !important;
    }

    /* Input & Textbox Styling */
    .stTextInput>div>div>input, .stTextArea>div>textarea {
        background-color: #1E1E1E !important;
        color: white !important;
        border: 1px solid #D3D3D3 !important;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
        <h1 style="font-size: 2.5em;">AQI ANALYSIS DASHBOARD</h1>
        <p style="font-size: 1.1em; margin-bottom: 10px;">See Your Air. Know Your Health.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

import streamlit as st
import pandas as pd

# Load dataset (Replace 'your_file.csv' with the actual file path)
@st.cache_data
def load_data():
    df = pd.read_csv("modified_AQI.csv")  # Ensure column names match exactly as in the chart
    return df

data = load_data()

# Streamlit app
st.title("Air Quality Index (AQI) Statistical Summary")

# Dropdown for selecting AQI or a pollutant
selected_column = st.selectbox("Select a parameter to view statistical summary:", ["AQI Value", "CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"])

# Show statistical summary for the selected column
st.subheader(f"Statistical Summary of {selected_column}")
summary = data[selected_column].describe()

# Custom CSS for styling metric cards
st.markdown("""
    <style>
        .metric-card {
            border: 3px solid black;
            padding: 10px;
            border-radius: 25px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            background-color: #CC5500;
            margin: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Function to display metrics with custom styling
def metric_card(label, value):
    st.markdown(f"""
        <div class="metric-card">
            <p style="margin:0;">{label}</p>
            <h3 style="margin:0;">{value}</h3>
        </div>
    """, unsafe_allow_html=True)

# Displaying metrics
cols1 = st.columns(3)
with cols1[0]: metric_card("Count", f"{summary['count']:.2f}")
with cols1[1]: metric_card("Min", f"{summary['min']:.2f}")
with cols1[2]: metric_card("Max", f"{summary['max']:.2f}")

cols2 = st.columns(3)
with cols2[0]: metric_card("25%", f"{summary['25%']:.2f}")
with cols2[1]: metric_card("50% (Median)", f"{summary['50%']:.2f}")
with cols2[2]: metric_card("75%", f"{summary['75%']:.2f}")

cols3 = st.columns(2)
with cols3[0]: metric_card("Mean", f"{summary['mean']:.2f}")
with cols3[1]: metric_card("Std Dev", f"{summary['std']:.2f}")


#pie chart
# --- Rest of your Streamlit content ---
# ... (Your other Streamlit components)

categories = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]

percentages = [46.2, 42.4, 5.4, 4.9, 0.8, 0.4]

total_rows = 16695

# Calculate counts from percentages
counts = [(p / 100) * total_rows for p in percentages]

# --- Colors for the Pie Chart ---
pie_colors = ["lime", "yellow", "orange", "red", "purple", "maroon"]

# --- Create the Pie Chart using Plotly ---
st.title("Distribution of AQI Categories")
fig_pie = go.Figure(
    data=[
        go.Pie(
            labels=categories,
            values=percentages,
            text=[f"{c:.1f}%" for c in percentages],
            textinfo="label+percent",
            hoverinfo="label+percent+text",
            hovertext=[f"Count: {int(count)}" for count in counts],
            marker=dict(colors=pie_colors, line=dict(color="black", width=1)),
            sort=False,  # Keep the order from the provided data
        )
    ]
)

fig_pie.update_layout(
    title="",  # Remove title as it's in the subheader
    showlegend=True,
    legend=dict(title="AQI Categories"),
    uniformtext_minsize=12,
    uniformtext_mode="hide",
    width=800,  # Adjust the width as needed
    height=400  # Adjust the height as needed
)

st.plotly_chart(fig_pie, use_container_width=True)


st.title("Average AQI & Pollutant Levels")

# --- Data for the Bar Chart ---
pollutants = ["AQI Value", "CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"]
average_values = [63.00, 1.34, 31.77, 3.82, 59.82]

# --- Colors from the Image ---
colors = ["#800000", "#228B22", "#DAA520", "#DC143C", "#9932CC"]  # Exact colors from the image

# --- Create the Bar Chart using Plotly ---
fig = go.Figure(
    data=[
        go.Bar(
            x=pollutants,
            y=average_values,
            text=[f"{val:.2f}" for val in average_values],  # Display values on top of bars
            textposition="outside",
            marker=dict(color=colors, line=dict(color="black", width=1)),
            hovertext=[f"Average Value: {val:.2f}" for val in average_values],  # Hover text
            hoverinfo="x+y+text"  # Show x, y, and hover text
        )
    ]
)

# --- Customize the Layout ---
fig.update_layout(
    xaxis_title="Pollutants",
    yaxis_title="Average AQI Value",
    yaxis=dict(gridcolor="lightgray", gridwidth=0.5),  # Gray gridlines
    plot_bgcolor="black",  # White background
    font=dict(size=12),
)

fig.update_xaxes(tickangle=20) # rotate x axis labels

# --- Display the Plot in Streamlit ---
st.plotly_chart(fig, use_container_width=True)



# Load dataset (Replace 'your_file.csv' with the actual file path)
@st.cache_data
def load_data():
    df = pd.read_csv("modified_AQI.csv")  # Ensure column names match exactly as in the chart
    return df

data = load_data()

# Define AQI categories and colors
def categorize_aqi(value):
    if value <= 50:
        return "green"
    elif value <= 100:
        return "yellow"
    elif value <= 150:
        return "orange"
    elif value <= 200:
        return "red"
    elif value <= 300:
        return "purple"
    else:
        return "maroon"

data["AQI Category"] = data["AQI Value"].apply(categorize_aqi)

# Streamlit app
st.title("Scatter Plot: AQI Value vs Different Pollutants")

# Define pollutant pairs
pollutants = [
    ("AQI Value", "CO AQI Value", "AQI Value vs CO AQI Value"),
    ("AQI Value", "Ozone AQI Value", "AQI Value vs Ozone AQI Value"),
    ("AQI Value", "NO2 AQI Value", "AQI Value vs NO2 AQI Value"),
    ("AQI Value", "PM2.5 AQI Value", "AQI Value vs PM2.5 AQI Value"),
]

# Create interactive plots
for x_col, y_col, title in pollutants:
    fig = px.scatter(
        data, x=x_col, y=y_col, color="AQI Category",
        category_orders={"AQI Category": ["green", "yellow", "orange", "red", "purple", "maroon"]},
        color_discrete_map={"green": "green", "yellow": "yellow", "orange": "orange", "red": "red", "purple": "purple", "maroon": "maroon"},
        title=title, hover_data={x_col: True, y_col: True}
    )
    st.plotly_chart(fig)




st.title("Distribution of AQI Values with AQI Colors")

# --- Data for the Bar Chart ---
aqi_ranges = ["0-50", "51-100", "101-200", "201-300", "301-500"]
frequencies = [7295, 7467, 869, 866, 136, 62]  # Frequencies from the image
aqi_levels = ["Good", "Moderate", "Unhealthy (Sensitive)", "Unhealthy", "Very Unhealthy", "Hazardous"]

# --- Colors from the Image ---
colors = ["green", "yellow", "orange", "red", "purple", "maroon"]

# --- Create the Bar Chart using Plotly ---
fig = go.Figure(
    data=[
        go.Bar(
            x=aqi_ranges,
            y=frequencies,
            text=frequencies,  # Display values on top of bars
            textposition="outside",
            marker=dict(color=colors, line=dict(color="black", width=1)),
            hovertext=[f"Frequency: {freq}" for freq in frequencies],  # Hover text
            hoverinfo="x+y+text"  # Show x, y, and hover text
        )
    ]
)

# --- Customize the Layout ---
fig.update_layout(
    xaxis_title="AQI Value",
    yaxis_title="Frequency",
    yaxis=dict(gridcolor="lightgray", gridwidth=0.5),  # Gray gridlines
    plot_bgcolor="black",  # black background
    font=dict(size=12),
    width=1600,  # Adjusted width to match the image
    height=600,
    showlegend=True,  # Show legend
    legend_title_text="AQI Levels",  # Legend title
    legend=dict(
        x=1.02,  # Place legend outside the plot
        y=1,
        bgcolor="black",
        bordercolor="black",
        borderwidth=1,
    ),
)

# --- Add Legend Items Manually ---
for i, level in enumerate(aqi_levels):
    fig.add_trace(go.Bar(
        x=[None],  # Empty x-axis data to create legend items
        y=[None],  # Empty y-axis data to create legend items
        marker=dict(color=colors[i]),
        name=level,
        showlegend=True,
    ))

# --- Display the Plot in Streamlit ---
st.plotly_chart(fig, use_container_width=True)

st.title("AQI Map: Air Quality Level")

try:
    df = pd.read_csv('modified_AQI.csv')
    # Assuming your CSV has columns named 'Longitude', 'Latitude', 'AQI Value'
    if not all(col in df.columns for col in ['lng', 'lat', 'AQI Value']):
        st.error("Error: The CSV file must contain the required columns: 'Longitude', 'Latitude', and 'AQI Value'.")
        st.stop()
except Exception as e:
    st.error(f"Error: Could not read CSV file. Please ensure the link is correct and the file is a valid CSV. Error details: {e}")
    st.stop()

# --- Define Color Mapping based on AQI Value Ranges ---
def get_aqi_color(aqi_value):
    if 0 <= aqi_value <= 50:
        return 'green'
    elif 51 <= aqi_value <= 100:
        return 'yellow'
    elif 101 <= aqi_value <= 200:
        return 'orange'
    elif 201 <= aqi_value <= 300:
        return 'red'
    elif 301 <= aqi_value <= 500:
        return 'purple'
    else:
        return 'maroon' # For values > 500 or negative (you might want to handle these differently)

df['Color'] = df['AQI Value'].apply(get_aqi_color)

# --- Create Scatter Plot with Map ---
fig = go.Figure(go.Scattergeo(
    lon=df['lng'],
    lat=df['lat'],
    mode='markers',
    marker=dict(
        size=8,  # Adjust marker size as needed
        color=df['Color'],
        opacity=0.7,
        line=dict(width=0.5, color='black')
    ),
    hoverinfo='text',
    text=[f"AQI: {val}<br>Color: {color.capitalize()}" for val, color in zip(df['AQI Value'], df['Color'])]
))

# --- Customize Layout ---
fig.update_layout(
    title="AQI Map: Air Quality Level",
)

# --- Add Legend ---
legend_entries = [
    ('green', '0-50'),
    ('yellow', '51-100'),
    ('orange', '101-200'),
    ('red', '201-300'),
    ('purple', '301-500'),
    ('maroon', '> 500'),
]

for color, range_label in legend_entries:
    fig.add_trace(go.Scattergeo(
        lon=[None],
        lat=[None],
        mode='markers',
        marker=dict(size=10, color=color),
        name=f"{range_label}",
        showlegend=True
    ))

# --- Display Plot in Streamlit ---
st.plotly_chart(fig, use_container_width=True)


st.markdown("""
    <p style="text-align: center; font-size: 16px;">
        | Built with ❤️ using Streamlit
    </p>
""", unsafe_allow_html=True)
