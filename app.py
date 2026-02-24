# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 16:57:10 2026

@author: marta
"""

# Open in browser
# http://127.0.0.1:8050
#

import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import numpy as np
import os
import base64
import io
import plotly.graph_objects as go
from calculations_dash import ecbc_conversion

# Set working directory
# os.chdir(r"C:\Users\marta\Documents\GitHub\EC_BC_conversion")
app = dash.Dash(__name__)
server = app.server  # THIS MUST BE PRESENT
# ------------------------
# Helper functions
# ------------------------
def parse_csv(contents):
    """Parse uploaded CSV contents into pandas DataFrame."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode('utf-8')))

def safe_val(x):
    """Convert None to 0."""
    return 0 if x is None else x

# ------------------------
# Dash app
# ------------------------

app.layout = html.Div([

    html.H1("EC–BC Canonisation"),
    
    html.P("1. Upload your values.  2. Select your measurements metadata (dropdown tabs).  3. Export converted values."),
           # style={'marginBottom': '20px', 'fontSize': '16px', 'color': '#555'}),
    
    dcc.Upload(
        id="upload-data",
        children=html.Button("Upload CSV"),
        multiple=False
    ),

    # Dropdowns for parameters
    dcc.Dropdown(
        id="ec_bc",
        options=[{"label": "EC → ECnorm", "value": 0},
                 {"label": "eBC → ECnorm", "value": 1}],
        placeholder="EC or eBC"
    ),
    dcc.Dropdown(
        id="protocol",
        options=[{"label": "EUSAAR", "value": 0},
                 {"label": "NIOSH", "value": 1},
                 {"label": "Not applicable (eBC data)", "value": 2}],
        placeholder="Protocol"
    ),
    dcc.Dropdown(
        id="instrEC",
        options=[{"label": "High-Volume filters", "value": 0},
                 {"label": "Low-Volume Filters", "value": 1},
                 {"label": "Not applicable (eBC data)", "value": 2}],
        placeholder="EC instrument"
    ),
    dcc.Dropdown(
        id="instrBC",
        options=[{"label": "AE33", "value": 0},
                 {"label": "MAAP", "value": 1},
                 {"label": "Not applicable (EC data)", "value": 2}],
        placeholder="BC instrument"
    ),
    dcc.Dropdown(
        id="sizecut",
        options=[{"label": "PM1", "value": 2},
                 {"label": "PM2.5", "value": 0},
                 {"label": "PM10", "value": 3},
                 {"label": "PM TSP", "value": 4}],
        placeholder="Size cut"
    ),

    # Table for outputs
    dash_table.DataTable(
        id="output-table",
        columns=[
            {"name": "Datetime", "id": "Datetime"},
            {"name": "Original", "id": "original"},
            {"name": "Converted (median)", "id": "converted_EC_BC"},
            {"name": "p05", "id": "p05"},
            {"name": "p95", "id": "p95"}
        ],
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "padding": "5px"}
    ),

    # Graph for visualization
    dcc.Graph(id="output-graph"),

    # Manual CSV download
    html.Button("Download CSV", id="download-button", n_clicks=0),
    dcc.Download(id="download-dataframe-csv")
])

# ------------------------
# Main callback: compute and plot
# ------------------------
@app.callback(
    Output("output-graph", "figure"),
    Output("output-table", "data"),
    Input("upload-data", "contents"),
    Input("ec_bc", "value"),
    Input("protocol", "value"),
    Input("instrEC", "value"),
    Input("instrBC", "value"),
    Input("sizecut", "value")
)
def run_conversion(contents, ec_bc, prot, instrEC, instrBC, sizecut):

    # If no CSV uploaded, return empty outputs
    if contents is None:
        return go.Figure(), []

    # Convert None dropdowns to 0
    ec_bc = safe_val(ec_bc)
    prot = safe_val(prot)
    instrEC = safe_val(instrEC)
    instrBC = safe_val(instrBC)
    sizecut = safe_val(sizecut)
    
    na_to_zero = lambda x: 0 if x == 2 else x
    
    prot = na_to_zero(prot)
    instrEC = na_to_zero(instrEC)
    instrBC = na_to_zero(instrBC)
    
    # Parse CSV
    df = parse_csv(contents)
    n = len(df)

    # Map sizecut to binary flags for ecbc_conversion
    scec1 = np.full(n, int(sizecut == 2))
    scec10 = np.full(n, int(sizecut == 3))
    scbc10 = np.full(n, int(sizecut == 3))
    scbct = np.full(n, int(sizecut == 4))

    # Run conversion
    converted, conv_df = ecbc_conversion(
        orig=df["orig"].values,
        datetime=df["datetime"].values,
        ec_bc=np.full(n, ec_bc),
        prot=np.full(n, prot),
        instrEC=np.full(n, instrEC),
        instrBC=np.full(n, instrBC),
        scec1=scec1,
        scec10=scec10,
        scbc10=scbc10,
        scbct=scbct
    )

    # Compute 5th/95th percentiles for uncertainty
    lower = np.percentile(converted, 5, axis=1)
    upper = np.percentile(converted, 95, axis=1)
    conv_df["p05"] = lower
    conv_df["p95"] = upper

    # Create plot with median + uncertainty shading
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=conv_df["Datetime"], y=conv_df["original"],
        mode="lines", name="Original"
    ))
    fig.add_trace(go.Scatter(
        x=conv_df["Datetime"], y=conv_df["p95"],
        mode="lines", line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=conv_df["Datetime"], y=conv_df["p05"],
        mode="lines", fill="tonexty", fillcolor="rgba(0,100,80,0.2)",
        line=dict(width=0), name="90% uncertainty"
    ))
    fig.add_trace(go.Scatter(
        x=conv_df["Datetime"], y=conv_df["converted_EC_BC"],
        mode="lines", name="Converted (median)"
    ))

    fig.update_layout(
        title="EC–BC Canonisation",
        xaxis_title="Datetime",
        yaxis_title="Concentration"
    )

    return fig, conv_df.to_dict("records")

# ------------------------
# Manual download callback
# ------------------------
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-button", "n_clicks"),  # triggers only when button clicked
    State("output-table", "data"),         # table data as state
    prevent_initial_call=True
)
def download_csv(n_clicks, table_data):
    if not table_data:  # nothing to download
        return dash.no_update
    df = pd.DataFrame(table_data)
    return dcc.send_data_frame(df.to_csv, "EC_BC_converted.csv", index=False)

# ------------------------
# Run the app
# ------------------------
if __name__ == "__main__":
    app.run(debug=False)
