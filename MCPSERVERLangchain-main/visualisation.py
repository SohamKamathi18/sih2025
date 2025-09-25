from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Visualisation")


@mcp.tool()
def plot_data(
    table: Dict[str, Any],
    plot_type: str = "auto",
    x: Optional[str] = None,
    y: Optional[str] = None,
    z: Optional[str] = None,
    color: Optional[str] = None,
) -> Dict[str, Any]:
    """Render a Plotly figure from tabular data provided by another tool."""

    columns = table.get("columns")
    rows = table.get("rows")

    if not columns or rows is None:
        return {"error": "table must include 'columns' and 'rows'."}

    df = pd.DataFrame(rows, columns=columns)
    cols_lower = {col.lower(): col for col in columns}

    def resolve_column(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        lowered = name.lower()
        if lowered in cols_lower:
            return cols_lower[lowered]
        if name in columns:
            return name
        return None

    x_col = resolve_column(x)
    y_col = resolve_column(y)
    z_col = resolve_column(z)
    color_col = resolve_column(color)

    inferred_plot = plot_type.lower()
    if inferred_plot == "auto":
        inferred_plot = _infer_plot_type(df)

    try:
        fig = _build_plot(
            df=df,
            plot_type=inferred_plot,
            x=x_col,
            y=y_col,
            z=z_col,
            color=color_col,
        )
    except ValueError as exc:
        return {"error": str(exc)}

    buf = io.BytesIO()
    fig.write_image(buf, format="png")
    buf.seek(0)
    img_bytes = buf.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    payload: Dict[str, Any] = {
        "media": {
            "type": "image/png",
            "base64": img_base64,
            "data_uri": f"data:image/png;base64,{img_base64}",
        },
        "figure_json": fig.to_json(),
        "details": {
            "plot_type": inferred_plot,
            "x": x_col,
            "y": y_col,
            "z": z_col,
            "color": color_col,
            "row_count": len(df),
        },
    }
    return payload


def _infer_plot_type(df: pd.DataFrame) -> str:
    """Choose a plot when none is specified based on available columns."""

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    lower_names = {col.lower() for col in df.columns}

    if {"latitude", "lat"} & lower_names and {"longitude", "lon", "lng"} & lower_names:
        return "scattergeo"

    if len(df.columns) == 3 and all(col in numeric_cols for col in df.columns):
        return "heatmap"

    if len(df.columns) >= 2 and len(numeric_cols) >= 2:
        return "line"

    if len(df.columns) >= 2 and numeric_cols:
        return "bar"

    if len(df.columns) == 1 and numeric_cols:
        return "histogram"

    return "table"


def _build_plot(
    df: pd.DataFrame,
    plot_type: str,
    x: Optional[str],
    y: Optional[str],
    z: Optional[str],
    color: Optional[str],
) -> go.Figure:
    """Construct the requested Plotly figure."""

    plot_type = plot_type.lower()

    if plot_type == "table":
        header = dict(values=list(df.columns), fill_color="#1f77b4", align="left")
        cells = dict(values=[df[col] for col in df.columns], align="left")
        return go.Figure(data=[go.Table(header=header, cells=cells)])

    if plot_type == "histogram":
        column = x or y or df.columns[0]
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError("Histogram requires a numeric column.")
        fig = go.Figure(go.Histogram(x=df[column], name=column))
        fig.update_layout(title=f"Histogram of {column}")
        return fig

    if plot_type == "scattergeo":
        lat_col = _find_column(df, x, ["lat", "latitude"])
        lon_col = _find_column(df, y, ["lon", "lng", "longitude"])
        if not lat_col or not lon_col:
            raise ValueError("Scattergeo requires latitude and longitude columns.")
        text_col = color or next((col for col in df.columns if col not in {lat_col, lon_col}), None)
        fig = go.Figure(
            go.Scattergeo(
                lat=df[lat_col],
                lon=df[lon_col],
                mode="markers",
                marker=dict(size=8, color="blue"),
                text=df[text_col] if text_col else None,
            )
        )
        fig.update_layout(title="Geographical Scatter Plot", geo=dict(scope="world"))
        return fig

    if plot_type == "heatmap":
        if z is None and len(df.columns) < 3:
            raise ValueError("Heatmap requires three columns or an explicit z column.")
        x_col = x or df.columns[0]
        y_col = y or df.columns[1 if len(df.columns) > 1 else 0]
        z_col = z or df.columns[2 if len(df.columns) > 2 else 0]
        for col in (x_col, y_col, z_col):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' required for heatmap not found.")
        if not all(pd.api.types.is_numeric_dtype(df[col]) for col in (x_col, y_col, z_col)):
            raise ValueError("Heatmap requires numeric x, y, and z columns.")
        fig = go.Figure(data=go.Heatmap(x=df[x_col], y=df[y_col], z=df[z_col], colorscale="Viridis"))
        fig.update_layout(title="Heatmap")
        return fig

    if plot_type in {"line", "scatter", "bar"}:
        x_col = x or df.columns[0]
        y_col = y or (df.columns[1] if len(df.columns) > 1 else None)
        if not y_col:
            raise ValueError(f"Plot type '{plot_type}' requires at least two columns.")

        if plot_type == "line":
            if not all(pd.api.types.is_numeric_dtype(df[col]) for col in (x_col, y_col)):
                raise ValueError("Line chart requires numeric x and y columns.")
            fig = go.Figure(go.Scatter(x=df[x_col], y=df[y_col], mode="lines+markers"))
        elif plot_type == "scatter":
            fig = go.Figure(go.Scatter(x=df[x_col], y=df[y_col], mode="markers"))
        else:  # bar
            fig = go.Figure(go.Bar(x=df[x_col], y=df[y_col]))

        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, title=f"{plot_type.title()} Chart")
        return fig

    raise ValueError(f"Unsupported plot_type '{plot_type}'.")


def _find_column(df: pd.DataFrame, explicit: Optional[str], aliases: List[str]) -> Optional[str]:
    """Resolve a column either explicitly or via alias list."""

    if explicit and explicit in df.columns:
        return explicit

    for alias in aliases:
        for col in df.columns:
            if col.lower() == alias:
                return col

    return None


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
