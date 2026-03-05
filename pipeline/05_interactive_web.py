"""Step 5: Generate interactive HTML visualizations using Plotly."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    NX, NY, NZ, ORIGIN, SPACING, NODATA,
    LITHOLOGY_MAP, LITHOLOGY_COLORS, MASKS_DIR, MASK_FILES, OUTPUT_DIR
)

WEB_DIR = os.path.join(OUTPUT_DIR, "web")


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))


# Build color scale for plotly (discrete)
FORMATION_COLORS_HEX = {code: rgb_to_hex(LITHOLOGY_COLORS[code]) for code in LITHOLOGY_MAP}


def load_volume():
    return np.load(os.path.join(OUTPUT_DIR, "voxet_lithology.npy"))


def get_coords():
    """Get coordinate arrays for each axis."""
    x = ORIGIN[0] + np.arange(NX) * SPACING[0]
    y = ORIGIN[1] + np.arange(NY) * SPACING[1]
    z = ORIGIN[2] + np.arange(NZ) * SPACING[2]
    return x, y, z


# ─── 1. Interactive Horizontal Slice Explorer ───────────────────────────────

def build_slice_explorer(volume):
    """Interactive map-view slice with depth slider."""
    print("Building horizontal slice explorer...")
    x, y, z = get_coords()

    # Subsample for performance: take every 2nd cell in XY
    step_xy = 2
    x_sub = x[::step_xy]
    y_sub = y[::step_xy]
    vol_sub = volume[::step_xy, ::step_xy, :]

    # Build discrete colorscale
    codes = sorted(LITHOLOGY_MAP.keys())
    n = len(codes)
    colorscale = []
    for i, code in enumerate(codes):
        hex_color = FORMATION_COLORS_HEX[code]
        colorscale.append([i / n, hex_color])
        colorscale.append([(i + 1) / n, hex_color])

    # Pre-select a few depth indices spread through the volume
    k_indices = list(range(0, NZ, 5))

    # Build frames for the slider
    frames = []
    for k in k_indices:
        slc = vol_sub[:, :, k].T.astype(float)
        slc[slc == NODATA] = np.nan
        frames.append(go.Frame(
            data=[go.Heatmap(
                z=slc, x=x_sub, y=y_sub,
                colorscale=colorscale, zmin=0.5, zmax=13.5,
                hovertemplate="E: %{x:.0f}m<br>N: %{y:.0f}m<br>Lithology: %{z}<extra></extra>",
            )],
            name=f"{z[k]:.0f}m"
        ))

    # Initial frame
    k0 = NZ // 3
    slc0 = vol_sub[:, :, k0].T.astype(float)
    slc0[slc0 == NODATA] = np.nan

    fig = go.Figure(
        data=[go.Heatmap(
            z=slc0, x=x_sub, y=y_sub,
            colorscale=colorscale, zmin=0.5, zmax=13.5,
            colorbar=dict(
                title="Lithology",
                tickvals=list(LITHOLOGY_MAP.keys()),
                ticktext=list(LITHOLOGY_MAP.values()),
            ),
            hovertemplate="E: %{x:.0f}m<br>N: %{y:.0f}m<br>Lithology: %{z}<extra></extra>",
        )],
        frames=frames,
    )

    # Slider
    steps = []
    for k in k_indices:
        step = dict(
            method="animate",
            args=[[f"{z[k]:.0f}m"], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
            label=f"{z[k]:.0f}"
        )
        steps.append(step)

    fig.update_layout(
        title=f"Adavale Basin - Interactive Map View (depth slider)",
        xaxis_title="Easting (m)",
        yaxis_title="Northing (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=1000, height=900,
        sliders=[dict(
            active=k0 // 5,
            currentvalue=dict(prefix="Depth: ", suffix=" m"),
            steps=steps,
            pad=dict(t=50),
        )],
    )

    out = os.path.join(WEB_DIR, "slice_explorer.html")
    fig.write_html(out, include_plotlyjs='cdn')
    print(f"  Saved: {out}")


# ─── 2. Interactive W-E Cross Section Explorer ──────────────────────────────

def build_cross_section_ew(volume):
    """Interactive W-E cross section with northing slider."""
    print("Building W-E cross section explorer...")
    x, y, z = get_coords()

    step_x = 2
    step_z = 1
    x_sub = x[::step_x]
    z_sub = z[::step_z]
    vol_sub = volume[::step_x, :, ::step_z]

    codes = sorted(LITHOLOGY_MAP.keys())
    n = len(codes)
    colorscale = []
    for i, code in enumerate(codes):
        hex_color = FORMATION_COLORS_HEX[code]
        colorscale.append([i / n, hex_color])
        colorscale.append([(i + 1) / n, hex_color])

    j_indices = list(range(0, NY, 10))

    frames = []
    for j in j_indices:
        slc = vol_sub[:, j, :].T.astype(float)
        slc[slc == NODATA] = np.nan
        frames.append(go.Frame(
            data=[go.Heatmap(z=slc, x=x_sub, y=z_sub,
                             colorscale=colorscale, zmin=0.5, zmax=13.5)],
            name=f"{y[j]:.0f}m"
        ))

    j0 = NY // 2
    slc0 = vol_sub[:, j0, :].T.astype(float)
    slc0[slc0 == NODATA] = np.nan

    fig = go.Figure(
        data=[go.Heatmap(
            z=slc0, x=x_sub, y=z_sub,
            colorscale=colorscale, zmin=0.5, zmax=13.5,
            colorbar=dict(
                title="Lithology",
                tickvals=list(LITHOLOGY_MAP.keys()),
                ticktext=list(LITHOLOGY_MAP.values()),
            ),
            hovertemplate="E: %{x:.0f}m<br>Z: %{y:.0f}m<br>Lithology: %{z}<extra></extra>",
        )],
        frames=frames,
    )

    steps = []
    for j in j_indices:
        steps.append(dict(
            method="animate",
            args=[[f"{y[j]:.0f}m"], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
            label=f"{y[j]:.0f}"
        ))

    fig.update_layout(
        title="Adavale Basin - W-E Cross Section (slide northing)",
        xaxis_title="Easting (m)",
        yaxis_title="Elevation (m)",
        width=1200, height=600,
        sliders=[dict(
            active=j0 // 10,
            currentvalue=dict(prefix="Northing: ", suffix=" m"),
            steps=steps,
        )],
    )

    out = os.path.join(WEB_DIR, "cross_section_ew.html")
    fig.write_html(out, include_plotlyjs='cdn')
    print(f"  Saved: {out}")


# ─── 3. Interactive S-N Cross Section Explorer ──────────────────────────────

def build_cross_section_sn(volume):
    """Interactive S-N cross section with easting slider."""
    print("Building S-N cross section explorer...")
    x, y, z = get_coords()

    step_y = 2
    step_z = 1
    y_sub = y[::step_y]
    z_sub = z[::step_z]
    vol_sub = volume[:, ::step_y, ::step_z]

    codes = sorted(LITHOLOGY_MAP.keys())
    n = len(codes)
    colorscale = []
    for i, code in enumerate(codes):
        hex_color = FORMATION_COLORS_HEX[code]
        colorscale.append([i / n, hex_color])
        colorscale.append([(i + 1) / n, hex_color])

    i_indices = list(range(0, NX, 10))

    frames = []
    for i in i_indices:
        slc = vol_sub[i, :, :].T.astype(float)
        slc[slc == NODATA] = np.nan
        frames.append(go.Frame(
            data=[go.Heatmap(z=slc, x=y_sub, y=z_sub,
                             colorscale=colorscale, zmin=0.5, zmax=13.5)],
            name=f"{x[i]:.0f}m"
        ))

    i0 = NX // 2
    slc0 = vol_sub[i0, :, :].T.astype(float)
    slc0[slc0 == NODATA] = np.nan

    fig = go.Figure(
        data=[go.Heatmap(
            z=slc0, x=y_sub, y=z_sub,
            colorscale=colorscale, zmin=0.5, zmax=13.5,
            colorbar=dict(
                title="Lithology",
                tickvals=list(LITHOLOGY_MAP.keys()),
                ticktext=list(LITHOLOGY_MAP.values()),
            ),
            hovertemplate="N: %{x:.0f}m<br>Z: %{y:.0f}m<br>Lithology: %{z}<extra></extra>",
        )],
        frames=frames,
    )

    steps = []
    for i in i_indices:
        steps.append(dict(
            method="animate",
            args=[[f"{x[i]:.0f}m"], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
            label=f"{x[i]:.0f}"
        ))

    fig.update_layout(
        title="Adavale Basin - S-N Cross Section (slide easting)",
        xaxis_title="Northing (m)",
        yaxis_title="Elevation (m)",
        width=1200, height=600,
        sliders=[dict(
            active=i0 // 10,
            currentvalue=dict(prefix="Easting: ", suffix=" m"),
            steps=steps,
        )],
    )

    out = os.path.join(WEB_DIR, "cross_section_sn.html")
    fig.write_html(out, include_plotlyjs='cdn')
    print(f"  Saved: {out}")


# ─── 4. Interactive Formation Masks Map ─────────────────────────────────────

def build_formation_masks_map():
    """Interactive map of all formation boundary polygons."""
    print("Building formation masks map...")

    fig = go.Figure()

    for name, filename in MASK_FILES.items():
        df = pd.read_csv(os.path.join(MASKS_DIR, filename))
        code = [k for k, v in LITHOLOGY_MAP.items() if v == name][0]
        hex_color = FORMATION_COLORS_HEX[code]

        # Close polygon
        xs = list(df['X']) + [df['X'].iloc[0]]
        ys = list(df['Y']) + [df['Y'].iloc[0]]

        r, g, b = int(LITHOLOGY_COLORS[code][0]*255), int(LITHOLOGY_COLORS[code][1]*255), int(LITHOLOGY_COLORS[code][2]*255)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            name=name, line=dict(color=hex_color, width=2),
            fill='toself', fillcolor=f"rgba({r},{g},{b},0.25)",
            hovertemplate=f"{name}<br>E: %{{x:.0f}}m<br>N: %{{y:.0f}}m<extra></extra>",
        ))

    fig.update_layout(
        title="Adavale Basin - Formation Distribution Masks (click legend to toggle)",
        xaxis_title="Easting (m)",
        yaxis_title="Northing (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=1000, height=900,
        hovermode='closest',
    )

    out = os.path.join(WEB_DIR, "formation_masks.html")
    fig.write_html(out, include_plotlyjs='cdn')
    print(f"  Saved: {out}")


# ─── 5. Interactive Class Distribution ──────────────────────────────────────

def build_class_distribution(volume):
    """Interactive bar chart of lithology class counts."""
    print("Building class distribution chart...")

    valid = volume[volume != NODATA]
    unique, counts = np.unique(valid, return_counts=True)

    names = [LITHOLOGY_MAP.get(int(u), f"?{int(u)}") for u in unique]
    colors = [FORMATION_COLORS_HEX.get(int(u), "#808080") for u in unique]
    percentages = counts / counts.sum() * 100

    fig = go.Figure(data=[
        go.Bar(
            x=names, y=counts,
            marker_color=colors,
            text=[f"{c/1e6:.1f}M ({p:.1f}%)" for c, p in zip(counts, percentages)],
            textposition='outside',
            hovertemplate="%{x}<br>Count: %{y:,}<br>Percent: %{text}<extra></extra>",
        )
    ])

    fig.update_layout(
        title="Adavale Basin - Lithology Class Distribution",
        xaxis_title="Formation",
        yaxis_title="Voxel Count",
        width=1000, height=500,
    )

    out = os.path.join(WEB_DIR, "class_distribution.html")
    fig.write_html(out, include_plotlyjs='cdn')
    print(f"  Saved: {out}")


# ─── 6. 3D Isosurface / Scatter Visualization ──────────────────────────────

def build_3d_scatter(volume):
    """Interactive 3D view using subsampled scatter plot (works in browser)."""
    print("Building 3D scatter visualization (subsampled)...")
    x, y, z = get_coords()

    # Heavy subsample for 3D browser rendering
    step = 8
    vol_sub = volume[::step, ::step, ::step]
    x_sub = x[::step]
    y_sub = y[::step]
    z_sub = z[::step]

    # Create meshgrid
    xx, yy, zz = np.meshgrid(x_sub, y_sub, z_sub, indexing='ij')

    # Flatten and filter out nodata
    mask = vol_sub != NODATA
    xf = xx[mask]
    yf = yy[mask]
    zf = zz[mask]
    vf = vol_sub[mask]

    print(f"  {mask.sum():,} points after subsampling (step={step})")

    # Build one trace per formation for legend toggling
    fig = go.Figure()

    for code in sorted(LITHOLOGY_MAP.keys()):
        fm = vf == code
        if fm.sum() == 0:
            continue
        name = LITHOLOGY_MAP[code]
        hex_color = FORMATION_COLORS_HEX[code]

        fig.add_trace(go.Scatter3d(
            x=xf[fm], y=yf[fm], z=zf[fm],
            mode='markers',
            name=name,
            marker=dict(size=1.5, color=hex_color, opacity=0.6),
            hovertemplate=f"{name}<br>E: %{{x:.0f}}m<br>N: %{{y:.0f}}m<br>Z: %{{z:.0f}}m<extra></extra>",
        ))

    fig.update_layout(
        title="Adavale Basin - 3D Lithology Model (click legend to toggle formations)",
        scene=dict(
            xaxis_title="Easting (m)",
            yaxis_title="Northing (m)",
            zaxis_title="Elevation (m)",
            aspectmode='manual',
            aspectratio=dict(x=2, y=2.5, z=0.5),
        ),
        width=1200, height=800,
        legend=dict(itemsizing='constant'),
    )

    out = os.path.join(WEB_DIR, "3d_model.html")
    fig.write_html(out, include_plotlyjs='cdn')
    print(f"  Saved: {out}")


# ─── 7. Dashboard index page ───────────────────────────────────────────────

def build_index():
    """Create an index HTML page linking all visualizations."""
    print("Building index page...")

    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Adavale Basin 3D Geological Model</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a; color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 40px 20px; text-align: center;
            border-bottom: 2px solid #e94560;
        }
        .header h1 { font-size: 2.5em; color: #fff; margin-bottom: 10px; }
        .header p { color: #aaa; font-size: 1.1em; }
        .grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px; padding: 30px; max-width: 1400px; margin: 0 auto;
        }
        .card {
            background: #1a1a2e; border-radius: 12px;
            overflow: hidden; transition: transform 0.2s, box-shadow 0.2s;
            border: 1px solid #2a2a4a;
        }
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(233, 69, 96, 0.2);
        }
        .card a { text-decoration: none; color: inherit; display: block; }
        .card-icon { font-size: 3em; padding: 30px 20px 10px; text-align: center; }
        .card-body { padding: 15px 20px 25px; }
        .card-body h3 { color: #e94560; margin-bottom: 8px; font-size: 1.2em; }
        .card-body p { color: #888; font-size: 0.9em; line-height: 1.5; }
        .stats {
            display: grid; grid-template-columns: repeat(4, 1fr);
            gap: 15px; padding: 0 30px 30px; max-width: 1400px; margin: 0 auto;
        }
        .stat {
            background: #1a1a2e; border-radius: 8px; padding: 20px;
            text-align: center; border: 1px solid #2a2a4a;
        }
        .stat .num { font-size: 2em; color: #e94560; font-weight: bold; }
        .stat .label { color: #888; font-size: 0.85em; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Adavale Basin 3D Geological Model</h1>
        <p>Interactive web visualizations &mdash; GDA94 / MGA Zone 55 &mdash; 500m x 500m x 20m resolution</p>
    </div>

    <div class="stats">
        <div class="stat"><div class="num">109M</div><div class="label">Total Voxels</div></div>
        <div class="stat"><div class="num">13</div><div class="label">Formations</div></div>
        <div class="stat"><div class="num">256 x 317 km</div><div class="label">Area (XY)</div></div>
        <div class="stat"><div class="num">6.7 km</div><div class="label">Depth Range</div></div>
    </div>

    <div class="grid">
        <div class="card"><a href="3d_model.html">
            <div class="card-icon">&#127758;</div>
            <div class="card-body">
                <h3>3D Model Viewer</h3>
                <p>Full 3D scatter visualization of all 13 formations. Rotate, zoom, pan.
                   Toggle individual formations on/off via the legend.</p>
            </div>
        </a></div>

        <div class="card"><a href="slice_explorer.html">
            <div class="card-icon">&#128506;</div>
            <div class="card-body">
                <h3>Horizontal Slice Explorer</h3>
                <p>Map view at any depth. Use the slider to scroll through depths
                   from +690m (surface) to -5990m (deep basement).</p>
            </div>
        </a></div>

        <div class="card"><a href="cross_section_ew.html">
            <div class="card-icon">&#8596;</div>
            <div class="card-body">
                <h3>W-E Cross Section</h3>
                <p>West-to-East vertical cross section. Slide the northing coordinate
                   to explore the basin structure from south to north.</p>
            </div>
        </a></div>

        <div class="card"><a href="cross_section_sn.html">
            <div class="card-icon">&#8597;</div>
            <div class="card-body">
                <h3>S-N Cross Section</h3>
                <p>South-to-North vertical cross section. Slide the easting coordinate
                   to explore the basin structure from west to east.</p>
            </div>
        </a></div>

        <div class="card"><a href="formation_masks.html">
            <div class="card-icon">&#128200;</div>
            <div class="card-body">
                <h3>Formation Masks</h3>
                <p>2D map of formation distribution polygons. Click legend entries
                   to toggle individual formations on/off.</p>
            </div>
        </a></div>

        <div class="card"><a href="class_distribution.html">
            <div class="card-icon">&#128202;</div>
            <div class="card-body">
                <h3>Class Distribution</h3>
                <p>Bar chart showing voxel counts per lithology class.
                   Hover for exact counts and percentages.</p>
            </div>
        </a></div>
    </div>
</body>
</html>"""

    out = os.path.join(WEB_DIR, "index.html")
    with open(out, 'w') as f:
        f.write(html)
    print(f"  Saved: {out}")


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(WEB_DIR, exist_ok=True)
    volume = load_volume()
    print(f"Volume loaded: {volume.shape}\n")

    build_class_distribution(volume)
    build_formation_masks_map()
    build_slice_explorer(volume)
    build_cross_section_ew(volume)
    build_cross_section_sn(volume)
    build_3d_scatter(volume)
    build_index()

    print(f"\nAll interactive visualizations saved to: {WEB_DIR}")
    print(f"Open in browser: file://{os.path.join(WEB_DIR, 'index.html')}")
