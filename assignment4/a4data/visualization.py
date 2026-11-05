"""Export online samples as a standalone camera and point-cloud viewer."""

import base64
import html
import io
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from plotly.offline import get_plotlyjs


def _image_url(image, format="PNG"):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format=format)
    mime = "jpeg" if format == "JPEG" else "png"
    return f"data:image/{mime};base64," + base64.b64encode(buffer.getvalue()).decode()


def _scene(sample, max_points, depth_max=None):
    traces, previews = [], []
    stage = sample.get("visualization_stage", "full")
    units = "m" if stage == "unprojection" else "normalized units"
    height, width = sample["rgb"].shape[-2:]
    valid_depths = sample["depth"][sample["depth_valid"]].numpy()
    if depth_max is None:
        depth_max = float(np.percentile(valid_depths, 95)) if valid_depths.size else 1.0
    for view, frame in enumerate(sample["frame_numbers"].tolist()):
        rgb = sample["rgb"][view].permute(1, 2, 0).numpy()
        valid = sample["depth_valid"][view, 0].numpy()
        points = sample["points_world"][view].permute(1, 2, 0).numpy()[valid]
        colors = (rgb[valid].clip(0, 1) * 255).astype(np.uint8)
        finite = np.isfinite(points).all(axis=1)
        points, colors = points[finite], colors[finite]
        if len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
            points, colors = points[indices], colors[indices]
        group = f"view {view} · frame {frame}"
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": group,
                "legendgroup": group,
                "showlegend": False,
                "x": points[:, 0].round(6).tolist(),
                "y": points[:, 1].round(6).tolist(),
                "z": points[:, 2].round(6).tolist(),
                "marker": {
                    "size": 1.5,
                    "color": [f"rgb({r},{g},{b})" for r, g, b in colors],
                },
                "hoverinfo": "skip",
            }
        )
        # A separate legend marker keeps its color independent of the RGB points.
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": group,
                "legendgroup": group,
                "x": [None],
                "y": [None],
                "z": [None],
                "marker": {"size": 6, "color": "crimson" if view == 0 else "royalblue"},
                "hoverinfo": "skip",
            }
        )
        # Frustum corners are backprojected through K, then placed in the same
        # coordinate system as points_world.
        pixels = np.array(
            [
                [0, 0, 1],
                [width - 1, 0, 1],
                [width - 1, height - 1, 1],
                [0, height - 1, 1],
            ],
            dtype=float,
        )
        corners = pixels @ np.linalg.inv(sample["K"][view].numpy()).T
        corners *= 0.08 / corners[:, 2:3]
        pose = sample["T_world_from_camera"][view].numpy()
        corners = corners @ pose[:3, :3].T + pose[:3, 3]
        center = pose[:3, 3]
        lines = []
        for i in range(4):
            lines.extend(
                [
                    center.tolist(),
                    corners[i].tolist(),
                    [None] * 3,
                    corners[i].tolist(),
                    corners[(i + 1) % 4].tolist(),
                    [None] * 3,
                ]
            )
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": f"camera {view}",
                "legendgroup": group,
                "showlegend": False,
                "x": [p[0] for p in lines],
                "y": [p[1] for p in lines],
                "z": [p[2] for p in lines],
                "line": {"width": 4, "color": "crimson" if view == 0 else "royalblue"},
                "hoverinfo": "name",
            }
        )
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers+text",
                "name": f"camera {view}",
                "legendgroup": group,
                "showlegend": False,
                "x": [float(center[0])],
                "y": [float(center[1])],
                "z": [float(center[2])],
                "text": [str(view)],
                "textposition": "top center",
                "marker": {"size": 4, "color": "crimson" if view == 0 else "royalblue"},
                "hovertemplate": (
                    "Camera center<br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<br>"
                    + units
                    + "<extra>camera "
                    + str(view)
                    + "</extra>"
                ),
            }
        )
        depth = sample["depth"][view, 0].numpy()
        depth_rgb = (
            matplotlib.colormaps["viridis"](
                np.clip(depth / max(depth_max, 1e-8), 0, 1)
            )[..., :3]
            * 255
        ).astype(np.uint8)
        depth_rgb[~valid] = 0
        previews.append(
            {
                "frame": frame,
                "rgb": _image_url((rgb.clip(0, 1) * 255).astype(np.uint8), "JPEG"),
                "depth": _image_url(depth_rgb),
            }
        )
    points = sample["points_world"].permute(0, 2, 3, 1).numpy()
    valid = sample["depth_valid"][:, 0].numpy() & np.isfinite(points).all(axis=-1)
    centroid_distance = (
        float(np.linalg.norm(points[valid].mean(axis=0, dtype=np.float64)))
        if valid.any()
        else None
    )
    return {
        "name": sample["sequence_name"],
        "traces": traces,
        "previews": previews,
        "depth_max": depth_max,
        "scale": float(sample["scene_scale"]) if "scene_scale" in sample else None,
        "centroid_distance": centroid_distance,
        "first_camera_center": sample["T_world_from_camera"][0, :3, 3].tolist(),
        "units": units,
        "coordinates": "Original world coordinates"
        if stage == "unprojection"
        else "Canonical first-view coordinates",
        "label": sample.get("visualization_label", "Cameras and points"),
        "reference": sample.get("visualization_reference", False),
    }


def export_html(
    samples, output, max_points=5000, predictions=None, point_predictions=None
):
    """Write a self-contained HTML with scene selection, RGB/depth and 3D orbit."""
    if max_points < 1:
        raise ValueError("max_points must be positive")
    scenes = [_scene(sample, max_points) for sample in samples]
    if not scenes:
        raise ValueError("At least one scene is required")
    for key, variants in (
        ("prediction", predictions),
        ("point_prediction", point_predictions),
    ):
        if variants is None:
            continue
        if len(variants) != len(scenes):
            raise ValueError("Predictions must match the number of scenes")
        for scene, variant in zip(scenes, variants):
            scene[key] = _scene(variant, max_points, scene["depth_max"])
    # Keep the same axes when switching GT/predictions, without aligning predictions.
    for scene in scenes:
        variants = [scene] + [
            scene[k] for k in ("prediction", "point_prediction") if k in scene
        ]
        scene["ranges"] = []
        for axis in ("x", "y", "z"):
            values = [
                v
                for variant in variants
                for trace in variant["traces"]
                for v in trace[axis]
                if v is not None
            ]
            low, high = min(values), max(values)
            padding = max((high - low) * 0.05, 0.1)
            scene["ranges"].append([low - padding, high + padding])
    payload = json.dumps(scenes, allow_nan=False).replace("</", "<\\/")
    options = "".join(
        f'<option value="{i}">{html.escape(scene["name"])}</option>'
        for i, scene in enumerate(scenes)
    )
    page = """<!doctype html><html><head><meta charset="utf-8">
<title>ScanNet++ cameras and points</title>
<style>
body{margin:0;font:15px system-ui;background:#f5f6f8;color:#17212f}
header{padding:16px 24px;background:white;border-bottom:1px solid #ddd}
#reference-note{padding:12px;background:#fff0db;color:#833900;font-weight:700}
select,button{font:inherit;padding:6px;margin:0 8px 8px 0}
main{display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:12px;padding:12px}
#plot{height:78vh;min-height:450px;background:white}aside{background:white;padding:16px}
img{width:100%;display:block}p{line-height:1.5}small{color:#556}
@media(max-width:800px){main{grid-template-columns:1fr}#plot{height:65vh}}
</style><script>__PLOTLY__</script></head><body>
<header><strong>ScanNet++ cameras and points</strong><p>
Scene <select id="scene">__OPTIONS__</select><button id="reset">Reset camera</button>
<select id="geometry" aria-label="Geometry"></select><br><small>Drag to rotate · scroll to zoom · right-drag to pan · click a legend entry to toggle a view.</small></p>
<p id="reference-note" hidden>Provided validation reference — do not submit. Submit visualizations generated from your training geometry implementation.</p></header>
<main><div id="plot"></div><aside>View <select id="view"></select>
<p id="details"></p><strong>RGB</strong><img id="rgb" alt="Selected RGB view">
<p><strong>Depth</strong><img id="depth" alt="Selected depth view"><small id="depth-note"></small></p>
<p id="coordinate-note"></p><p>Red frustum: first sampled camera. Black depth pixels: missing measurements. Predictions use the same depth colors and coordinates as GT.</p></aside></main>
<script>
const scenes=__SCENES__, sceneSelect=document.getElementById('scene'), viewSelect=document.getElementById('view');
const geometrySelect=document.getElementById('geometry');
geometrySelect.add(new Option('Ground truth','gt'));
if(scenes[0].prediction) geometrySelect.add(new Option('Predicted camera + depth','prediction'));
if(scenes[0].point_prediction) geometrySelect.add(new Option('Predicted point head','point_prediction'));
geometrySelect.hidden=geometrySelect.options.length===1;
function selectedGeometry(){const s=scenes[Number(sceneSelect.value)];return geometrySelect.value==='gt'?s:s[geometrySelect.value];}
const defaultCamera={eye:{x:1.4,y:-1.4,z:1.1},up:{x:0,y:-1,z:0}};
function preview(){
 const s=selectedGeometry(), i=Number(viewSelect.value), p=s.previews[i];
 document.getElementById('reference-note').hidden=!s.reference;
 document.getElementById('rgb').src=p.rgb; document.getElementById('depth').src=p.depth;
 document.getElementById('details').textContent=`View ${i} · frame ${p.frame} · ${s.previews.length} sampled views`;
 document.getElementById('depth-note').textContent=`Shared depth colors: 0–${s.depth_max.toFixed(3)} ${s.units} (95th percentile).`;
 document.getElementById('coordinate-note').textContent=`${s.label}. ${s.coordinates}.`+
  ` First camera center (X, Y, Z): (${s.first_camera_center.map(v=>v.toFixed(4)).join(', ')}) ${s.units}.`+
  (s.scale===null?'':` Scene scale: ${s.scale.toFixed(3)} m.`)+
  (s.centroid_distance===null?'':` Valid-point centroid distance: ${s.centroid_distance.toFixed(3)} ${s.units}.`);
}
function showScene(resetView=true){
 const root=scenes[Number(sceneSelect.value)], s=selectedGeometry();
 if(resetView) viewSelect.replaceChildren(...s.previews.map((p,i)=>new Option(`${i}: frame ${p.frame}`,i)));
 preview();
 Plotly.react('plot',s.traces,{margin:{l:0,r:0,t:35,b:0},title:{text:s.name},
  scene:{aspectmode:'data',camera:structuredClone(defaultCamera),xaxis:{title:{text:`X (${s.units})`},range:root.ranges[0]},yaxis:{title:{text:`Y (${s.units})`},range:root.ranges[1]},zaxis:{title:{text:`Z (${s.units})`},range:root.ranges[2]}},
  legend:{groupclick:'togglegroup'},uirevision:s.name},{responsive:true,displaylogo:false});
}
sceneSelect.onchange=()=>showScene();viewSelect.onchange=preview;
geometrySelect.onchange=()=>showScene(false);
document.getElementById('reset').onclick=()=>Plotly.relayout('plot',{'scene.camera':structuredClone(defaultCamera)});
showScene();
</script></body></html>"""
    page = (
        page.replace("__OPTIONS__", options)
        .replace("__SCENES__", payload)
        .replace("__PLOTLY__", get_plotlyjs())
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page)
    return destination
