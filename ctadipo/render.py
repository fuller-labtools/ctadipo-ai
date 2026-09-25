r"""Build meshes for the browser. No OpenGL, no display, no GPU on the server.

The paper figures are rendered with pyvista, which needs a graphics context; shinyapps.io and
Connect Cloud have neither a GPU nor a display, so that path cannot ship. Here the server does only
the part that is pure arithmetic -- marching cubes over a binary mask -- and hands vertices and
faces to the browser, which draws them with WebGL. The user gets to rotate the animal, which a
server-rendered PNG could never offer.

THESE MESHES ARE FOR LOOKING AT, NEVER FOR MEASURING. Every volume CTAdipo reports is counted on the
full-resolution voxel mask. The mesh is downsampled, closed and smoothed to be legible on screen,
and each of those steps changes its enclosed volume. Keeping the two apart is deliberate: the
display cleanup exists precisely so it can be aggressive without touching a single reported number.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

ISO = 0.15


def _prepare(mask, step, close_mm):
    """Downsample, then close -- WITH ENOUGH PADDING, which is not optional.

    Marching cubes over a mask touching the array boundary leaves the surface open there, so the
    animal renders as a shell with a hole at the crop face. binary_closing additionally erodes with
    border_value=0, eating r-1 voxels back from every face and re-opening exactly the hole the pad
    exists to prevent: measured on a mask against the crop face, r=2 loses a whole plane and r=5
    loses 71% of it. So the pad is r+1, not 1, and the mesh offset below matches it.
    """
    m = mask[::step, ::step, ::step] if step > 1 else mask
    r = max(1, int(round(close_mm / (ISO * step)))) if close_mm > 0 else 0
    pad = r + 1
    m = np.pad(m.astype(bool), pad)
    if r > 0:
        m = ndi.binary_closing(m, ndi.generate_binary_structure(3, 3), iterations=r)
    return m, pad


def mesh(mask, step=2, close_mm=0.0, smooth=1.0, min_voxels=200):
    """(vertices in mm, faces) for one binary mask, or None if there is nothing worth drawing.

    `smooth` blurs the occupancy field before iso-surfacing rather than smoothing the mesh after it.
    That keeps the surface watertight -- vertex-averaging a marching-cubes mesh pulls thin structures
    apart, and adipose depots are full of thin structures.
    """
    from skimage.measure import marching_cubes
    if mask is None or mask.sum() < min_voxels:
        return None
    m, pad = _prepare(mask, step, close_mm)
    f = m.astype(np.float32)
    if smooth > 0:
        f = ndi.gaussian_filter(f, smooth)
    if not (f.min() < 0.5 < f.max()):
        return None
    v, fc, _, _ = marching_cubes(f, level=0.5, spacing=(ISO * step,) * 3)
    v -= ISO * step * pad                               # undo the pad, back into volume coordinates
    return v.astype(np.float32), fc.astype(np.int32)


def decimate(v, f, target_faces=30000):
    """Reduce a mesh to something a browser can rotate smoothly. Returns (v, f, note).

    THE KEYWORD IS NOT OPTIONAL. trimesh's first positional parameter is `percent`, not
    `face_count`, so a positional call asks for 20000 PERCENT and the reduction silently does
    nothing. The method also needs the `fast_simplification` backend, which trimesh declares only
    under an extra -- without it the import raises. Both failures used to land in a bare
    `except: return v, f`, so the app shipped the FULL marching-cubes mesh, hundreds of thousands of
    faces per depot, serialised into one websocket message. That kills the browser tab and logs
    nothing. The failure is now returned as a note so it can be seen.
    """
    if f is None or len(f) <= target_faces:
        return v, f, ""
    try:
        import trimesh
        t = trimesh.Trimesh(vertices=v, faces=f, process=False)
        t = t.simplify_quadric_decimation(face_count=int(target_faces))
        return (np.asarray(t.vertices, np.float32), np.asarray(t.faces, np.int32), "")
    except Exception as e:
        # Fall back by throwing away resolution in a way that always works, rather than by
        # returning a mesh fifty times the target size.
        keep = max(1, len(f) // target_faces)
        return v, f[::keep], ("3-D view simplified crudely: mesh decimation unavailable (%s). "
                              "Install fast-simplification for a cleaner render." % type(e).__name__)


def largest_component(mask, keep=1):
    """Drop detached specks so the render is the animal and not the bedding around it.

    Display-only. It is applied to the mesh input, never to anything that is counted.
    """
    lbl, n = ndi.label(mask)
    if n <= keep:
        return mask
    sz = np.bincount(lbl.ravel())
    sz[0] = 0
    keep_ids = np.argsort(sz)[::-1][:keep]
    return np.isin(lbl, keep_ids)


def plotly_mesh(v, f, colour, name, opacity=1.0, hover=True):
    """One plotly Mesh3d trace. z is the craniocaudal axis, so it is plotted as the vertical."""
    import plotly.graph_objects as go
    return go.Mesh3d(x=v[:, 2], y=v[:, 1], z=v[:, 0],
                     i=f[:, 0], j=f[:, 1], k=f[:, 2],
                     color=colour, opacity=opacity, name=name, showlegend=True,
                     lighting=dict(ambient=0.55, diffuse=0.85, specular=0.12, roughness=0.6),
                     lightposition=dict(x=200, y=200, z=400),
                     hoverinfo="name" if hover else "skip", flatshading=False)


def scene(traces, title=""):
    """A figure with equal axes, because an unequal one silently distorts the animal's shape."""
    import plotly.graph_objects as go
    fig = go.Figure(data=[t for t in traces if t is not None])
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=30 if title else 0, b=0),
                      scene=dict(aspectmode="data", xaxis_title="", yaxis_title="", zaxis_title="",
                                 xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                                 yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                                 zaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                                 camera=dict(eye=dict(x=1.6, y=1.4, z=0.75))),
                      legend=dict(orientation="h", y=-0.02), paper_bgcolor="rgba(0,0,0,0)")
    return fig
