"""Local, self-hosted model-viewer preview matching the DiffusionGS project page."""
from html import escape
from pathlib import Path
from urllib.parse import quote

from gradio.route_utils import API_PREFIX


ASSET_DIRECTORY = Path(__file__).with_name("mesh_viewer").resolve()


def _file_url(path):
    # Relative URLs also work when Gradio is mounted below a URL prefix.
    return "." + API_PREFIX + "/file=" + quote(str(Path(path).resolve()), safe="/")


def viewer_head():
    script = ASSET_DIRECTORY / "viewer.js"
    source = _file_url(script) + f"?v={script.stat().st_mtime_ns}"
    return f'<script type="module" src="{escape(source, quote=True)}"></script>'


def mesh_preview_html(path=None):
    if path is None:
        return ('<div class="object-mesh-preview"><p class="mesh-status">'
                'Extract a mesh to preview it here.</p></div>')
    path = Path(path)
    source = escape(_file_url(path) + f"?v={path.stat().st_mtime_ns}", quote=True)
    return f'''<div class="object-mesh-preview">
      <model-viewer src="{source}" aria-label="Interactive exported object mesh"
        loading="eager" camera-controls auto-rotate rotation-per-second="16deg"
        camera-orbit="-45deg 80deg 160%" field-of-view="35deg"
        min-camera-orbit="auto 5deg 35%" max-camera-orbit="auto 175deg 300%"
        shadow-intensity="1" shadow-softness="0.85" environment-image="neutral"
        exposure="1" interaction-prompt="none" touch-action="pan-y">
        <div slot="progress-bar" class="mesh-progress"><span></span></div>
      </model-viewer>
      <p class="mesh-status" role="status">Loading mesh…</p>
      <div class="mesh-controls">
        <button type="button" data-mesh-action="rotation" aria-pressed="true">Pause rotation</button>
        <button type="button" data-mesh-action="reset">Reset view</button>
      </div>
      <span class="mesh-hint">Drag to orbit · Scroll or pinch to zoom</span>
    </div>'''
