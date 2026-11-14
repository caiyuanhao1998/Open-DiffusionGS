import './model-viewer.min.js';

const initialOrbit = '-45deg 80deg 160%';
const attached = new WeakSet();
function raiseMesh(viewer) {
  const offset = viewer.clientWidth * 0.01;
  viewer.style.transform = `translateY(${-offset}px)`;
  viewer.querySelector('.mesh-progress').style.top = `${offset}px`;
}
const previewResize = new ResizeObserver(() => {
  document.querySelectorAll('.object-mesh-preview model-viewer').forEach(raiseMesh);
});
let observedPreview = null;
function attachViewers() {
  const preview = document.getElementById('object-mesh-preview');
  if (preview !== observedPreview) {
    previewResize.disconnect();
    if (preview) previewResize.observe(preview);
    observedPreview = preview;
  }
  document.querySelectorAll('.object-mesh-preview model-viewer').forEach(viewer => {
    if (attached.has(viewer)) return;
    attached.add(viewer);
    viewer.loading = 'eager';
    const container = viewer.closest('.object-mesh-preview');
    const status = container.querySelector('[role="status"]');
    const progress = container.querySelector('.mesh-progress');
    raiseMesh(viewer);
    const loaded = () => { status.hidden = true; progress.hidden = true; };
    viewer.addEventListener('load', loaded);
    viewer.addEventListener('progress', ({detail}) => {
      progress.firstElementChild.style.width = `${Math.round(detail.totalProgress * 100)}%`;
    });
    viewer.addEventListener('error', () => {
      status.hidden = false;
      status.textContent = 'This mesh could not be loaded. Extract it again or download the GLB below.';
    });
    if (viewer.loaded) loaded();
  });
}
document.addEventListener('click', event => {
  const button = event.target.closest('[data-mesh-action]');
  if (!button) return;
  const viewer = button.closest('.object-mesh-preview').querySelector('model-viewer');
  if (button.dataset.meshAction === 'rotation') {
    viewer.autoRotate = !viewer.autoRotate;
    button.setAttribute('aria-pressed', String(viewer.autoRotate));
    button.textContent = viewer.autoRotate ? 'Pause rotation' : 'Resume rotation';
  } else {
    viewer.cameraOrbit = initialOrbit;
    viewer.cameraTarget = 'auto auto auto';
    viewer.jumpCameraToGoal();
  }
});
new MutationObserver(attachViewers).observe(document.body, {childList: true, subtree: true});
attachViewers();
