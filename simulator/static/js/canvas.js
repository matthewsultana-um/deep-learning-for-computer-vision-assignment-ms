/**
 * Canvas drawing helpers shared by LoadedImageDisplay and StepImageView.
 *
 * Colours per box role:
 *   pivot      → amber
 *   candidate  → blue
 *   kept       → green
 *   suppressed → red (faded outline only)
 *   highlighted → brighter outline + label background
 */

const COLOURS = {
  pivot:      '#d97706',
  candidate:  '#2563eb',
  kept:       '#16a34a',
  suppressed: '#dc2626',
  raw:        '#2563eb',
};

/**
 * Draw an image + array of box objects onto a canvas element.
 * opts.thick = true → thicker lines + larger font (used for popup view)
 */
function drawBoxes(canvas, img, boxes, opts = {}) {
  const thick  = opts.thick || false;
  const lw     = thick ? 4 : 2;
  const hlW    = thick ? 6 : 3;
  const fScale = thick ? 0.026 : 0.018;

  const ctx = canvas.getContext('2d');
  canvas.width  = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);

  for (const box of boxes) {
    const [x, y, w, h] = box.bbox;
    const role  = box.role || 'raw';
    const color = COLOURS[role] || COLOURS.raw;
    const hl    = box.highlighted || false;

    ctx.globalAlpha = role === 'suppressed' ? 0.35 : 1.0;
    ctx.strokeStyle = hl ? '#facc15' : color;
    ctx.lineWidth   = hl ? hlW : lw;
    ctx.strokeRect(x, y, w, h);

    // Label — omit confidence for GT boxes (confidence === 1.0 means GT)
    const label = box.confidence === 1.0 && box.role === 'kept'
      ? box.class_name
      : `${box.class_name}: ${box.confidence.toFixed(2)}`;
    const fs     = Math.max(thick ? 13 : 10, Math.round(canvas.width * fScale));
    ctx.font     = `bold ${fs}px sans-serif`;
    const tw     = ctx.measureText(label).width;
    const lh     = fs + 4;
    const lx     = x;
    const ly     = y > lh ? y - lh : y;
    ctx.fillStyle   = hl ? '#facc15' : color;
    ctx.globalAlpha = 0.88;
    ctx.fillRect(lx, ly, tw + 6, lh);
    ctx.globalAlpha = 1.0;
    ctx.fillStyle   = '#fff';
    ctx.fillText(label, lx + 3, ly + lh - 3);
  }
  ctx.globalAlpha = 1.0;
}

/**
 * Load an image from a URL and draw it on a canvas with boxes.
 * Returns a promise that resolves to the loaded HTMLImageElement.
 */
function loadAndDraw(canvas, imageUrl, boxes, opts = {}) {
  return new Promise((resolve, reject) => {
    const img  = new Image();
    img.onload = () => { drawBoxes(canvas, img, boxes, opts); resolve(img); };
    img.onerror = reject;
    img.src = imageUrl;
  });
}
