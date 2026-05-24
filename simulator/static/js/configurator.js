/**
 * Alpine.js component for ConfiguratorPanel.
 *
 * Internal/large state (_loadedImg, _progressTimer) lives in closure
 * variables so Alpine never tries to make them reactive.
 */
function configurator() {
  // ── Closure vars ─────────────────────────────────────────────────────────
  let _progressTimer = null;

  return {
    collapsed:       false,
    activeSource:    null,
    previewTiles:    [],
    selectedImageId: null,
    loadedImageSrc:  null,
    detecting:       false,
    detectingMsg:    '',
    detectionError:  null,
    detectProgress:  0,
    uploadImage:     null,
    uploadLabel:     null,
    uploadError:     null,
    uploading:       false,
    gtBoxes:         [],
    gtMatchingData:  [],   // populated after AP is computed
    gtMatchById:     {},   // gt_id → match entry for O(1) template lookup

    async init() {
      await this.setSource('sample');

      SimState.on('state_change', ({ state }) => {
        if (state === 'idle') {
          this.loadedImageSrc  = null;
          this.selectedImageId = null;
        }
      });

      SimState.on('gt_loaded', ({ boxes }) => {
        this.gtBoxes       = boxes;
        this.gtMatchingData = [];
        this.gtMatchById    = {};
        this._drawGT();
      });

      SimState.on('ap_complete', ({ gtMatching }) => {
        this.gtMatchingData = gtMatching || [];
        this.gtMatchById    = Object.fromEntries(this.gtMatchingData.map(m => [m.gt_id, m]));
      });

      SimState.on('recomputing', () => {
        this.gtMatchingData = [];
        this.gtMatchById    = {};
      });
    },

    // Returns the gtMatching entry for a given GT box id (reactive-safe).
    getGtMatch(gtId) {
      return this.gtMatchingData.find(m => m.gt_id === gtId) || null;
    },

    async setSource(src) {
      this.activeSource = src;
      this.previewTiles = [];
      if (src === 'sample') {
        this.previewTiles = await API.getSamples().catch(() => []);
      } else if (src === 'examples') {
        this.previewTiles = await API.getExamples().catch(() => []);
      }
    },

    async _drawGT() {
      const canvas = document.getElementById('gt-image-canvas');
      const url    = this.loadedImageSrc;
      if (!canvas || !url || !this.gtBoxes.length) return;

      // Draw GT boxes in green with class labels
      const gtForDraw = this.gtBoxes.map(b => ({
        ...b,
        confidence: 1.0,   // GT boxes don't have confidence — draw at 1.0
        role: 'kept',
      }));
      await loadAndDraw(canvas, url, gtForDraw).catch(() => null);
    },

    async selectImage(tile) {
      this.selectedImageId = tile.id;
      this.loadedImageSrc  = tile.url;
      this.gtBoxes         = [];
      SimState.setImage(tile.id, tile.url);
      await this._runDetection(tile.id, tile.url);
    },

    _startProgress() {
      this.detectProgress = 0;
      clearInterval(_progressTimer);
      let elapsed = 0;
      _progressTimer = setInterval(() => {
        elapsed += 100;
        if (elapsed <= 2000) {
          this.detectProgress = Math.round(75 * elapsed / 2000);
        } else {
          const extra = elapsed - 2000;
          this.detectProgress = Math.round(75 + 20 * (1 - Math.exp(-extra / 8000)));
        }
      }, 100);
    },

    _stopProgress() {
      clearInterval(_progressTimer);
      this.detectProgress = 100;
      setTimeout(() => { this.detectProgress = 0; }, 400);
    },

    async _runDetection(imageId, imageUrl) {
      this.detecting      = true;
      this.detectionError = null;
      this.detectingMsg   = 'Running YOLO…';
      this._startProgress();
      try {
        SimState.recomputing();
        const result = await API.detect({ image_id: imageId });
        this._stopProgress();
        this.detectingMsg = 'Drawing boxes…';
        // Pass boxes via SimState — NOT stored on Alpine reactive scope
        SimState.setRawBoxes(result.boxes);
        // Load ground-truth labels (silently — no error if missing)
        API.getGroundTruth(imageId)
          .then(gt => SimState.setGtBoxes(gt.boxes))
          .catch(() => SimState.setGtBoxes([]));
        const canvas = document.getElementById('loaded-image-canvas');
        if (canvas) {
          await loadAndDraw(canvas, imageUrl,
            result.boxes.map(b => ({ ...b, role: 'raw' })));
        }
      } catch (err) {
        this._stopProgress();
        this.detectionError = `Detection failed: ${err.message}`;
      } finally {
        this.detecting    = false;
        this.detectingMsg = '';
      }
    },

    async submitUpload() {
      this.uploadError = null;
      if (!this.uploadImage || !this.uploadLabel) {
        this.uploadError = 'Both an image and a label file are required.';
        return;
      }
      this.uploading = true;
      try {
        const form = new FormData();
        form.append('image', this.uploadImage);
        form.append('label', this.uploadLabel);
        const result = await API.upload(form);
        await this.setSource('sample');
        const tile = this.previewTiles.find(t => t.id === result.id)
          || { id: result.id, url: result.url, filename: result.filename };
        await this.selectImage(tile);
        this.activeSource = 'sample';
      } catch (err) {
        this.uploadError = err.message;
      } finally {
        this.uploading = false;
      }
    },
  };
}
