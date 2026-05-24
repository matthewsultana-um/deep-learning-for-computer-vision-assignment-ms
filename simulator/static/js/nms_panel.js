/**
 * Alpine.js component for NMSPanel.
 *
 * Step -1 = "pre-filter": ALL raw detections shown in candidates, nothing discarded.
 * Step  0 = confidence filter applied: low-conf boxes move to Discarded.
 * Steps 1+ = NMS iterations.
 */
function nmsPanel() {
  // ── Closure vars ─────────────────────────────────────────────────────────
  let _rawBoxes    = [];
  let _boxMap      = {};
  let _imageUrl    = null;
  let _loadedImg   = null;
  let _runTimer    = null;
  let _popupCache  = {};

  return {
    // ── Reactive state ────────────────────────────────────────────────────
    algorithm:      'NMS',
    confThreshold:  0.25,
    iouThreshold:   0.50,
    softNmsMethod:  'Linear',
    sigma:          0.50,
    minScore:       0.001,

    steps:          [],
    stepsSoft:      [],
    currentStep:    -1,   // -1 = pre-filter state
    currentStepSoft:-1,

    currentCandidateBoxes: [],
    suppressedBoxes:       [],
    lowConfidenceBoxes:    [],
    keptBoxes:             [],
    keptBoxesSoft:         [],
    commentLog:            [],
    commentLogSoft:        [],
    highlightedBoxIds:     [],
    nmsWarning:            null,

    popupVisible:  false,
    popupDataUrl:  null,
    popupInfo:     null,

    paramInfo: { badge: 'NMS', name: 'Hard NMS', value: null,
      text: 'Greedy algorithm. The highest-confidence box becomes the pivot; every other box whose overlap (IoU or containment) with the pivot meets the threshold is immediately removed. Fast and simple, but may discard genuine detections that happen to overlap.' },

    // ── Lifecycle ────────────────────────────────────────────────────────
    async init() {
      SimState.on('image_loaded', ({ boxes }) => {
        _rawBoxes   = boxes;
        _boxMap     = {};
        for (const b of boxes) _boxMap[b.id] = b;
        _imageUrl   = SimState.imageUrl;
        _loadedImg  = null;
        _popupCache = {};
        this._resetTimeline();
        this._triggerRun(false);
      });

      SimState.on('recomputing', () => {
        this._resetTimeline();
      });
    },

    _resetTimeline() {
      this.steps = []; this.stepsSoft = [];
      this.currentStep = -1; this.currentStepSoft = -1;
      this.currentCandidateBoxes = []; this.suppressedBoxes = [];
      this.lowConfidenceBoxes    = []; this.keptBoxes = [];
      this.keptBoxesSoft = []; this.commentLog = [];
      this.commentLogSoft = []; this.highlightedBoxIds = [];
      this.nmsWarning = null;
    },

    // Show all raw detections before any filter is applied
    _showAllDetections() {
      this.currentCandidateBoxes = [..._rawBoxes].sort((a, b) => b.confidence - a.confidence);
      this.suppressedBoxes    = [];
      this.lowConfidenceBoxes = [];
      this.keptBoxes          = [];
    },

    async _triggerRun(autoComplete) {
      if (!_rawBoxes.length) return;
      SimState.recomputing();
      this.nmsWarning = null;

      const base = {
        boxes:          _rawBoxes,
        conf_threshold: this.confThreshold,
        iou_threshold:  this.iouThreshold,
        method:         this.softNmsMethod.toLowerCase(),
        sigma:          this.sigma,
        min_score:      this.minScore,
      };

      try {
        if (this.algorithm === 'Compare') {
          const [nmsResult, snmsResult] = await Promise.all([
            API.runNms({ ...base, algorithm: 'nms' }),
            API.runNms({ ...base, algorithm: 'soft_nms' }),
          ]);
          this.steps          = nmsResult.steps;
          this.stepsSoft      = snmsResult.steps;
          this.commentLog     = nmsResult.steps;
          this.commentLogSoft = snmsResult.steps;
          if (nmsResult.warning) this.nmsWarning = nmsResult.warning;
          SimState.setSteps(nmsResult.steps);

          // Show pre-filter state first
          this.currentStep     = -1;
          this.currentStepSoft = -1;
          this._showAllDetections();
          this._drawStep('nms');
          this._drawStep('soft');

          if (autoComplete && nmsResult.steps.length && snmsResult.steps.length) {
            const keptNms  = nmsResult.steps[nmsResult.steps.length-1].kept_box_ids_so_far.map(id=>_boxMap[id]).filter(Boolean);
            const keptSoft = snmsResult.steps[snmsResult.steps.length-1].kept_box_ids_so_far.map(id=>_boxMap[id]).filter(Boolean);
            SimState.setNmsComplete(keptNms, keptSoft);
          }
        } else {
          const result = await API.runNms({
            ...base,
            algorithm: this.algorithm === 'NMS' ? 'nms' : 'soft_nms',
          });
          this.steps      = result.steps;
          this.commentLog = result.steps;
          if (result.warning) this.nmsWarning = result.warning;
          SimState.setSteps(result.steps);

          // Show pre-filter state first — user must press ▶ to apply filter
          this.currentStep = -1;
          this._showAllDetections();
          this._drawStep('nms');

          if (autoComplete && result.steps.length) {
            this._signalComplete(result.steps);
          }
        }
      } catch (err) {
        console.error('NMS run failed:', err);
      }
    },

    _signalComplete(steps) {
      const lastStep = steps[steps.length - 1];
      const kept = lastStep.kept_box_ids_so_far.map(id => _boxMap[id]).filter(Boolean);
      SimState.setNmsComplete(kept);
    },

    setAlgorithm(algo) {
      this.algorithm = algo;
      this.paramInfo = algo === 'NMS'
        ? { badge: 'NMS', name: 'Hard NMS', value: null,
            text: 'Greedy algorithm. The highest-confidence box becomes the pivot; every other box whose overlap (IoU or containment) with the pivot meets the threshold is immediately removed. Fast and simple, but may discard genuine detections that happen to overlap.' }
        : { badge: 'SNMS', name: 'Soft-NMS', value: null,
            text: 'Instead of hard removal, Soft-NMS decays the confidence of overlapping boxes proportionally to their overlap. A box is only discarded if its score falls below the minimum-confidence threshold after decay — so partially-overlapping true detections often survive.' };
      if (_rawBoxes.length) this._triggerRun(true);
    },

    setConfThreshold(v) {
      this.confThreshold = v;
      this.paramInfo = {
        badge: 'CONF', name: this.algorithm === 'Soft-NMS' ? 'Initial Confidence' : 'Confidence',
        value: v.toFixed(2),
        text: 'Pre-filter applied before the algorithm runs. Detections whose raw YOLO confidence is below this value are moved to Discarded immediately and never compared. Raise it to reduce noise; lower it to keep more marginal detections.'
      };
      this._debounceRun();
    },

    setIouThreshold(v) {
      this.iouThreshold = v;
      this.paramInfo = {
        badge: 'IoU', name: 'IoU Threshold', value: v.toFixed(2),
        text: 'Suppression gate. When the overlap score between the pivot and a candidate reaches this value, the candidate is suppressed (NMS) or begins decaying (Soft-NMS Linear). Lower → more aggressive; fewer boxes kept. Higher → more lenient; more boxes kept.'
      };
      this._debounceRun();
    },

    setSoftNmsMethod(m) {
      this.softNmsMethod = m;
      this.paramInfo = m === 'Linear'
        ? { badge: 'LIN', name: 'Linear Decay', value: null,
            text: 'Applies a penalty only when IoU exceeds the threshold: new_conf = conf × (1 − IoU). High overlap yields a steep confidence cut; boxes below the threshold are untouched. Simple and interpretable.' }
        : { badge: 'GAUSS', name: 'Gaussian Decay', value: null,
            text: 'Decay is applied to every same-class box using: new_conf = conf × exp(−IoU² / σ). Higher overlap = steeper penalty. Unlike Linear, the IoU threshold slider does not gate when decay kicks in — σ controls the curve instead.' };
      if (_rawBoxes.length) this._triggerRun(true);
    },

    setSigma(v) {
      this.sigma = v;
      this.paramInfo = {
        badge: 'σ', name: 'Sigma', value: v.toFixed(2),
        text: 'Gaussian decay bandwidth. Small σ concentrates the penalty on high-IoU boxes (steep curve); large σ spreads decay broadly across all candidates even with modest overlap. Only active in Gaussian mode.'
      };
      this._debounceRun();
    },

    setMinScore(v) {
      this.minScore = v;
      this.paramInfo = {
        badge: 'MIN', name: 'Min Conf After Decay', value: v.toFixed(3),
        text: 'Soft-NMS elimination floor. After each round of confidence decay, any box whose score has fallen below this threshold is moved to Discarded. Raise it to discard more aggressively; lower it to keep boxes even after heavy decay.'
      };
      this._debounceRun();
    },

    _debounceRun() {
      clearTimeout(_runTimer);
      _runTimer = setTimeout(() => { if (_rawBoxes.length) this._triggerRun(true); }, 300);
    },

    // Navigate; step -1 is the pre-filter view
    jumpToStep(idx, highlightReferenced = false) {
      if (!this.steps.length) return;

      if (idx === -1) {
        this.currentStep = -1;
        this.highlightedBoxIds = [];
        this._showAllDetections();
        this._clearCanvas('nms');
        return;
      }

      if (idx < 0 || idx >= this.steps.length) return;
      this.currentStep = idx;
      this.highlightedBoxIds = highlightReferenced
        ? (this.steps[idx].referenced_box_ids || []) : [];
      this._rebuildLists('nms');
      this._drawStep('nms');
      this._scrollLog(idx, 'comment-log');
    },

    jumpToStepSoft(idx, highlightReferenced = false) {
      if (!this.stepsSoft.length || idx < 0 || idx >= this.stepsSoft.length) return;
      this.currentStepSoft = idx;
      if (highlightReferenced) this.highlightedBoxIds = this.stepsSoft[idx].referenced_box_ids || [];
      this._rebuildLists('soft');
      this._drawStep('soft');
      this._scrollLog(idx, 'comment-log-nms');
    },

    _scrollLog(idx, id) {
      requestAnimationFrame(() => {
        const el = document.getElementById(id);
        if (el && el.children[idx]) el.children[idx].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      });
    },

    stepForward() {
      if (this.currentStep === -1) {
        // First ▶ applies the confidence filter (step 0)
        this.jumpToStep(0);
        if (this.algorithm === 'Compare') this.jumpToStepSoft(0);
        return;
      }
      if (this.currentStep < this.steps.length - 1) this.jumpToStep(this.currentStep + 1);
      if (this.algorithm === 'Compare' && this.currentStepSoft < this.stepsSoft.length - 1)
        this.jumpToStepSoft(this.currentStepSoft + 1);
      this._checkComplete();
    },

    runAll() {
      this.jumpToStep(this.steps.length - 1);
      if (this.algorithm === 'Compare') this.jumpToStepSoft(this.stepsSoft.length - 1);
      this._checkComplete();
    },

    _checkComplete() {
      const nmsAtEnd  = this.currentStep >= this.steps.length - 1;
      const softAtEnd = this.algorithm !== 'Compare' || this.currentStepSoft >= this.stepsSoft.length - 1;
      if (nmsAtEnd && softAtEnd && this.steps.length) this._signalComplete(this.steps);
    },

    _rebuildLists(which) {
      const steps = which === 'nms' ? this.steps : this.stepsSoft;
      const idx   = which === 'nms' ? this.currentStep : this.currentStepSoft;
      const step  = steps[idx];
      if (!step) return;

      const suppressed = new Set();
      for (let i = 0; i <= idx; i++)
        (steps[i].suppressed_box_ids || []).forEach(id => suppressed.add(id));
      const kept = new Set(step.kept_box_ids_so_far);

      // Soft-NMS decays confidence in-place; accumulate current confidence per box
      // across all steps so far so the display reflects the decayed value, not the
      // original YOLO score.
      const isSoft = this.algorithm === 'Soft-NMS' || which === 'soft';
      const decayedConf = {};
      if (isSoft) {
        for (let i = 0; i <= idx; i++) {
          for (const u of (steps[i].weight_updates || [])) {
            decayedConf[u.box_id] = u.new_conf;
          }
        }
      }
      const withDecay = (box) =>
        isSoft && decayedConf[box.id] !== undefined
          ? { ...box, confidence: decayedConf[box.id] }
          : box;

      const iouScores = step.iou_scores || {};
      const pivotClassId = step.pivot_box_id != null ? (_boxMap[step.pivot_box_id]?.class_id ?? null) : null;
      const candidates = _rawBoxes
        .filter(b => b.confidence >= this.confThreshold)
        .filter(b => !suppressed.has(b.id) && !kept.has(b.id))
        .map(b => ({ ...withDecay(b), iou: iouScores[String(b.id)] ?? null }))
        .sort((a, b) => {
          if (pivotClassId === null) return 0;
          const aMatch = a.class_id === pivotClassId ? 0 : 1;
          const bMatch = b.class_id === pivotClassId ? 0 : 1;
          return aMatch - bMatch;
        });

      if (which === 'nms') {
        this.currentCandidateBoxes = candidates;
        this.suppressedBoxes    = [...suppressed].map(id => _boxMap[id]).filter(Boolean).map(withDecay);
        this.lowConfidenceBoxes = _rawBoxes.filter(b => b.confidence < this.confThreshold);
        this.keptBoxes          = [...kept].map(id => _boxMap[id]).filter(Boolean);
      } else {
        this.keptBoxesSoft = [...kept].map(id => _boxMap[id]).filter(Boolean);
      }
    },

    highlightBox(id) {
      this.highlightedBoxIds = this.highlightedBoxIds.includes(id)
        ? this.highlightedBoxIds.filter(x => x !== id)
        : [...this.highlightedBoxIds, id];
      this._drawStep('nms');
      if (this.algorithm === 'Compare') this._drawStep('soft');
    },

    _clearCanvas(which) {
      const canvasId = this.algorithm === 'Compare'
        ? (which === 'nms' ? 'step-canvas-nms' : 'step-canvas-snms')
        : 'step-canvas';
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;
      // Draw all raw boxes (pre-filter view) with thick dim styling
      if (_loadedImg) {
        const allBoxes = _rawBoxes.map(b => ({ ...b, role: 'candidate', highlighted: false }));
        drawBoxes(canvas, _loadedImg, allBoxes);
      }
    },

    async _drawStep(which) {
      const steps = which === 'nms' ? this.steps : this.stepsSoft;
      const idx   = which === 'nms' ? this.currentStep : this.currentStepSoft;
      if (!_imageUrl) return;

      // Pre-filter state: draw all raw boxes dimmed
      if (idx === -1) {
        const canvasId = this.algorithm === 'Compare'
          ? (which === 'nms' ? 'step-canvas-nms' : 'step-canvas-snms')
          : 'step-canvas';
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const allBoxes = _rawBoxes.map(b => ({ ...b, role: 'candidate', highlighted: false }));
        if (!_loadedImg) {
          _loadedImg = await loadAndDraw(canvas, _imageUrl, allBoxes).catch(() => null);
        } else {
          drawBoxes(canvas, _loadedImg, allBoxes);
        }
        return;
      }

      if (!steps.length || idx < 0) return;
      const step = steps[idx];
      const canvasId = this.algorithm === 'Compare'
        ? (which === 'nms' ? 'step-canvas-nms' : 'step-canvas-snms')
        : 'step-canvas';
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;

      const suppressed = new Set();
      for (let i = 0; i <= idx; i++)
        (steps[i].suppressed_box_ids || []).forEach(id => suppressed.add(id));
      const kept = new Set(step.kept_box_ids_so_far);

      const isSoft = this.algorithm === 'Soft-NMS' || which === 'soft';
      const decayedConf = {};
      if (isSoft) {
        for (let i = 0; i <= idx; i++)
          for (const u of (steps[i].weight_updates || []))
            decayedConf[u.box_id] = u.new_conf;
      }

      const boxes = _rawBoxes
        .filter(b => b.confidence >= this.confThreshold)
        .filter(b => !suppressed.has(b.id))
        .map(b => ({
          ...b,
          confidence: isSoft && decayedConf[b.id] !== undefined ? decayedConf[b.id] : b.confidence,
          role: b.id === step.pivot_box_id ? 'pivot'
              : kept.has(b.id)              ? 'kept'
              :                               'candidate',
          highlighted: this.highlightedBoxIds.includes(b.id),
        }));

      if (!_loadedImg) {
        _loadedImg = await loadAndDraw(canvas, _imageUrl, boxes).catch(() => null);
      } else {
        drawBoxes(canvas, _loadedImg, boxes);
      }
    },

    // ── Popup (hover preview) ─────────────────────────────────────────────
    async openPopup(idx) {
      if (!this.steps.length || !_imageUrl || idx < 0) return;
      const step = this.steps[idx];
      if (!step) return;

      const suppressed = new Set();
      for (let i = 0; i <= idx; i++)
        (this.steps[i].suppressed_box_ids || []).forEach(id => suppressed.add(id));
      const kept = new Set(step.kept_box_ids_so_far);

      const boxes = _rawBoxes
        .filter(b => b.confidence >= this.confThreshold)
        .filter(b => !suppressed.has(b.id))
        .map(b => ({
          ...b,
          role: b.id === step.pivot_box_id ? 'pivot'
              : kept.has(b.id)              ? 'kept'
              :                               'candidate',
          highlighted: false,
        }));

      this.popupInfo = {
        stepIndex:   step.step_index,
        pivotId:     step.pivot_box_id,
        nKept:       kept.size,
        nSuppressed: suppressed.size,
        nActive:     boxes.filter(b => b.role === 'candidate').length,
        pivot:       step.pivot_box_id !== null ? _boxMap[step.pivot_box_id] : null,
      };
      this.popupVisible = true;

      if (_popupCache[idx]) { this.popupDataUrl = _popupCache[idx]; return; }

      const offscreen = document.createElement('canvas');
      if (_loadedImg) {
        drawBoxes(offscreen, _loadedImg, boxes, { thick: true });
      } else {
        await loadAndDraw(offscreen, _imageUrl, boxes, { thick: true }).catch(() => null);
      }
      const dataUrl = offscreen.toDataURL('image/jpeg', 0.88);
      _popupCache[idx]  = dataUrl;
      this.popupDataUrl = dataUrl;
    },

    closePopup() { this.popupVisible = false; },

    // Returns the class_name of the current pivot, or null when no pivot is active.
    pivotClassName() {
      const step = this.steps[this.currentStep];
      if (!step || step.pivot_box_id == null) return null;
      return (_boxMap[step.pivot_box_id] || {}).class_name ?? null;
    },

    // Returns the full box object for the current pivot, with decayed confidence applied.
    pivotBox() {
      const step = this.steps[this.currentStep];
      if (!step || step.pivot_box_id == null) return null;
      const raw = _boxMap[step.pivot_box_id];
      if (!raw) return null;
      const isSoft = this.algorithm === 'Soft-NMS';
      if (!isSoft) return raw;
      // Accumulate decayed confidence up to (but not including) this step —
      // the pivot's own confidence at the moment it was selected.
      let conf = raw.confidence;
      for (let i = 0; i < this.currentStep; i++)
        for (const u of (this.steps[i].weight_updates || []))
          if (u.box_id === raw.id) conf = u.new_conf;
      return { ...raw, confidence: conf };
    },

    // Returns the weight_update entry for boxId at the current step, or null.
    weightUpdate(boxId) {
      const step = this.commentLog[this.currentStep];
      if (!step || !step.weight_updates) return null;
      return step.weight_updates.find(u => u.box_id === boxId) || null;
    },
  };
}
