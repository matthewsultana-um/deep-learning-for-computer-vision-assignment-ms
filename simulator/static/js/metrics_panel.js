/**
 * Alpine.js component for APmAPPanel.
 *
 * _prChart and other large/internal objects MUST be closure variables.
 * Putting a Chart.js instance inside Alpine's data object caused Alpine
 * to recursively proxy Chart.js's circular-reference-heavy internals,
 * overflowing the call stack ("Maximum call stack size exceeded").
 */
function apPanel() {
  // ── Closure vars — NOT reactive ──────────────────────────────────────────
  let _prChart       = null;
  let _allPrData     = {};
  let _keptBoxes     = null;
  let _keptBoxesSoft = null;

  return {
    // ── Reactive state ────────────────────────────────────────────────────
    nmsComplete:          false,
    metricFocus:          'AP',
    selectedClass:        '__all__',
    classes:              [],
    interpolationPoints:  11,
    iouMode:              'single',
    apIouValue:           0.50,
    apValue:              null,
    mapValue:             null,
    interimTable:         [],
    gtBoxes:              [],
    keptBoxesList:           [],   // reactive copy of kept boxes — set on nms_complete
    keptBoxesListSoft:       [],
    gtMatching:              [],
    falsePositives:          [],
    predictionsMatching:     [],
    matchByPredId:           {},   // pred_id → match entry, rebuilt on each AP compute
    falseNegatives:          [],
    refreshing:           false,
    // Compare mode
    compareMode:          false,
    compareView:          'nms',   // 'nms' | 'soft_nms'
    classesSoft:          [],
    apValueSoft:          null,
    mapValueSoft:         null,
    gtMatchingSoft:          [],
    falsePositivesSoft:      [],
    predictionsMatchingSoft: [],
    matchByPredIdSoft:       {},
    falseNegativesSoft:      [],
    interimTableSoft:     [],

    async init() {
      SimState.on('nms_complete', ({ keptBoxes, keptBoxesSoft, compareMode }) => {
        this.nmsComplete       = true;
        this.compareMode       = !!compareMode;
        this.compareView       = 'nms';
        _keptBoxes             = keptBoxes;
        _keptBoxesSoft         = keptBoxesSoft || null;
        // Expose kept boxes reactively so Dataset Labels renders immediately
        this.keptBoxesList     = [...(keptBoxes || [])];
        this.keptBoxesListSoft = [...(keptBoxesSoft || [])];
        this._computeAP();
        if (compareMode && keptBoxesSoft) {
          this._computeAPSoft(keptBoxesSoft);
        }
      });

      SimState.on('recomputing', () => {
        this.nmsComplete             = false;
        this.compareMode             = false;
        this.apValue                 = null;
        this.mapValue                = null;
        this.interimTable            = [];
        this.keptBoxesList           = [];
        this.keptBoxesListSoft       = [];
        this.gtMatching              = [];
        this.falsePositives          = [];
        this.predictionsMatching     = [];
        this.matchByPredId           = {};
        this.falseNegatives          = [];
        this.classesSoft             = [];
        this.apValueSoft             = null;
        this.mapValueSoft            = null;
        this.gtMatchingSoft          = [];
        this.falsePositivesSoft      = [];
        this.predictionsMatchingSoft = [];
        this.matchByPredIdSoft       = {};
        this.falseNegativesSoft      = [];
        this.interimTableSoft        = [];
        if (_prChart) {
          _prChart.data.datasets[0].data = [];
          _prChart.data.datasets[1].data = [];
          _prChart.update();
        }
      });

      SimState.on('image_loaded', () => {
        _keptBoxes           = null;
        this.gtBoxes         = [];
        this.keptBoxesList   = [];
        this.keptBoxesListSoft = [];
        this.predictionsMatching = [];
        this.matchByPredId   = {};
        this.falseNegatives  = [];
      });

      SimState.on('gt_loaded', ({ boxes }) => {
        this.gtBoxes = boxes;
      });

      this._initChart();
    },

    _initChart() {
      const canvas = document.getElementById('pr-curve-canvas');
      if (!canvas) return;
      _prChart = new Chart(canvas, {
        type: 'line',
        data: {
          datasets: [
            {
              label: 'Raw PR',
              data: [],
              borderColor: 'rgba(37,99,235,0.4)',
              backgroundColor: 'transparent',
              fill: false,
              tension: 0,
              borderWidth: 1.5,
              borderDash: [5, 3],
              pointRadius: 2,
              pointBackgroundColor: 'rgba(37,99,235,0.5)',
              order: 2,
            },
            {
              label: 'Interpolated',
              data: [],
              borderColor: '#2563eb',
              backgroundColor: 'rgba(37,99,235,0.08)',
              fill: true,
              stepped: 'before',
              borderWidth: 2,
              pointRadius: 3,
              pointBackgroundColor: '#2563eb',
              order: 1,
            },
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'Recall' } },
            y: { min: 0, max: 1, title: { display: true, text: 'Precision' } },
          },
          plugins: {
            legend: {
              display: true,
              position: 'top',
              labels: { boxWidth: 20, font: { size: 11 } },
            },
          },
          animation: { duration: 250 },
        }
      });
    },

    selectClass(name) {
      this.selectedClass = name;
      this._updateDisplay();
    },

    setInterpolation(v) {
      this.interpolationPoints = Math.max(1, Math.min(101, v));
      if (this.nmsComplete) this._computeAP();
    },

    setIouMode(mode) {
      this.iouMode = mode;
      if (this.nmsComplete) this._computeAP();
    },

    setApIouValue(v) {
      this.apIouValue = v;
      if (this.nmsComplete) this._computeAP();
    },

    async _computeAP() {
      if (!_keptBoxes || !SimState.imageId) return;
      try {
        const result = await API.computeAP({
          image_id:             SimState.imageId,
          kept_boxes:           _keptBoxes,
          interpolation_points: this.interpolationPoints,
          iou_mode:             this.iouMode,
          iou_value:            this.apIouValue,
        });
        _allPrData                   = result.pr_curves;
        this.classes                 = result.per_class_ap.map(c => ({ name: c.class_name, ap: c.ap }));
        this.mapValue                = result.map;
        this.gtMatching              = result.gt_matching          || [];
        this.falsePositives          = result.false_positives      || [];
        this.predictionsMatching     = result.predictions_matching || [];
        this.falseNegatives          = result.false_negatives      || [];
        this.matchByPredId           = this._buildMatchMap(this.keptBoxesList, this.gtMatching);
        SimState.emit('ap_complete', { gtMatching: this.gtMatching });
        this._updateDisplay(result);
      } catch (err) {
        console.error('AP computation failed:', err);
      }
    },

    async refreshAP() {
      if (!this.nmsComplete || this.refreshing) return;
      this.refreshing = true;
      try {
        await this._computeAP();
        if (this.compareMode && _keptBoxesSoft) {
          await this._computeAPSoft(_keptBoxesSoft);
        }
      } finally {
        this.refreshing = false;
      }
    },

    async _computeAPSoft(keptBoxesSoft) {
      if (!SimState.imageId) return;
      try {
        const result = await API.computeAP({
          image_id:             SimState.imageId,
          kept_boxes:           keptBoxesSoft,
          interpolation_points: this.interpolationPoints,
          iou_mode:             this.iouMode,
          iou_value:            this.apIouValue,
        });
        // Store under Soft-NMS keys without touching main display state
        this.classesSoft             = result.per_class_ap.map(c => ({ name: c.class_name, ap: c.ap }));
        this.mapValueSoft            = result.map;
        this.apValueSoft             = result.map;
        this.gtMatchingSoft          = result.gt_matching          || [];
        this.falsePositivesSoft      = result.false_positives      || [];
        this.predictionsMatchingSoft = result.predictions_matching || [];
        this.matchByPredIdSoft       = this._buildMatchMap(this.keptBoxesListSoft, this.gtMatchingSoft);
        this.falseNegativesSoft      = result.false_negatives      || [];
        this.interimTableSoft        = result.interim_table        || [];
      } catch (err) {
        console.error('Soft-NMS AP failed:', err);
      }
    },

    // Build pred_id → {is_tp, iou, gt_taken} from gtMatching + keptList.
    // gtMatching is GT-centric and already correct (drives the configurator GT list).
    _buildMatchMap(keptList, gtMatchingList) {
      const map = {};
      // Start: every kept box is FP
      for (const box of (keptList || [])) {
        map[box.id] = { is_tp: false, iou: 0, gt_taken: false };
      }
      // Mark TPs — each matched GT row names exactly one pred_id
      for (const gm of (gtMatchingList || [])) {
        if (gm.pred_id != null && gm.matched) {
          map[gm.pred_id] = { is_tp: true, iou: gm.best_iou, gt_taken: false };
        }
      }
      // Mark "Taken" — FP prediction whose class has at least one matched GT
      // (meaning the GT of that class was already claimed by a higher-conf detection)
      const matchedClasses = new Set(
        (gtMatchingList || []).filter(gm => gm.matched).map(gm => gm.class_name)
      );
      for (const box of (keptList || [])) {
        const entry = map[box.id];
        if (entry && !entry.is_tp && matchedClasses.has(box.class_name)) {
          entry.gt_taken = true;
        }
      }
      return map;
    },

    // Returns the predictionsMatching entry for a given pred id (reactive-safe).
    getMatch(predId) {
      const list = this.compareMode && this.compareView === 'soft_nms'
        ? this.predictionsMatchingSoft : this.predictionsMatching;
      return list.find(m => m.pred_id === predId) || null;
    },

    _updateDisplay(result) {
      if (!result && !Object.keys(_allPrData).length) return;

      let recalls, precisions, rawRecalls, rawPrecisions, ap;

      if (this.selectedClass === '__all__') {
        const curves = Object.values(_allPrData);
        if (!curves.length) return;

        // Interpolated macro-average (N sparse staircase points)
        recalls    = curves[0].recalls;
        precisions = recalls.map((_, i) =>
          curves.reduce((s, c) => s + (c.precisions[i] || 0), 0) / curves.length
        );
        ap = this.mapValue;

        // Dense raw macro-average: per-class precision envelope at 101 levels, averaged
        const N_DENSE = 101;
        rawRecalls    = [];
        rawPrecisions = [];
        for (let i = 0; i < N_DENSE; i++) {
          const rl = i / (N_DENSE - 1);
          rawRecalls.push(rl);
          let sum = 0;
          for (const c of curves) {
            const rr = c.raw_recalls    || [];
            const rp = c.raw_precisions || [];
            let maxP = 0;
            for (let j = 0; j < rr.length; j++) {
              if (rr[j] >= rl && rp[j] > maxP) maxP = rp[j];
            }
            sum += maxP;
          }
          rawPrecisions.push(sum / curves.length);
        }
      } else {
        const curve = _allPrData[this.selectedClass];
        if (!curve) return;
        recalls       = curve.recalls;
        precisions    = curve.precisions;
        rawRecalls    = curve.raw_recalls    || [];
        rawPrecisions = curve.raw_precisions || [];
        const cls  = this.classes.find(c => c.name === this.selectedClass);
        ap = cls ? cls.ap : null;
      }

      this.apValue      = ap;
      this.interimTable = (recalls || []).map((r, i) => ({
        conf_threshold: 1 - (i / Math.max(recalls.length - 1, 1)),
        recall:         r,
        precision:      precisions[i],
      }));

      if (_prChart) {
        _prChart.data.datasets[0].data = rawRecalls.map((r, i) => ({ x: r, y: rawPrecisions[i] }));
        _prChart.data.datasets[1].data = recalls.map((r, i) => ({ x: r, y: precisions[i] }));
        _prChart.update();
      }
    },
  };
}
