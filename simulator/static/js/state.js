/**
 * Simulator-wide state bus.
 *
 * Components do NOT import each other directly.  Instead they emit events on
 * SimState and listen for events fired by other components.  This keeps the
 * Alpine components decoupled.
 *
 * States: idle | image_loaded | nms_running | nms_complete | faulted | recomputing
 */

window.SimState = (() => {
  let _state      = 'idle';
  let _imageId    = null;
  let _imageUrl   = null;
  let _rawBoxes   = [];
  let _steps      = [];
  let _keptBoxes  = [];
  let _gtBoxes    = [];    // ground-truth boxes from label file

  const listeners = {};

  function on(event, cb)    { (listeners[event] = listeners[event] || []).push(cb); }
  function off(event, cb)   { listeners[event] = (listeners[event] || []).filter(f => f !== cb); }
  function emit(event, data){ (listeners[event] || []).forEach(cb => cb(data)); }

  function transition(next, data = {}) {
    _state = next;
    emit('state_change', { state: next, ...data });
    emit(next, data);
  }

  return {
    on, off, emit,

    get state()     { return _state; },
    get imageId()   { return _imageId; },
    get imageUrl()  { return _imageUrl; },
    get rawBoxes()  { return _rawBoxes; },
    get steps()     { return _steps; },
    get keptBoxes() { return _keptBoxes; },
    get gtBoxes()   { return _gtBoxes; },

    setImage(id, url) {
      _imageId  = id;
      _imageUrl = url;
      _rawBoxes  = [];
      _steps     = [];
      _keptBoxes = [];
      _gtBoxes   = [];
    },

    setGtBoxes(boxes) {
      _gtBoxes = boxes;
      emit('gt_loaded', { boxes });
    },

    setRawBoxes(boxes) {
      _rawBoxes = boxes;
      transition('image_loaded', { boxes });
    },

    setSteps(steps) {
      _steps = steps;
      transition('nms_running', { steps });
    },

    setNmsComplete(keptBoxes, keptBoxesSoft = null) {
      _keptBoxes = keptBoxes;
      transition('nms_complete', { keptBoxes, keptBoxesSoft, compareMode: keptBoxesSoft !== null });
    },

    recomputing() { transition('recomputing'); },
    fault(msg)    { transition('faulted', { message: msg }); },
    reset()       { _imageId = null; _imageUrl = null; _rawBoxes = []; _steps = []; _keptBoxes = []; transition('idle'); },
  };
})();
