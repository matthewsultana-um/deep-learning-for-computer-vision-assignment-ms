/**
 * Thin fetch wrappers for every backend endpoint.
 * All functions return parsed JSON or throw on non-2xx.
 */

window.API = (() => {
  async function request(method, path, body = null, timeoutMs = 90000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    const opts = { method, headers: {}, signal: controller.signal };
    if (body && !(body instanceof FormData)) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    } else if (body) {
      opts.body = body;
    }

    try {
      const res = await fetch(path, opts);
      if (!res.ok) {
        const detail = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(detail.detail || res.statusText);
      }
      return res.json();
    } catch (err) {
      if (err.name === 'AbortError') {
        throw new Error('Request timed out. Is the server running?');
      }
      if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
        throw new Error('Cannot reach server. Run: python -m uvicorn app.main:app --port 8000');
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  return {
    getSamples:     ()       => request('GET',  '/api/samples'),
    getExamples:    ()       => request('GET',  '/api/examples'),
    upload:         (form)   => request('POST', '/api/upload',    form),
    detect:         (body)   => request('POST', '/api/detect',    body),
    runNms:         (body)   => request('POST', '/api/nms/run',   body),
    computeAP:      (body)   => request('POST', '/api/metrics/ap',body),
    getGroundTruth: (id)     => request('GET',  `/api/groundtruth/${id}`),
  };
})();
