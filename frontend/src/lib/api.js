const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }

  return response.json();
}

export const api = {
  getConfig() {
    return request("/config");
  },
  reset(payload) {
    return request("/reset", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
  step(steps = 1) {
    return request("/step", {
      method: "POST",
      body: JSON.stringify({ steps }),
    });
  },
  getState() {
    return request("/state");
  },
  getMetrics() {
    return request("/metrics");
  },
  compare(payload) {
    return request("/compare", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
};
