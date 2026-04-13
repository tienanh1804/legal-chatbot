const API_BASE_URL =
  window.API_BASE_URL ||
  (() => {
    const host = window.location.hostname || "localhost";
    // Docker Compose mapping: frontend -> 8088, backend -> 8002.
    return `http://${host}:8002`;
  })();

async function apiRequest(endpoint, method = "GET", body = null, token = null, contentType = "application/json") {
  const headers = {};

  if (contentType) {
    headers["Content-Type"] = contentType;
  }

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method,
    headers,
    body
  });

  if (!response.ok) {
    let errorMessage = "Yêu cầu thất bại";
    try {
      const errorData = await response.json();
      errorMessage = errorData.detail || JSON.stringify(errorData);
    } catch (error) {
      errorMessage = await response.text();
    }
    throw new Error(errorMessage || "Có lỗi xảy ra");
  }

  return response.json();
}

/** Upload a file (multipart). Do not set Content-Type manually (browser sets boundary). */
async function apiUploadFile(endpoint, file, token) {
  const fd = new FormData();
  fd.append("file", file);
  const headers = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    headers,
    body: fd,
  });
  if (!response.ok) {
    let errorMessage = "Tải file thất bại";
    try {
      const errorData = await response.json();
      errorMessage = errorData.detail || JSON.stringify(errorData);
    } catch {
      errorMessage = await response.text();
    }
    throw new Error(errorMessage || "Có lỗi xảy ra");
  }
  return response.json();
}
