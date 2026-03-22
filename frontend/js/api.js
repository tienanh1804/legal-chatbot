const API_BASE_URL = "http://localhost:8001";

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
