function showAuthAlert(message, type = "danger") {
  const alertBox = document.getElementById("auth-alert");
  if (!alertBox) return;
  alertBox.innerHTML = `<div class="alert alert-${type}">${message}</div>`;
}

const loginForm = document.getElementById("login-form");
const registerForm = document.getElementById("register-form");

if (loginForm) {
  loginForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    const username = document.getElementById("login-username").value.trim();
    const password = document.getElementById("login-password").value.trim();

    try {
      const formData = new URLSearchParams();
      formData.append("username", username);
      formData.append("password", password);

      const data = await apiRequest(
        "/users/token",
        "POST",
        formData.toString(),
        null,
        "application/x-www-form-urlencoded"
      );

      localStorage.setItem("token", data.access_token);
      showAuthAlert("Đăng nhập thành công.", "success");

      setTimeout(() => {
        window.location.href = "index.html";
      }, 800);
    } catch (error) {
      showAuthAlert(error.message);
    }
  });
}

if (registerForm) {
  registerForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    const username = document.getElementById("register-name").value.trim();
    const email = document.getElementById("register-email").value.trim();
    const password = document.getElementById("register-password").value.trim();

    try {
      await apiRequest(
        "/users/register",
        "POST",
        JSON.stringify({ username, email, password }),
        null,
        "application/json"
      );

      showAuthAlert(`Đăng ký thành công cho ${username}.`, "success");

      setTimeout(() => {
        window.location.href = "login.html";
      }, 800);
    } catch (error) {
      showAuthAlert(error.message);
    }
  });
}
