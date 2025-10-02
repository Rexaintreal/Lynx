// Theme toggle
const toggleBtn = document.getElementById("theme-toggle");
const body = document.body;

if (toggleBtn) {
  const savedTheme = localStorage.getItem("theme") || "light";
  body.className = savedTheme;

  const icon = toggleBtn.querySelector("i");
  if (savedTheme === "dark") {
    icon.classList.replace("fa-moon", "fa-sun");
  }

  toggleBtn.addEventListener("click", () => {
    body.classList.toggle("dark");
    body.classList.toggle("light");

    const icon = toggleBtn.querySelector("i");
    if (body.classList.contains("dark")) {
      icon.classList.replace("fa-moon", "fa-sun");
      localStorage.setItem("theme", "dark");
    } else {
      icon.classList.replace("fa-sun", "fa-moon");
      localStorage.setItem("theme", "light");
    }
  });
}

// Upload logic
const fileUpload = document.getElementById("file-upload");
const uploadBox = document.getElementById("upload-box");
const preview = document.getElementById("preview");
const previewImg = document.getElementById("preview-img");
const fileName = document.getElementById("file-name");
const processBtn = document.getElementById("process-btn");

if (fileUpload) {
  fileUpload.addEventListener("change", () => {
    const file = fileUpload.files[0];
    if (!file) return;

    // Validate
    if (file.size > 10 * 1024 * 1024) {
      alert("File too large (max 10MB)");
      return;
    }
    if (!["image/png", "image/jpeg", "image/jpg"].includes(file.type)) {
      alert("Only PNG/JPG allowed");
      return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = e => {
      uploadBox.classList.add("hidden");      
      previewImg.src = e.target.result;
      fileName.textContent = file.name;
      preview.classList.remove("hidden");    
      processBtn.classList.remove("hidden");  
    };
    reader.readAsDataURL(file);
  });
}
