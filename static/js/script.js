// Theme toggle functionality (shared across all pages)
const toggleBtn = document.getElementById("theme-toggle");
const body = document.body;
const html = document.documentElement;

if (toggleBtn) {
  // Get saved theme or default to light
  const savedTheme = localStorage.getItem("theme") || "light";
  
  // Apply theme to both html and body
  body.className = savedTheme;
  html.className = savedTheme;
  
  // Update icon based on theme
  const icon = toggleBtn.querySelector("i");
  if (savedTheme === "dark") {
    icon.classList.remove("fa-moon");
    icon.classList.add("fa-sun");
  } else {
    icon.classList.remove("fa-sun");
    icon.classList.add("fa-moon");
  }
  
  // Toggle theme on button click
  toggleBtn.addEventListener("click", () => {
    // Toggle classes on both html and body
    body.classList.toggle("dark");
    body.classList.toggle("light");
    html.classList.toggle("dark");
    html.classList.toggle("light");
    
    // Update icon
    const icon = toggleBtn.querySelector("i");
    if (body.classList.contains("dark")) {
      icon.classList.remove("fa-moon");
      icon.classList.add("fa-sun");
      localStorage.setItem("theme", "dark");
    } else {
      icon.classList.remove("fa-sun");
      icon.classList.add("fa-moon");
      localStorage.setItem("theme", "light");
    }
  });
}

// Upload logic (for face detection/recognition pages)
const fileUpload = document.getElementById("file-upload");
const uploadBox = document.getElementById("upload-box");
const preview = document.getElementById("preview");
const previewImg = document.getElementById("preview-img");
const fileName = document.getElementById("file-name");
const processBtn = document.getElementById("process-btn");

if (fileUpload && !document.querySelector('.filters-page')) {
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
      
      if (processBtn) {
        processBtn.classList.remove("hidden");
      }
    };
    reader.readAsDataURL(file);
  });
}