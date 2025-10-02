// Theme toggle functionality
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
      
      if (processBtn) {
        processBtn.classList.remove("hidden");
      }
    };
    reader.readAsDataURL(file);
  });
}

// FILTERS PAGE FUNCTIONALITY


// Filter controls
const brightnessSlider = document.getElementById("brightness");
const contrastSlider = document.getElementById("contrast");
const sepiaSlider = document.getElementById("sepia");
const blurSlider = document.getElementById("blur");
const presetSelect = document.getElementById("preset-filters");
const resetBtn = document.getElementById("reset-btn");
const saveBtn = document.getElementById("save-btn");

// Store original image data
let originalImageSrc = null;

// Check if we're on the filters page by looking for filter controls
const isFiltersPage = brightnessSlider !== null;

// If on filters page, handle upload differently
if (isFiltersPage && fileUpload) {
  fileUpload.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Validate file size
    if (file.size > 10 * 1024 * 1024) {
      alert("File too large! Maximum size is 10MB.");
      return;
    }

    // Validate file type
    if (!["image/png", "image/jpeg", "image/jpg"].includes(file.type)) {
      alert("Invalid file type! Please upload PNG, JPG, or JPEG.");
      return;
    }

    // Read and display image
    const reader = new FileReader();
    reader.onload = (e) => {
      originalImageSrc = e.target.result;
      if (previewImg) {
        previewImg.src = originalImageSrc;
      }
      if (fileName) {
        fileName.textContent = file.name;
      }
      
      // Hide upload box, show preview
      if (uploadBox) {
        uploadBox.style.display = "none";
      }
      if (preview) {
        preview.classList.remove("hidden");
      }
      
      // Reset filters
      resetFilters();
    };
    reader.readAsDataURL(file);
  });
}

// Apply filters function
function applyFilters() {
  if (!previewImg) return;
  
  const brightness = brightnessSlider.value;
  const contrast = contrastSlider.value;
  const sepia = sepiaSlider.value;
  const blur = blurSlider.value;
  
  const filterString = `
    brightness(${brightness}%)
    contrast(${contrast}%)
    sepia(${sepia}%)
    blur(${blur}px)
  `;
  
  previewImg.style.filter = filterString;
}

// Attach event listeners to sliders
if (brightnessSlider) brightnessSlider.addEventListener("input", applyFilters);
if (contrastSlider) contrastSlider.addEventListener("input", applyFilters);
if (sepiaSlider) sepiaSlider.addEventListener("input", applyFilters);
if (blurSlider) blurSlider.addEventListener("input", applyFilters);

// Preset filters
if (presetSelect) {
  presetSelect.addEventListener("change", (e) => {
    const preset = e.target.value;
    
    // Reset sliders first
    resetSliders();
    
    switch(preset) {
      case "grayscale":
        sepiaSlider.value = 0;
        previewImg.style.filter = "grayscale(100%)";
        break;
      case "sepia":
        sepiaSlider.value = 100;
        applyFilters();
        break;
      case "invert":
        previewImg.style.filter = "invert(100%)";
        break;
      case "cool":
        brightnessSlider.value = 110;
        contrastSlider.value = 120;
        previewImg.style.filter = "brightness(110%) contrast(120%) hue-rotate(180deg)";
        break;
      case "vibrant":
        contrastSlider.value = 150;
        brightnessSlider.value = 105;
        applyFilters();
        break;
      case "none":
        resetFilters();
        break;
    }
  });
}

// Reset sliders to default
function resetSliders() {
  if (brightnessSlider) brightnessSlider.value = 100;
  if (contrastSlider) contrastSlider.value = 100;
  if (sepiaSlider) sepiaSlider.value = 0;
  if (blurSlider) blurSlider.value = 0;
}

// Reset all filters
function resetFilters() {
  resetSliders();
  if (presetSelect) presetSelect.value = "none";
  if (previewImg) previewImg.style.filter = "none";
}

// Reset button
if (resetBtn) {
  resetBtn.addEventListener("click", () => {
    resetFilters();
  });
}

// Save button - download image with filters
if (saveBtn) {
  saveBtn.addEventListener("click", () => {
    if (!previewImg || !originalImageSrc) return;
    
    // Create a canvas to apply filters and download
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      
      // Apply filters to canvas context
      const brightness = brightnessSlider.value / 100;
      const contrast = contrastSlider.value / 100;
      const sepia = sepiaSlider.value / 100;
      const blur = blurSlider.value;
      
      // Set filters
      ctx.filter = `
        brightness(${brightness})
        contrast(${contrast})
        sepia(${sepia})
        blur(${blur}px)
      `;
      
      // Draw filtered image
      ctx.drawImage(img, 0, 0);
      
      // Download
      canvas.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "filtered_" + (fileName.textContent || "image.png");
        a.click();
        URL.revokeObjectURL(url);
      });
    };
    img.src = originalImageSrc;
  });
}