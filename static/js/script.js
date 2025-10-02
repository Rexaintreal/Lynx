
document.addEventListener("DOMContentLoaded", () => {
    const toggle = document.getElementById("theme-toggle");
    const body = document.body;

    // Restore theme from localStorage
    if (localStorage.getItem("theme") === "light") {
    body.classList.remove("dark");
    toggle.checked = false;
    } else {
    body.classList.add("dark");
    toggle.checked = true;
    }

    toggle.addEventListener("change", () => {
    if (toggle.checked) {
        body.classList.add("dark");
        localStorage.setItem("theme", "dark");
    } else {
        body.classList.remove("dark");
        localStorage.setItem("theme", "light");
    }
    });
});
