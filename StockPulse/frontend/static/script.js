const dropArea = document.getElementById("drop-area");
const fileElem = document.getElementById("fileElem");
const fileNameDisplay = document.getElementById("fileName");
const clearBtn = document.getElementById("clearBtn");
const uploadForm = document.getElementById("uploadForm");
const previewArea = document.getElementById("previewArea");
let uploadedFile = null;
let fileValid = false;

if (dropArea && fileElem) {
  dropArea.addEventListener("click", () => fileElem.click());
}

if (fileElem) {
  fileElem.addEventListener("change", (e) => handleFileUpload(e.target.files[0]));
}

if (dropArea) {
  dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.style.background = "rgba(0, 188, 212, 0.2)";
  });

  dropArea.addEventListener("dragleave", () => {
    dropArea.style.background = "rgba(255, 255, 255, 0.05)";
  });

  dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
      fileElem.files = e.dataTransfer.files;
      e.dataTransfer.clearData();
    }
    dropArea.style.background = "rgba(255, 255, 255, 0.05)";
  });
}

function handleFileUpload(file) {
  uploadedFile = file;
  if (!uploadedFile) return;
  fileNameDisplay.textContent = "📄 " + uploadedFile.name;
  const formData = new FormData();
  formData.append("file", uploadedFile);
  fetch(`${window.location.origin}/preview`, { method: "POST", body: formData })
    .then(res => res.json())
    .then(data => {
      if (data.status === "ok") {
        previewArea.innerHTML = data.html;
        fileValid = true;
      } else {
        previewArea.innerHTML = `<p style="color:red; font-weight:bold;">${data.message}</p>`;
        fileValid = false;
      }
    })
    .catch(err => {
      previewArea.innerHTML = `<p style="color:red;">Error: ${err.message}</p>`;
      fileValid = false;
    });
}

if (clearBtn) {
  clearBtn.addEventListener("click", () => {
    fileElem.value = "";
    fileNameDisplay.textContent = "";
    previewArea.innerHTML = "";
    uploadedFile = null;
    fileValid = false;
  });
}


const analyzeBtn = document.getElementById("analyzeBtn");
const paramModal = document.getElementById("paramModal");
const applyParamsBtn = document.getElementById("applyParamsBtn");
const skipParamsBtn = document.getElementById("skipParamsBtn");

const n_estimators_hidden = document.getElementById("n_estimators_hidden");
const contamination_hidden = document.getElementById("contamination_hidden");
const random_state_hidden = document.getElementById("random_state_hidden");

const n_estimators_input = document.getElementById("n_estimators_input");
const contamination_input = document.getElementById("contamination_input");
const random_state_input = document.getElementById("random_state_input");

if (analyzeBtn) {
  analyzeBtn.addEventListener("click", (e) => {
    e.preventDefault();
    if (!uploadedFile || !fileValid) {
      alert("⚠️ Please upload a valid file first");
      return;
    }
    paramModal.style.display = "flex";
  });
}

if (applyParamsBtn) {
  applyParamsBtn.addEventListener("click", (e) => {
    e.preventDefault();
    n_estimators_hidden.value = n_estimators_input.value;
    contamination_hidden.value = contamination_input.value;
    random_state_hidden.value = random_state_input.value;
    paramModal.style.display = "none";
    uploadForm.submit();
  });
}

if (skipParamsBtn) {
  skipParamsBtn.addEventListener("click", (e) => {
    e.preventDefault();
    paramModal.style.display = "none";
    uploadForm.submit();
  });
}


window.addEventListener("pageshow", () => {
    if (clearBtn) {
        clearBtn.click();
    }
});