const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-upload');
const fileNameDisplay = document.getElementById('fileName');
const uploadForm = document.getElementById('uploadForm');
const analyzeBtn = document.getElementById('analyzeBtn');
const btnText = document.getElementById('btnText');
const loadingSpinner = document.getElementById('loadingSpinner');

function showUploadSuccess(file) {
    dropzone.classList.remove('border-gray-200', 'hover:border-sage');
    dropzone.classList.add('border-sage', 'bg-cream');
    fileNameDisplay.textContent = `File "${file.name}" uploaded successfully.`;
    fileNameDisplay.classList.remove('hidden');
}

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) showUploadSuccess(file);
});

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('bg-cream', 'border-sage');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('bg-cream', 'border-sage');
    if (fileInput.files.length === 0) {
        dropzone.classList.add('border-gray-200', 'hover:border-sage');
    }
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('bg-cream');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        fileInput.files = e.dataTransfer.files;
        showUploadSuccess(file);
    }
});

uploadForm.addEventListener('submit', function (e) {
    if (!fileInput.files || fileInput.files.length === 0) {
        e.preventDefault();
        alert('Please select an image file first');
        return;
    }

    btnText.textContent = 'Analyzing...';
    loadingSpinner.classList.remove('hidden');
    analyzeBtn.disabled = true;
    analyzeBtn.classList.add('opacity-75');
});

window.addEventListener('unload', function () {
    btnText.textContent = 'Analisis Gambar';
    loadingSpinner.classList.add('hidden');
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove('opacity-75');
});