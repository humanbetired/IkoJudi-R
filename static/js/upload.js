document.addEventListener("DOMContentLoaded", function () {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-upload');
    const fileNameDisplay = document.getElementById('fileName');
    const uploadForm = document.getElementById('uploadForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = document.getElementById('btnText');
    const loadingSpinner = document.getElementById('loadingSpinner');

    if (dropzone && fileInput) {

        function showUploadSuccess(file) {
            dropzone.classList.remove('border-gray-200', 'hover:border-sage');
            dropzone.classList.add('border-sage', 'bg-cream');
            fileNameDisplay.textContent = `File "${file.name}" uploaded successfully.`;
            fileNameDisplay.classList.remove('hidden');
        }

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];

            if (file) {
                const maxSize = 1024 * 1024 * 1024; 

                if (file.size > maxSize) {
                    alert("File terlalu besar! Maksimal 5MB.");
                    fileInput.value = "";
                    return;
                }

                showUploadSuccess(file);
            }
        });

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('bg-cream', 'border-sage');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('bg-cream', 'border-sage');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            const files = e.dataTransfer.files;

            if (files.length > 0) {
                const file = files[0];
                const maxSize = 1024 * 1024 * 1024;

                if (file.size > maxSize) {
                    alert("File terlalu besar! Maksimal 5MB.");
                    return;
                }

                fileInput.files = files;
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
        });
    }
    const btn = document.getElementById("menu-btn");
    const menu = document.getElementById("mobile-menu");

    if (btn && menu) {
        btn.addEventListener("click", function () {
            menu.classList.toggle("hidden");
        });
    }

});