const fileInput = document.querySelector('#file-input');
const dropZone = document.querySelector('#drop-zone');
const fileName = document.querySelector('#file-name');
const verifyButton = document.querySelector('#verify-button');
const previewImage = document.querySelector('#preview-image');
const previewEmpty = document.querySelector('#preview-empty');
const dimensions = document.querySelector('#image-dimensions');
const loadingCard = document.querySelector('#loading-card');
const errorCard = document.querySelector('#error-card');
const resultCard = document.querySelector('#result-card');
let selectedFile = null;

function chooseFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    showError('Please select a supported image file.');
    return;
  }
  selectedFile = file;
  fileName.textContent = file.name;
  verifyButton.disabled = false;
  resultCard.classList.add('hidden');
  errorCard.classList.add('hidden');
  const objectUrl = URL.createObjectURL(file);
  previewImage.src = objectUrl;
  previewImage.onload = () => {
    dimensions.textContent = `${previewImage.naturalWidth} × ${previewImage.naturalHeight}`;
    URL.revokeObjectURL(objectUrl);
  };
  previewImage.classList.remove('hidden');
  previewEmpty.classList.add('hidden');
}

function showError(message) {
  errorCard.textContent = message;
  errorCard.classList.remove('hidden');
  loadingCard.classList.add('hidden');
}

fileInput.addEventListener('change', (event) => chooseFile(event.target.files[0]));
dropZone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') fileInput.click();
});
['dragenter', 'dragover'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropZone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropZone.classList.remove('dragging');
}));
dropZone.addEventListener('drop', (event) => chooseFile(event.dataTransfer.files[0]));

verifyButton.addEventListener('click', async () => {
  if (!selectedFile) return;
  loadingCard.classList.remove('hidden');
  errorCard.classList.add('hidden');
  resultCard.classList.add('hidden');
  verifyButton.disabled = true;
  const formData = new FormData();
  formData.append('file', selectedFile);
  try {
    const response = await fetch('/verify', { method: 'POST', body: formData });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'Verification failed.');
    renderResult(body);
  } catch (error) {
    showError(error.message);
  } finally {
    loadingCard.classList.add('hidden');
    verifyButton.disabled = false;
  }
});

function renderResult(body) {
  const forged = body.label === 'forged';
  const genuinePercent = Math.round(body.genuine_probability * 1000) / 10;
  const forgedPercent = Math.round(body.forged_probability * 1000) / 10;
  document.querySelector('#result-label').textContent = forged ? 'Potentially forged' : 'Likely genuine';
  document.querySelector('#result-confidence').textContent = `${forged ? forgedPercent : genuinePercent}%`;
  document.querySelector('#genuine-value').textContent = `${genuinePercent}%`;
  document.querySelector('#forged-value').textContent = `${forgedPercent}%`;
  document.querySelector('#genuine-bar').style.width = `${genuinePercent}%`;
  document.querySelector('#forged-bar').style.width = `${forgedPercent}%`;
  document.querySelector('#heatmap-image').src = body.heatmap;
  document.querySelector('#result-device').textContent = body.device;
  document.querySelector('#result-threshold').textContent = body.threshold;
  document.querySelector('#result-explanation').textContent = forged
    ? 'The model classified this image as forged. The heatmap shows the regions that contributed most to that decision; review these regions with the original document.'
    : 'The model classified this image as genuine. The heatmap shows the regions that contributed most to that decision; this result should still be reviewed in a real verification workflow.';
  resultCard.classList.remove('hidden');
}
