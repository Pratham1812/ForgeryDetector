// frontend/script.js

document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const loadingDiv = document.getElementById('loading');
    const resultsDiv = document.getElementById('results');
    const processingTimeSpan = document.getElementById('processingTime');
    const edgePlotImg = document.getElementById('edgePlot');
    const classificationContainer = document.getElementById('classificationContainer');
    const classificationPlotImg = document.getElementById('classificationPlot');
    const errorDiv = document.getElementById('error');
    const errorMessageP = document.querySelector('.error-message');

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Hide previous results and errors
        resultsDiv.classList.add('hidden');
        errorDiv.classList.add('hidden');

        // Show loading indicator
        loadingDiv.classList.remove('hidden');

        // Prepare form data
        const formData = new FormData(uploadForm);
        formData.binary_classification = document.getElementById('binaryClassification').checked;

        // Get selected method
        const method = document.getElementById('method').value;
        try {
            const response = await fetch(`http://localhost:5000/predict/${method}`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                // Display processing time
                processingTimeSpan.textContent = "http://loclahost:5000"+data.processing_time;

                // Display edge intensity plot
                edgePlotImg.src = "http://loclahost:5000"+data.plot_url;
                edgePlotImg.alt = 'Edge Intensity Plot';
                console.log(data.plot_url);
                console.log(data.classification_plot_url);
                // Check if classification plot exists
                if (data.classification_plot_url) {
                    classificationPlotImg.src = "''"+data.classification_plot_url;
                    classificationPlotImg.alt = 'Classification Plot';
                    classificationContainer.classList.remove('hidden');
                } else {
                    classificationContainer.classList.add('hidden');
                }

                // Show results
                resultsDiv.classList.remove('hidden');
            } else {
                // Display error message
                errorMessageP.textContent = data.error || 'An error occurred during processing.';
                errorDiv.classList.remove('hidden');
            }
        } catch (error) {
            console.error('Error:', error);
            errorMessageP.textContent = 'An unexpected error occurred.';
            errorDiv.classList.remove('hidden');
        } finally {
            // Hide loading indicator
            loadingDiv.classList.add('hidden');
        }
    });
});
