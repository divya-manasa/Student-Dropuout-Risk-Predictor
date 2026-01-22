document.getElementById('predictionForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const submitBtn = document.querySelector('.btn-predict');
    const originalBtnText = submitBtn.innerText;
    submitBtn.innerText = 'Calculating...';
    submitBtn.disabled = true;

    // Collect data
    const formData = new FormData(this);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = value;
    });

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();

        displayResult(result);

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while predicting. Please try again.');
    } finally {
        submitBtn.innerText = originalBtnText;
        submitBtn.disabled = false;
    }
});

function displayResult(result) {
    const resultContainer = document.getElementById('resultContainer');
    const probValue = document.getElementById('probValue');
    const riskValue = document.getElementById('riskValue');
    const probBar = document.getElementById('probBar');

    // Reset animations
    resultContainer.classList.remove('result-visible');

    setTimeout(() => {
        // Update values
        probValue.innerText = (result.probability * 100).toFixed(1) + '%';
        riskValue.innerText = result.risk;

        // Remove old classes
        probValue.className = 'value';
        riskValue.className = 'value';
        probBar.className = 'prob-bar';

        // Add new classes
        riskValue.classList.add(result.risk_class);
        probBar.classList.add('bg-' + result.risk_class.split('-')[0]); // e.g., bg-low

        // Set bar width
        // Wait slightly for transition
        setTimeout(() => {
            probBar.style.width = (result.probability * 100) + '%';
        }, 100);

        // Show result
        resultContainer.style.display = 'block';
        // Force reflow
        void resultContainer.offsetWidth;
        resultContainer.classList.add('result-visible');

        // Scroll to result on mobile
        if (window.innerWidth < 600) {
            resultContainer.scrollIntoView({ behavior: 'smooth' });
        }
    }, 100);
}
