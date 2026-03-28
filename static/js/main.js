const API_URL = 'http://localhost:5000';

async function analyzeEmail() {
    const subject = document.getElementById('subject').value;
    const body = document.getElementById('body').value;
    const headers = document.getElementById('headers').value;

    if (!subject && !body) {
        alert('Please enter subject or body of the email.');
        return;
    }

    try {
        // Prediction
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ subject, body, headers })
        });

        const data = await response.json();
        if (!response.ok) {
            console.error('Server error:', data);
            return;
        }

        displayPrediction(data);

        // Explanation
        const expResponse = await fetch(`${API_URL}/explain`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ subject, body })
        });

        const expData = await expResponse.json();
        if (!expResponse.ok) {
            console.error('Server error (explain):', expData);
            return;
        }

        displayExplanation(expData);

        // Refresh history
        fetchHistory();

    } catch (err) {
        console.error('Error:', err);
    }
}

function displayPrediction(data) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `
        <div class="alert alert-info">
            <strong>Label:</strong> ${data.final_hint} <br>
            <strong>Phishing Score:</strong> ${data.phishing_score.toFixed(2)}
        </div>
    `;

    // Show security signals
    document.getElementById('securityBox').style.display = 'block';
    document.getElementById('authPanel').innerHTML = Object.entries(data.auth)
        .map(([k,v]) => `<div>${k}: ${v}</div>`).join('');
    document.getElementById('urlRisk').textContent = `Overall URL risk: ${data.urls.url_risk.toFixed(2)}`;
    document.getElementById('urlList').innerHTML = (data.urls.urls || [])
    .map(u => `<li style="color:${u.risk > 0.7 ? 'red':'black'}">${u.url} (risk: ${u.risk.toFixed(2)})</li>`)
    .join('');
}

function displayExplanation(expData) {
    const tokensDiv = document.getElementById('tokens');
    tokensDiv.innerHTML = expData.tokens
        .map(t => `<span class="badge bg-secondary m-1">${t.token}: ${t.weight.toFixed(2)}</span>`)
        .join('');
    document.getElementById('explainBox').style.display = 'block';
}

async function fetchHistory() {
    try {
        const response = await fetch(`${API_URL}/history`);
        const data = await response.json();
        if (!response.ok) return;

        const tbody = document.getElementById('historyBody');
        tbody.innerHTML = data.items.map(item => `
            <tr>
                <td>${item.timestamp}</td>
                <td>${item.subject}</td>
                <td>${item.label}</td>
                <td>${item.probs.legit?.toFixed(2)}</td>
                <td>${item.probs.spam?.toFixed(2)}</td>
                <td>${item.probs.phishing?.toFixed(2)}</td>
                <td>${item.auth.auth_risk}</td>
                <td>${item.urls.url_risk}</td>
                <td>${item.final_hint}</td>
            </tr>
        `).join('');

    } catch (err) {
        console.error('Error fetching history:', err);
    }
}

// Event listeners
document.getElementById('checkBtn').addEventListener('click', analyzeEmail);
document.getElementById('clearBtn').addEventListener('click', () => {
    document.getElementById('subject').value = '';
    document.getElementById('body').value = '';
    document.getElementById('headers').value = '';
    document.getElementById('result').innerHTML = '';
    document.getElementById('tokens').innerHTML = '';
    document.getElementById('explainBox').style.display = 'none';
    document.getElementById('securityBox').style.display = 'none';
});
document.getElementById("predict-resnet-btn").addEventListener("click", async () => {
    let subject = document.getElementById("subject").value;
    let body = document.getElementById("body").value;

    let response = await fetch("/predict_resnet", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({subject: subject, body: body})
    });

    let result = await response.json();
    document.getElementById("result").innerText =
        "ResNet Prediction: " + result.prediction + "\n" +
        JSON.stringify(result.probabilities, null, 2);
});
document.getElementById("predict-cnn-btn").addEventListener("click", async () => {
    let subject = document.getElementById("subject").value;
    let body = document.getElementById("body").value;

    try {
        let response = await fetch(`${API_URL}/predict_cnn`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ subject: subject, body: body })
        });

        let result = await response.json();

        if (!response.ok) {
            console.error("Error:", result);
            return;
        }

        // Show CNN result in the same result div
        const resultDiv = document.getElementById("result");
        resultDiv.innerHTML = `
            <div class="alert alert-warning">
                <strong>CNN Prediction:</strong> ${result.prediction} <br>
                <strong>Probabilities:</strong> ${JSON.stringify(result.probabilities, null, 2)}
            </div>
        `;
    } catch (err) {
        console.error("Error calling CNN API:", err);
    }
});
document.getElementById("final-result-btn").addEventListener("click", async () => {
    const subject = document.getElementById("subject").value;
    const body = document.getElementById("body").value;

    if (!subject && !body) {
        alert("Please enter subject or body of the email.");
        return;
    }

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ subject, body })
        });

        const result = await response.json();

        if (!response.ok) {
            console.error("Error fetching final result:", result);
            return;
        }

        // Show the final prediction
const finalDiv = document.getElementById("finalResultDiv");
finalDiv.innerHTML = `
    <div class="alert alert-dark">
        <strong>Final Prediction:</strong> ${result.best_label} <br>
        <strong>Best Model:</strong> ${result.best_model} <br>
        <strong>Confidence:</strong> ${(result.best_confidence * 100).toFixed(2)}%
    </div>
    <hr>
    <strong>Per-Model Details:</strong>
    <ul>
        ${result.per_model.map(m => `<li>${m.model}: ${m.label} (${(m.confidence*100).toFixed(2)}%)</li>`).join('')}
    </ul>
`;


    } catch (err) {
        console.error("Error calling final result API:", err);
    }
});


// Initial load of history
fetchHistory();
