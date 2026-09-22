const API_URL = "http://127.0.0.1:8000/api/analyze";

const analyzeButton = document.getElementById("analyzeButton");
const connectionStatus = document.getElementById("connectionStatus");
const loading = document.getElementById("loading");
const error = document.getElementById("error");
const result = document.getElementById("result");
const probability = document.getElementById("probability");
const emotion = document.getElementById("emotion");
const riskLevel = document.getElementById("riskLevel");
const riskText = document.getElementById("riskText");
const message = document.getElementById("message");

function setVisible(element, visible) {
    element.classList.toggle("hidden", !visible);
}

function setError(text) {
    error.textContent = text;
    setVisible(error, Boolean(text));
}

function setRiskClass(risk) {
    riskLevel.classList.remove("risk-low", "risk-medium", "risk-high");
    riskLevel.classList.add(`risk-${risk}`);
}

// The activeTab and scripting permissions let the popup inject this small
// content script only after the user clicks Analyze selection.
async function getSelectedText() {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab?.id) {
        throw new Error("No active browser tab was found.");
    }

    await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["content.js"]
    });

    // Message passing asks the content script for the current selection. The
    // content script never scans or sends the rest of the webpage.
    const response = await chrome.tabs.sendMessage(tab.id, {
        type: "GET_SELECTED_TEXT"
    });

    return response?.text || "";
}

async function analyzeSelection() {
    setError("");
    setVisible(result, false);
    setVisible(loading, true);
    analyzeButton.disabled = true;
    connectionStatus.textContent = "Backend status: connecting...";

    try {
        const selectedText = await getSelectedText();

        if (!selectedText) {
            throw new Error("Select some text on the webpage first.");
        }

        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: selectedText })
        });

        if (!response.ok) {
            throw new Error(`Backend returned HTTP ${response.status}.`);
        }

        const data = await response.json();
        const percentage = data.depression_probability * 100;

        probability.textContent = `${percentage.toFixed(2)}%`;
        emotion.textContent = data.emotion;
        riskText.textContent = data.risk_level;
        riskLevel.textContent = data.risk_level;
        message.textContent = data.message;
        setRiskClass(data.risk_level);
        connectionStatus.textContent = "Backend status: connected";
        setVisible(result, true);
    } catch (requestError) {
        connectionStatus.textContent = "Backend status: unavailable";
        setError(requestError.message || "Could not analyze the selected text.");
    } finally {
        setVisible(loading, false);
        analyzeButton.disabled = false;
    }
}

analyzeButton.addEventListener("click", analyzeSelection);
