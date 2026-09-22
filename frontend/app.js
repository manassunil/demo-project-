const API_URL = "http://127.0.0.1:8000/api/analyze";

const textInput = document.getElementById("textInput");
const analyzeButton = document.getElementById("analyzeButton");

const characterCount = document.getElementById("characterCount");

const loading = document.getElementById("loading");
const error = document.getElementById("error");
const result = document.getElementById("result");

const probability = document.getElementById("probability");
const progress = document.getElementById("progress");

const emotion = document.getElementById("emotion");
const riskLevel = document.getElementById("riskLevel");
const riskText = document.getElementById("riskText");

const message = document.getElementById("message");


textInput.addEventListener("input", () => {

    characterCount.textContent =
        `${textInput.value.length} / 5000`;

});


function hide(element) {
    element.classList.add("hidden");
}


function show(element) {
    element.classList.remove("hidden");
}


function setRiskClass(element, risk) {

    element.classList.remove(
        "risk-low",
        "risk-medium",
        "risk-high"
    );

    element.classList.add(`risk-${risk}`);
}


async function analyzeText() {

    const text = textInput.value.trim();

    hide(error);
    hide(result);

    if (!text) {

        error.textContent =
            "Please enter some text before analyzing.";

        show(error);

        return;
    }


    analyzeButton.disabled = true;

    analyzeButton.textContent = "Analyzing...";

    show(loading);


    try {

        const response = await fetch(API_URL, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                text: text
            })

        });


        if (!response.ok) {

            throw new Error(
                `Server returned ${response.status}`
            );

        }


        const data = await response.json();


        const percentage =
            data.depression_probability * 100;


        probability.textContent =
            `${percentage.toFixed(2)}%`;

        progress.style.width =
            `${percentage}%`;


        emotion.textContent =
            data.emotion;


        riskText.textContent =
            data.risk_level;


        riskLevel.textContent =
            data.risk_level;


        setRiskClass(
            riskLevel,
            data.risk_level
        );


        message.textContent =
            data.message;


        hide(loading);

        show(result);

    }

    catch (err) {

        hide(loading);

        error.textContent =
            `Could not connect to MindLens AI: ${err.message}`;

        show(error);

    }

    finally {

        analyzeButton.disabled = false;

        analyzeButton.textContent =
            "Analyze Text";

    }

}


analyzeButton.addEventListener(
    "click",
    analyzeText
);