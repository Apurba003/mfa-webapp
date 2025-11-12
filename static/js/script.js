
let keystrokes = [];
let pressTimes = {};
let startTime = null;
const passwordInput = document.getElementById("passwordInput");

passwordInput.addEventListener("focus", () => {
    if (!startTime) startTime = performance.now();
});

passwordInput.addEventListener("keydown", (event) => {
    const key = event.key;
    const currentTime = performance.now() - startTime;

    // If Enter is pressed, submit instead of creating newline
    if (key === "Enter") {
        event.preventDefault();
        sendData();
        return;
    }

    // Record press time
    pressTimes[key] = currentTime;
});

passwordInput.addEventListener("keyup", (event) => {
    const key = event.key;
    const currentTime = performance.now() - startTime;

    if (pressTimes[key] !== undefined) {
        const dwell = currentTime - pressTimes[key];
        keystrokes.push({
            key: key,
            press_time: pressTimes[key].toFixed(3),
            release_time: currentTime.toFixed(3),
            dwell_time: dwell.toFixed(3),
            flight_time: 0.0
        });

        delete pressTimes[key];

        // Compute flight time for previous key
        if (keystrokes.length > 1) {
            const prev = keystrokes[keystrokes.length - 2];
            keystrokes[keystrokes.length - 2].flight_time = (
                keystrokes[keystrokes.length - 1].press_time - prev.release_time
            ).toFixed(3);
        }
    }
});

function sendData() {
    if (keystrokes.length === 0) {
        alert("No keystrokes recorded!");
        return;
    }

    fetch("/record", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            keystrokes: keystrokes
        })
    })
        .then(response => response.json())
        .then(data => {
            alert(data.message);
            passwordInput.disabled = true;
        })
        .catch(error => {
            console.error("Error:", error);
            alert("Failed to send data.");
        });
    location.reload()
}