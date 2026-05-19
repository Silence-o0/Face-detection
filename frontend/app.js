const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const fileInput = document.getElementById("fileInput");

let stream = null;
let isRunning = false;

let latestBoxes = [];

let currentModel = null;

async function useCamera() {
    stopAll(false);

    stream = await navigator.mediaDevices.getUserMedia({
        video: {
            width: { ideal: 1280 },
            height: { ideal: 720 }
        }
    });

    video.srcObject = stream;
    isRunning = true;

    video.onloadedmetadata = () => {
        resizeCanvas();
        render();
        startLoops();
    };
}

function triggerUpload() {
    fileInput.click();
}

fileInput.onchange = (e) => {
    stopAll(false);

    const file = e.target.files[0];
    if (!file) return;

    video.srcObject = null;
    video.src = URL.createObjectURL(file);

    isRunning = true;

    video.onloadedmetadata = () => {
        resizeCanvas();
        render();
        startLoops();
    };
};

function resizeCanvas() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
}

function stopAll(full = true) {
    isRunning = false;

    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
    }

    video.srcObject = null;
    video.src = "";

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    latestBoxes = [];
}

async function loadModels() {
    const res = await fetch("http://localhost:8000/models");
    const models = await res.json();

    const container = document.getElementById("modelButtons");
    if (container) {
        container.innerHTML = "";
        models.forEach(m => {
            const btn = document.createElement("button");
            btn.innerText = m;
            btn.onclick = () => setModel(m);
            container.appendChild(btn);
        });
    }

    if (models.length > 0) {
        setModel(models[0]);
    }
}

async function setModel(name) {
    currentModel = name;

    await fetch(`http://localhost:8000/set_model/${name}`, {
        method: "POST"
    });
}

function drawBoxes() {
    ctx.strokeStyle = "#00FF00";
    ctx.lineWidth = 2;
    ctx.font = "16px Arial";
    ctx.textBaseline = "bottom";

    latestBoxes.forEach(b => {
        ctx.strokeRect(b.x, b.y, b.w, b.h);

        const label = `${b.class_name} ${b.confidence.toFixed(2)}`;
        const metrics = ctx.measureText(label);
        const textHeight = 18;
        const textY = Math.max(b.y, textHeight);

        ctx.fillStyle = "#00FF00";
        ctx.fillRect(b.x, textY - textHeight, metrics.width + 6, textHeight);

        ctx.fillStyle = "black";
        ctx.fillText(label, b.x + 3, textY - 2);
    });
}

function render() {
    if (!isRunning) return;

    if (video.readyState >= 2) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        drawBoxes();
    }

    requestAnimationFrame(render);
}

async function sendLoop() {
    if (!isRunning) return;

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;

    const tctx = tempCanvas.getContext("2d");
    tctx.drawImage(video, 0, 0);

    tempCanvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append("file", blob);

        await fetch("http://localhost:8000/frame", {
            method: "POST",
            body: formData
        });
    }, "image/jpeg", 0.7);

    setTimeout(sendLoop, 100);
}

async function resultLoop() {
    if (!isRunning) return;

    const res = await fetch("http://localhost:8000/result");
    latestBoxes = await res.json();

    setTimeout(resultLoop, 50);
}

async function metricsLoop() {
    const res = await fetch("http://localhost:8000/metrics");
    const data = await res.json();

    document.getElementById("metrics").innerText =
        `Model: ${data.model} | FPS: ${data.fps.toFixed(2)} | ${(data.inference_time * 1000).toFixed(1)} ms`;

    setTimeout(metricsLoop, 500);
}

function startLoops() {
    sendLoop();
    resultLoop();
    metricsLoop();
}

loadModels();
