// face-capture.js - Advanced Face Capture & Analysis

class FaceCapture {
    constructor() {
        this.stream = null;
        this.video = null;
        this.canvas = null;
        this.isCapturing = false;
    }

    // Initialize camera
    async initCamera(videoElement) {
        try {
            this.video = videoElement;
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                },
                audio: false
            });
            
            this.stream = stream;
            this.video.srcObject = stream;
            
            return true;
        } catch (error) {
            console.error('Camera initialization error:', error);
            return false;
        }
    }

    // Stop camera
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
    }

    // Capture frame
    captureFrame(canvasElement = null) {
        if (!this.video) return null;
        
        const canvas = canvasElement || document.createElement('canvas');
        canvas.width = this.video.videoWidth;
        canvas.height = this.video.videoHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(this.video, 0, 0);
        
        return canvas.toDataURL('image/jpeg');
    }

    // Check face quality
    async checkQuality(imageData) {
        // This would typically use a more sophisticated algorithm
        // For now, we return a placeholder
        return {
            quality: Math.random() * 0.5 + 0.5, // 0.5-1.0
            brightness: this.checkBrightness(imageData),
            focus: this.checkFocus(imageData)
        };
    }

    // Check brightness
    checkBrightness(canvas) {
        const ctx = canvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        
        let brightness = 0;
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            brightness += (r + g + b) / 3;
        }
        
        return brightness / (data.length / 4);
    }

    // Check focus (using Laplacian variance)
    checkFocus(canvas) {
        const ctx = canvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        
        // Simplified focus check
        let variance = 0;
        for (let i = 0; i < data.length - 4; i += 4) {
            const diff = Math.abs(data[i] - data[i + 4]);
            variance += diff * diff;
        }
        
        return variance / (data.length / 4);
    }

    // Get face coordinates (placeholder)
    async detectFace(imageData) {
        // This would use face detection library
        // Placeholder implementation
        return {
            x: 100,
            y: 100,
            width: 200,
            height: 200,
            confidence: 0.95
        };
    }

    // Draw face rectangle
    drawFaceRect(canvasElement, faceData) {
        const ctx = canvasElement.getContext('2d');
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.strokeRect(faceData.x, faceData.y, faceData.width, faceData.height);
    }

    // Get image as base64
    getBase64(canvasElement) {
        return canvasElement.toDataURL('image/jpeg');
    }

    // Get image as blob
    async getBlob(canvasElement) {
        return new Promise(resolve => {
            canvasElement.toBlob(resolve, 'image/jpeg');
        });
    }

    // Check if face is centered
    isFaceCentered(faceData, canvas) {
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const faceCenter = {
            x: faceData.x + faceData.width / 2,
            y: faceData.y + faceData.height / 2
        };
        
        const threshold = 100;
        return Math.abs(faceCenter.x - centerX) < threshold && 
               Math.abs(faceCenter.y - centerY) < threshold;
    }

    // Check if face fills frame properly
    isFaceProperSize(faceData, canvas) {
        const faceArea = faceData.width * faceData.height;
        const canvasArea = canvas.width * canvas.height;
        const ratio = faceArea / canvasArea;
        
        // Face should be between 20% and 50% of frame
        return ratio > 0.2 && ratio < 0.5;
    }
}

// Create global instance
const faceCapture = new FaceCapture();

// Export for use in templates
window.faceCapture = faceCapture;