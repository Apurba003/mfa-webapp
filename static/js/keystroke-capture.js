// keystroke_capture.js - Advanced Keystroke Capture & Analysis (FIXED)

class KeystrokeCapture {
    constructor() {
        this.keystrokeData = {
            timestamps: [],
            keys: [],
            events: []
        };
        this.isCapturing = false;
        this.startTime = null;
    }

    /**
     * Start capturing keystrokes
     */
    startCapture() {
        this.keystrokeData = {
            timestamps: [],
            keys: [],
            events: []
        };
        this.isCapturing = true;
        this.startTime = performance.now();
    }

    /**
     * Stop capturing keystrokes
     */
    stopCapture() {
        this.isCapturing = false;
        return this.keystrokeData;
    }

    /**
     * Add keydown event
     * @param {string} key - The key that was pressed
     */
    addKeyDown(key) {
        if (this.isCapturing) {
            try {
                const timestamp = performance.now() - this.startTime;
                this.keystrokeData.timestamps.push(timestamp);
                this.keystrokeData.keys.push(key);
                this.keystrokeData.events.push('keydown');
            } catch (error) {
                console.error('Error adding keydown:', error);
            }
        }
    }

    /**
     * Add keyup event
     * @param {string} key - The key that was released
     */
    addKeyUp(key) {
        if (this.isCapturing) {
            try {
                const timestamp = performance.now() - this.startTime;
                this.keystrokeData.timestamps.push(timestamp);
                this.keystrokeData.keys.push(key);
                this.keystrokeData.events.push('keyup');
            } catch (error) {
                console.error('Error adding keyup:', error);
            }
        }
    }

    /**
     * Calculate dwell time (how long each key is held)
     * @returns {array} Array of dwell times in milliseconds
     */
    getDwellTime() {
        try {
            let dwellTimes = [];
            
            for (let i = 0; i < this.keystrokeData.events.length - 1; i++) {
                if (this.keystrokeData.events[i] === 'keydown' && 
                    this.keystrokeData.events[i + 1] === 'keyup') {
                    
                    const dwell = this.keystrokeData.timestamps[i + 1] - this.keystrokeData.timestamps[i];
                    
                    if (dwell >= 0 && dwell < 5000) { // Valid range: 0-5000ms
                        dwellTimes.push(dwell);
                    }
                }
            }
            
            return dwellTimes.length > 0 ? dwellTimes : [0];
        } catch (error) {
            console.error('Error calculating dwell time:', error);
            return [0];
        }
    }

    /**
     * Calculate flight time (gap between consecutive key presses)
     * @returns {array} Array of flight times in milliseconds
     */
    getFlightTime() {
        try {
            let flightTimes = [];
            
            for (let i = 0; i < this.keystrokeData.events.length - 1; i++) {
                if (this.keystrokeData.events[i] === 'keyup' && 
                    this.keystrokeData.events[i + 1] === 'keydown') {
                    
                    const flight = this.keystrokeData.timestamps[i + 1] - this.keystrokeData.timestamps[i];
                    
                    if (flight >= 0 && flight < 5000) { // Valid range: 0-5000ms
                        flightTimes.push(flight);
                    }
                }
            }
            
            return flightTimes.length > 0 ? flightTimes : [0];
        } catch (error) {
            console.error('Error calculating flight time:', error);
            return [0];
        }
    }

    /**
     * Calculate typing speed (words per minute)
     * Assumes average word length of 5 characters
     * @returns {number} Typing speed in WPM
     */
    getTypingSpeed() {
        try {
            if (this.keystrokeData.timestamps.length < 2) {
                return 0;
            }
            
            const totalTime = this.keystrokeData.timestamps[this.keystrokeData.timestamps.length - 1] - 
                             this.keystrokeData.timestamps[0];
            
            if (totalTime <= 0) {
                return 0;
            }
            
            const keyCount = this.keystrokeData.events.filter(e => e === 'keydown').length;
            const wpm = (keyCount / 5) * (60000 / totalTime);
            
            return Math.max(0, Math.round(wpm * 100) / 100);
        } catch (error) {
            console.error('Error calculating typing speed:', error);
            return 0;
        }
    }

    /**
     * Calculate rhythm (inter-key intervals between consecutive key presses)
     * @returns {array} Array of inter-key intervals in milliseconds
     */
    getRhythm() {
        try {
            let intervals = [];
            let keyDownTimes = [];
            
            // Collect all keydown timestamps
            for (let i = 0; i < this.keystrokeData.events.length; i++) {
                if (this.keystrokeData.events[i] === 'keydown') {
                    keyDownTimes.push(this.keystrokeData.timestamps[i]);
                }
            }
            
            // Calculate intervals between consecutive keypresses
            for (let i = 1; i < keyDownTimes.length; i++) {
                const interval = keyDownTimes[i] - keyDownTimes[i - 1];
                
                if (interval >= 0 && interval < 5000) { // Valid range
                    intervals.push(interval);
                }
            }
            
            return intervals.length > 0 ? intervals : [0];
        } catch (error) {
            console.error('Error calculating rhythm:', error);
            return [0];
        }
    }

    /**
     * Calculate variance (measure of inconsistency in dwell times)
     * @returns {number} Variance value
     */
    getVariance() {
        try {
            const dwellTimes = this.getDwellTime();
            
            if (dwellTimes.length < 2) {
                return 0;
            }
            
            const mean = this.getMean(dwellTimes);
            const variance = dwellTimes.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / dwellTimes.length;
            
            return Math.round(variance * 100) / 100;
        } catch (error) {
            console.error('Error calculating variance:', error);
            return 0;
        }
    }

    /**
     * Calculate entropy (uniqueness of typing pattern)
     * @returns {number} Entropy value
     */
    getEntropy() {
        try {
            const dwellTimes = this.getDwellTime();
            
            if (dwellTimes.length < 2) {
                return 0;
            }
            
            // Normalize to 0-1
            const minVal = Math.min(...dwellTimes);
            const maxVal = Math.max(...dwellTimes);
            const range = maxVal - minVal;
            
            if (range === 0) {
                return 0;
            }
            
            const normalized = dwellTimes.map(v => (v - minVal) / range);
            
            // Bin into 10 intervals
            const bins = new Array(10).fill(0);
            normalized.forEach(val => {
                const binIndex = Math.floor(val * 10);
                bins[Math.min(binIndex, 9)]++;
            });
            
            // Calculate Shannon entropy
            let entropy = 0;
            const total = dwellTimes.length;
            
            for (let count of bins) {
                if (count > 0) {
                    const probability = count / total;
                    entropy -= probability * Math.log2(probability);
                }
            }
            
            return Math.round(entropy * 100) / 100;
        } catch (error) {
            console.error('Error calculating entropy:', error);
            return 0;
        }
    }

    /**
     * Get all keystroke statistics
     * @returns {object} Complete statistics object
     */
    getStatistics() {
        try {
            const dwellTimes = this.getDwellTime();
            const flightTimes = this.getFlightTime();
            const rhythmTimes = this.getRhythm();
            
            return {
                dwell_time_mean: this.getMean(dwellTimes),
                dwell_time_std: this.getStdDev(dwellTimes),
                flight_time_mean: this.getMean(flightTimes),
                flight_time_std: this.getStdDev(flightTimes),
                typing_speed: this.getTypingSpeed(),
                rhythm_mean: this.getMean(rhythmTimes),
                rhythm_std: this.getStdDev(rhythmTimes),
                variance: this.getVariance(),
                entropy: this.getEntropy(),
                total_time: this.keystrokeData.timestamps.length > 0 ? 
                           this.keystrokeData.timestamps[this.keystrokeData.timestamps.length - 1] : 0,
                key_count: this.keystrokeData.events.filter(e => e === 'keydown').length
            };
        } catch (error) {
            console.error('Error getting statistics:', error);
            return {};
        }
    }

    /**
     * Helper: Calculate mean (average)
     * @param {array} arr - Array of numbers
     * @returns {number} Mean value
     */
    getMean(arr) {
        try {
            if (!arr || arr.length === 0) {
                return 0;
            }
            
            const sum = arr.reduce((a, b) => a + b, 0);
            return Math.round((sum / arr.length) * 100) / 100;
        } catch (error) {
            console.error('Error calculating mean:', error);
            return 0;
        }
    }

    /**
     * Helper: Calculate standard deviation
     * @param {array} arr - Array of numbers
     * @returns {number} Standard deviation value
     */
    getStdDev(arr) {
        try {
            if (!arr || arr.length === 0) {
                return 0;
            }
            
            const mean = this.getMean(arr);
            const squareDiffs = arr.map(val => Math.pow(val - mean, 2));
            const avgSquareDiff = squareDiffs.reduce((a, b) => a + b, 0) / arr.length;
            const stdDev = Math.sqrt(avgSquareDiff);
            
            return Math.round(stdDev * 100) / 100;
        } catch (error) {
            console.error('Error calculating standard deviation:', error);
            return 0;
        }
    }

    /**
     * Get raw keystroke data
     * @returns {object} Raw keystroke data
     */
    getRawData() {
        try {
            return {
                timestamps: [...this.keystrokeData.timestamps],
                keys: [...this.keystrokeData.keys],
                events: [...this.keystrokeData.events]
            };
        } catch (error) {
            console.error('Error getting raw data:', error);
            return { timestamps: [], keys: [], events: [] };
        }
    }

    /**
     * Clear all keystroke data
     */
    clearData() {
        this.keystrokeData = {
            timestamps: [],
            keys: [],
            events: []
        };
        this.isCapturing = false;
        this.startTime = null;
    }

    /**
     * Check if enough data has been captured
     * @param {number} minKeystrokes - Minimum required keystrokes
     * @returns {boolean} True if enough data
     */
    hasEnoughData(minKeystrokes = 10) {
        try {
            const keyCount = this.keystrokeData.events.filter(e => e === 'keydown').length;
            return keyCount >= minKeystrokes;
        } catch (error) {
            console.error('Error checking data:', error);
            return false;
        }
    }

    /**
     * Get capture status as string
     * @returns {string} Status information
     */
    getStatus() {
        try {
            const keyCount = this.keystrokeData.events.filter(e => e === 'keydown').length;
            const totalTime = this.keystrokeData.timestamps.length > 0 ? 
                             this.keystrokeData.timestamps[this.keystrokeData.timestamps.length - 1] : 0;
            
            return {
                isCapturing: this.isCapturing,
                keyCount: keyCount,
                totalTime: Math.round(totalTime),
                hasEnoughData: this.hasEnoughData(),
                statistics: this.getStatistics()
            };
        } catch (error) {
            console.error('Error getting status:', error);
            return {};
        }
    }

    /**
     * Validate keystroke data
     * @returns {object} Validation result
     */
    validate() {
        try {
            const issues = [];
            
            // Check minimum keystrokes
            const keyCount = this.keystrokeData.events.filter(e => e === 'keydown').length;
            if (keyCount < 5) {
                issues.push('Not enough keystrokes (minimum 5 required)');
            }
            
            // Check data consistency
            if (this.keystrokeData.timestamps.length === 0) {
                issues.push('No timestamp data');
            }
            
            if (this.keystrokeData.keys.length === 0) {
                issues.push('No key data');
            }
            
            if (this.keystrokeData.events.length === 0) {
                issues.push('No event data');
            }
            
            // Check for monotonic timestamps
            for (let i = 1; i < this.keystrokeData.timestamps.length; i++) {
                if (this.keystrokeData.timestamps[i] < this.keystrokeData.timestamps[i - 1]) {
                    issues.push('Non-monotonic timestamps detected');
                    break;
                }
            }
            
            return {
                isValid: issues.length === 0,
                issues: issues,
                keyCount: keyCount
            };
        } catch (error) {
            console.error('Error validating keystroke data:', error);
            return {
                isValid: false,
                issues: ['Validation error: ' + error.message],
                keyCount: 0
            };
        }
    }
}

/**
 * Create global keystroke capture instance
 */
const keystrokeCapture = new KeystrokeCapture();

/**
 * Export keystroke capture for use in templates and other scripts
 */
window.keystrokeCapture = keystrokeCapture;
window.KeystrokeCapture = KeystrokeCapture;

console.log('✓ keystroke-capture.js loaded successfully');