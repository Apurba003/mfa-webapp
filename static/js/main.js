// main.js - Main JavaScript Functions (FIXED & COMPLETE)

/**
 * Get authentication token from localStorage
 * @returns {string|null} JWT token or null if not found
 */
function getToken() {
    try {
        const accessToken = localStorage.getItem('access_token');
        const tempToken = localStorage.getItem('temp_token');
        return accessToken || tempToken || null;
    } catch (error) {
        console.error('Error getting token:', error);
        return null;
    }
}

/**
 * Make authenticated API request with JWT token
 * @param {string} url - API endpoint URL
 * @param {object} options - Fetch options
 * @returns {Promise} Fetch response
 */
async function makeRequest(url, options = {}) {
    try {
        const token = getToken();
        
        const headers = {
            'Content-Type': 'application/json',
            ...(options.headers || {})
        };
        
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        
        const response = await fetch(url, {
            ...options,
            headers
        });
        
        // Handle authentication errors
        if (response.status === 401) {
            console.warn('Unauthorized: Redirecting to login');
            localStorage.clear();
            window.location.href = '/login';
        }
        
        return response;
    } catch (error) {
        console.error('Request error:', error);
        throw error;
    }
}

/**
 * Show alert message to user
 * @param {string} message - Message to display
 * @param {string} type - Alert type (success, error, warning, info)
 * @param {number} duration - Duration to show alert (ms)
 */
function showAlert(message, type = 'info', duration = 5000) {
    try {
        const alertBox = document.getElementById('alertBox');
        if (alertBox) {
            alertBox.className = `alert-box alert-${type}`;
            alertBox.textContent = message;
            alertBox.style.display = 'block';
            
            // Auto-hide after specified duration
            if (duration > 0) {
                setTimeout(() => {
                    if (alertBox.style.display === 'block') {
                        alertBox.style.display = 'none';
                    }
                }, duration);
            }
        }
    } catch (error) {
        console.error('Error showing alert:', error);
    }
}

/**
 * Hide alert message
 */
function hideAlert() {
    try {
        const alertBox = document.getElementById('alertBox');
        if (alertBox) {
            alertBox.style.display = 'none';
        }
    } catch (error) {
        console.error('Error hiding alert:', error);
    }
}

/**
 * Logout user and clear authentication
 */
function logout() {
    try {
        if (confirm('Are you sure you want to logout?')) {
            localStorage.clear();
            sessionStorage.clear();
            window.location.href = '/login';
        }
    } catch (error) {
        console.error('Error logging out:', error);
        window.location.href = '/login';
    }
}

/**
 * Check if user is authenticated
 * @returns {boolean} True if authenticated
 */
function isAuthenticated() {
    const token = getToken();
    return token !== null && token !== undefined && token !== '';
}

/**
 * Check authentication and redirect if not authenticated
 */
function checkAuth() {
    try {
        if (!isAuthenticated()) {
            window.location.href = '/login';
        }
    } catch (error) {
        console.error('Error checking auth:', error);
        window.location.href = '/login';
    }
}

/**
 * Format date string to local date/time
 * @param {string} dateString - ISO date string
 * @returns {string} Formatted date string
 */
function formatDate(dateString) {
    try {
        if (!dateString) return 'N/A';
        const date = new Date(dateString);
        if (isNaN(date.getTime())) return 'Invalid Date';
        return date.toLocaleString();
    } catch (error) {
        console.error('Error formatting date:', error);
        return 'Invalid Date';
    }
}

/**
 * Format number with specified decimal places
 * @param {number} num - Number to format
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted number string
 */
function formatNumber(num, decimals = 2) {
    try {
        if (num === null || num === undefined || isNaN(num)) {
            return '0.00';
        }
        return parseFloat(num).toFixed(decimals);
    } catch (error) {
        console.error('Error formatting number:', error);
        return '0.00';
    }
}

/**
 * Format percentage
 * @param {number} num - Number (0-1)
 * @param {number} decimals - Decimal places
 * @returns {string} Formatted percentage
 */
function formatPercentage(num, decimals = 1) {
    try {
        if (num === null || num === undefined || isNaN(num)) {
            return '0%';
        }
        return (parseFloat(num) * 100).toFixed(decimals) + '%';
    } catch (error) {
        console.error('Error formatting percentage:', error);
        return '0%';
    }
}

/**
 * Validate email address
 * @param {string} email - Email to validate
 * @returns {boolean} True if valid email
 */
function validateEmail(email) {
    try {
        if (!email || typeof email !== 'string') {
            return false;
        }
        
        // RFC 5322 simplified email regex
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email.trim());
    } catch (error) {
        console.error('Error validating email:', error);
        return false;
    }
}

/**
 * Validate password strength
 * @param {string} password - Password to validate
 * @returns {object} Validation result
 */
function validatePassword(password) {
    try {
        const issues = [];
        
        if (!password) {
            issues.push('Password is required');
        } else {
            if (password.length < 6) {
                issues.push('Password must be at least 6 characters');
            }
            if (!/[A-Z]/.test(password)) {
                issues.push('Password must contain uppercase letter');
            }
            if (!/[a-z]/.test(password)) {
                issues.push('Password must contain lowercase letter');
            }
            if (!/[0-9]/.test(password)) {
                issues.push('Password must contain number');
            }
        }
        
        return {
            isValid: issues.length === 0,
            issues: issues,
            strength: issues.length === 0 ? 'Strong' : 'Weak'
        };
    } catch (error) {
        console.error('Error validating password:', error);
        return {
            isValid: false,
            issues: ['Validation error'],
            strength: 'Unknown'
        };
    }
}

/**
 * Get user profile data
 * @returns {Promise<object>} User profile or null
 */
async function getUserData() {
    try {
        const response = await makeRequest('/api/user/profile');
        
        if (response.ok) {
            return await response.json();
        } else if (response.status === 401) {
            // Will be handled by makeRequest
            return null;
        } else {
            console.error('Failed to get user data:', response.status);
            return null;
        }
    } catch (error) {
        console.error('Error getting user data:', error);
        return null;
    }
}

/**
 * Get user statistics
 * @returns {Promise<object>} User statistics or null
 */
async function getUserStats() {
    try {
        const response = await makeRequest('/api/user/stats');
        
        if (response.ok) {
            return await response.json();
        } else {
            console.error('Failed to get user stats:', response.status);
            return null;
        }
    } catch (error) {
        console.error('Error getting user stats:', error);
        return null;
    }
}

/**
 * Get authentication history
 * @param {number} limit - Number of records to fetch
 * @returns {Promise<array>} Authentication attempts
 */
async function getAuthHistory(limit = 20) {
    try {
        if (!Number.isInteger(limit) || limit < 1) {
            limit = 20;
        }
        
        const response = await makeRequest(`/api/user/auth-history?limit=${limit}`);
        
        if (response.ok) {
            return await response.json();
        } else {
            console.error('Failed to get auth history:', response.status);
            return [];
        }
    } catch (error) {
        console.error('Error getting auth history:', error);
        return [];
    }
}

/**
 * Update user profile
 * @param {object} profileData - Profile data to update
 * @returns {Promise<object>} Updated profile or null
 */
async function updateProfile(profileData) {
    try {
        if (!profileData || typeof profileData !== 'object') {
            console.error('Invalid profile data');
            return null;
        }
        
        // Validate email if provided
        if (profileData.email && !validateEmail(profileData.email)) {
            console.error('Invalid email address');
            return null;
        }
        
        const response = await makeRequest('/api/user/update-profile', {
            method: 'POST',
            body: JSON.stringify(profileData)
        });
        
        if (response.ok) {
            return await response.json();
        } else {
            const error = await response.json();
            console.error('Failed to update profile:', error);
            return null;
        }
    } catch (error) {
        console.error('Error updating profile:', error);
        return null;
    }
}

/**
 * Enable or disable MFA method
 * @param {string} method - MFA method (keystroke, face)
 * @param {boolean} enabled - Enable or disable
 * @returns {Promise<object>} MFA status or null
 */
async function setMFAMethod(method, enabled) {
    try {
        if (!method || typeof method !== 'string') {
            console.error('Invalid MFA method');
            return null;
        }
        
        if (typeof enabled !== 'boolean') {
            console.error('Invalid enabled parameter');
            return null;
        }
        
        const response = await makeRequest('/api/user/enable-mfa', {
            method: 'POST',
            body: JSON.stringify({ method, enabled })
        });
        
        if (response.ok) {
            return await response.json();
        } else {
            const error = await response.json();
            console.error('Failed to set MFA method:', error);
            return null;
        }
    } catch (error) {
        console.error('Error setting MFA method:', error);
        return null;
    }
}

/**
 * Register new user
 * @param {object} userData - User registration data
 * @returns {Promise<object>} Registration result
 */
async function registerUser(userData) {
    try {
        if (!userData || typeof userData !== 'object') {
            return { success: false, error: 'Invalid user data' };
        }
        
        // Validate required fields
        if (!userData.username || userData.username.length < 3) {
            return { success: false, error: 'Username must be at least 3 characters' };
        }
        
        if (!validateEmail(userData.email)) {
            return { success: false, error: 'Invalid email address' };
        }
        
        const passwordValidation = validatePassword(userData.password);
        if (!passwordValidation.isValid) {
            return { success: false, error: passwordValidation.issues.join(', ') };
        }
        
        const response = await fetch('/api/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(userData)
        });
        
        if (response.ok) {
            return { success: true, data: await response.json() };
        } else {
            const error = await response.json();
            return { success: false, error: error.error || 'Registration failed' };
        }
    } catch (error) {
        console.error('Error registering user:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Login user
 * @param {object} credentials - Username and password
 * @returns {Promise<object>} Login result
 */
async function loginUser(credentials) {
    try {
        if (!credentials || !credentials.username || !credentials.password) {
            return { success: false, error: 'Missing username or password' };
        }
        
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(credentials)
        });
        
        if (response.ok) {
            const data = await response.json();
            
            // Store temporary or access token
            if (data.temp_token) {
                localStorage.setItem('temp_token', data.temp_token);
            }
            if (data.access_token) {
                localStorage.setItem('access_token', data.access_token);
                localStorage.removeItem('temp_token');
            }
            
            return { success: true, data: data };
        } else {
            const error = await response.json();
            return { success: false, error: error.error || 'Login failed' };
        }
    } catch (error) {
        console.error('Error logging in:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Copy text to clipboard
 * @param {string} text - Text to copy
 * @returns {boolean} Success status
 */
function copyToClipboard(text) {
    try {
        navigator.clipboard.writeText(text);
        showAlert('✓ Copied to clipboard', 'success', 2000);
        return true;
    } catch (error) {
        console.error('Error copying to clipboard:', error);
        return false;
    }
}

/**
 * Get current timestamp
 * @returns {string} Current timestamp
 */
function getCurrentTimestamp() {
    return new Date().toISOString();
}

/**
 * Initialize application
 */
function initializeApp() {
    try {
        console.log('✓ Main.js loaded and initialized');
        
        // Check if user is authenticated on protected pages
        const protectedPages = ['/dashboard', '/profile', '/keystroke_enroll', '/face_enroll'];
        const currentPage = window.location.pathname;
        
        if (protectedPages.some(page => currentPage.includes(page))) {
            checkAuth();
        }
        
        // Log user info if authenticated
        if (isAuthenticated()) {
            console.log('✓ User is authenticated');
            getUserData().then(user => {
                if (user) {
                    console.log(`✓ Logged in as: ${user.username}`);
                }
            });
        }
    } catch (error) {
        console.error('Error initializing app:', error);
    }
}

/**
 * Document ready event
 */
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * Export all functions for use in templates
 * Make them available on window object
 */
window.getToken = getToken;
window.makeRequest = makeRequest;
window.showAlert = showAlert;
window.hideAlert = hideAlert;
window.logout = logout;
window.isAuthenticated = isAuthenticated;
window.checkAuth = checkAuth;
window.formatDate = formatDate;
window.formatNumber = formatNumber;
window.formatPercentage = formatPercentage;
window.validateEmail = validateEmail;
window.validatePassword = validatePassword;
window.getUserData = getUserData;
window.getUserStats = getUserStats;
window.getAuthHistory = getAuthHistory;
window.updateProfile = updateProfile;
window.setMFAMethod = setMFAMethod;
window.registerUser = registerUser;
window.loginUser = loginUser;
window.copyToClipboard = copyToClipboard;
window.getCurrentTimestamp = getCurrentTimestamp;
window.initializeApp = initializeApp;

// Log that main.js is loaded
console.log('✓ main.js loaded successfully');