/*
================================================================================
NEUROGUARD AI - MAIN JAVASCRIPT
Clinical-Grade Epilepsy Prediction System
================================================================================
*/

// ============================================
// GLOBAL UTILITIES
// ============================================

const NeuroGuard = {
    // API Base URL
    apiBase: '',
    
    // Class colors mapping
    classColors: {
        'NORMAL': { primary: '#3b82f6', bg: 'rgba(59, 130, 246, 0.2)', name: 'Blue' },
        'PREICTAL': { primary: '#eab308', bg: 'rgba(234, 179, 8, 0.2)', name: 'Yellow' },
        'ICTAL': { primary: '#ef4444', bg: 'rgba(239, 68, 68, 0.2)', name: 'Red' },
        'POSTICTAL': { primary: '#22c55e', bg: 'rgba(34, 197, 94, 0.2)', name: 'Green' }
    },
    
    // ============================================
    // API METHODS
    // ============================================
    
    async fetchAPI(endpoint, options = {}) {
        try {
            const response = await fetch(this.apiBase + endpoint, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    },
    
    async getSystemStatus() {
        return this.fetchAPI('/api/system-status');
    },
    
    async getHistory() {
        return this.fetchAPI('/api/history');
    },
    
    async clearHistory() {
        return this.fetchAPI('/api/clear-history', { method: 'POST' });
    },
    
    async uploadFile(formData) {
        const response = await fetch(this.apiBase + '/api/upload', {
            method: 'POST',
            body: formData
        });
        return response.json();
    },
    
    async predict() {
        return this.fetchAPI('/api/predict', { method: 'POST' });
    },
    
    // ============================================
    // UI UTILITIES
    // ============================================
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        const bgColor = type === 'error' ? 'bg-red-500/20 border-red-500 text-red-400' 
                      : type === 'success' ? 'bg-green-500/20 border-green-500 text-green-400'
                      : 'bg-neon-aqua/20 border-neon-aqua text-neon-aqua';
        
        notification.className = `fixed top-24 right-4 z-50 p-4 rounded-lg border ${bgColor} max-w-sm transform translate-x-full transition-transform duration-300`;
        notification.innerHTML = `
            <div class="flex items-center gap-3">
                <svg class="w-5 h-5 flex-shrink-0" fill="currentColor" viewBox="0 0 24 24">
                    ${type === 'error' 
                        ? '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>'
                        : '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>'}
                </svg>
                <span>${message}</span>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Animate in
        requestAnimationFrame(() => {
            notification.classList.remove('translate-x-full');
        });
        
        // Auto remove
        setTimeout(() => {
            notification.classList.add('translate-x-full');
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    },
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
    
    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    },
    
    // ============================================
    // ANIMATION UTILITIES
    // ============================================
    
    animateElement(element, animation, duration = 500) {
        return new Promise(resolve => {
            element.style.animation = `${animation} ${duration}ms ease-out forwards`;
            setTimeout(() => {
                element.style.animation = '';
                resolve();
            }, duration);
        });
    },
    
    animateCounter(element, start, end, duration = 1000) {
        const startTime = performance.now();
        const update = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = Math.round(start + (end - start) * easeOut);
            element.textContent = current;
            
            if (progress < 1) {
                requestAnimationFrame(update);
            }
        };
        requestAnimationFrame(update);
    },
    
    // ============================================
    // CHART UTILITIES
    // ============================================
    
    createBarChart(ctx, labels, data, colors) {
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: colors.map(c => c + '40'),
                    borderColor: colors,
                    borderWidth: 2,
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });
    },
    
    createDoughnutChart(ctx, labels, data, colors) {
        return new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: colors,
                    borderColor: '#0f172a',
                    borderWidth: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#94a3b8', padding: 15 }
                    }
                },
                cutout: '60%'
            }
        });
    },
    
    // ============================================
    // PARTICLE SYSTEM
    // ============================================
    
    createParticles(container, count = 50) {
        for (let i = 0; i < count; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.cssText = `
                position: absolute;
                width: ${Math.random() * 4 + 2}px;
                height: ${Math.random() * 4 + 2}px;
                background: ${Math.random() > 0.5 ? '#0afff2' : '#a855f7'};
                border-radius: 50%;
                left: ${Math.random() * 100}%;
                top: ${Math.random() * 100}%;
                opacity: ${Math.random() * 0.5 + 0.2};
                animation: float ${Math.random() * 4 + 6}s ease-in-out infinite;
                animation-delay: ${Math.random() * 4}s;
            `;
            container.appendChild(particle);
        }
    },
    
    // ============================================
    // STORAGE UTILITIES
    // ============================================
    
    saveToStorage(key, data) {
        try {
            sessionStorage.setItem(key, JSON.stringify(data));
        } catch (e) {
            console.error('Storage error:', e);
        }
    },
    
    getFromStorage(key) {
        try {
            const data = sessionStorage.getItem(key);
            return data ? JSON.parse(data) : null;
        } catch (e) {
            console.error('Storage error:', e);
            return null;
        }
    },
    
    removeFromStorage(key) {
        try {
            sessionStorage.removeItem(key);
        } catch (e) {
            console.error('Storage error:', e);
        }
    }
};

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    // Create particles if container exists
    const particlesContainer = document.getElementById('particles');
    if (particlesContainer) {
        NeuroGuard.createParticles(particlesContainer);
    }
    
    // Check system status
    NeuroGuard.getSystemStatus().then(data => {
        const statusDot = document.getElementById('systemStatus');
        if (statusDot) {
            if (data.success && data.status.model_loaded) {
                statusDot.className = 'status-dot status-online';
            } else {
                statusDot.className = 'status-dot status-warning';
            }
        }
    }).catch(() => {
        const statusDot = document.getElementById('systemStatus');
        if (statusDot) {
            statusDot.className = 'status-dot status-danger';
        }
    });
    
    // Mobile menu toggle
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const mobileNav = document.getElementById('mobileNav');
    
    if (mobileMenuBtn && mobileNav) {
        mobileMenuBtn.addEventListener('click', () => {
            mobileNav.classList.toggle('active');
        });
        
        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!mobileMenuBtn.contains(e.target) && !mobileNav.contains(e.target)) {
                mobileNav.classList.remove('active');
            }
        });
    }
});

// Export for global use
window.NeuroGuard = NeuroGuard;
