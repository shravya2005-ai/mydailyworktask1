// JavaScript for the Movie Recommendation System

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Handle rating form submission
    const ratingForm = document.getElementById('ratingForm');
    if (ratingForm) {
        ratingForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const messageDiv = document.getElementById('ratingMessage');
            
            // Show loading state
            const submitBtn = this.querySelector('button[type="submit"]');
            const originalText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Adding...';
            submitBtn.disabled = true;
            
            fetch('/add_rating', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    messageDiv.innerHTML = `
                        <div class="alert alert-success alert-dismissible fade show" role="alert">
                            <i class="fas fa-check-circle me-2"></i>${data.message}
                            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                        </div>
                    `;
                    ratingForm.reset();
                } else {
                    messageDiv.innerHTML = `
                        <div class="alert alert-danger alert-dismissible fade show" role="alert">
                            <i class="fas fa-exclamation-circle me-2"></i>Error: ${data.message}
                            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                        </div>
                    `;
                }
            })
            .catch(error => {
                messageDiv.innerHTML = `
                    <div class="alert alert-danger alert-dismissible fade show" role="alert">
                        <i class="fas fa-exclamation-circle me-2"></i>An error occurred. Please try again.
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                `;
            })
            .finally(() => {
                // Restore button state
                submitBtn.innerHTML = originalText;
                submitBtn.disabled = false;
            });
        });
    }

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add hover effects to cards
    document.querySelectorAll('.card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });

    // Auto-hide alerts after 5 seconds
    setTimeout(function() {
        const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
        alerts.forEach(alert => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);

    // Add loading animation to buttons
    document.querySelectorAll('.btn').forEach(btn => {
        btn.addEventListener('click', function() {
            if (this.type !== 'submit' && !this.classList.contains('no-loading')) {
                const originalText = this.innerHTML;
                this.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Loading...';
                
                setTimeout(() => {
                    this.innerHTML = originalText;
                }, 1000);
            }
        });
    });

    // Initialize star ratings
    function initializeStarRatings() {
        const ratings = document.querySelectorAll('.rating');
        ratings.forEach(rating => {
            const stars = rating.querySelectorAll('.fa-star');
            stars.forEach((star, index) => {
                star.addEventListener('mouseenter', function() {
                    highlightStars(stars, index + 1);
                });
                
                star.addEventListener('mouseleave', function() {
                    resetStars(stars);
                });
                
                star.addEventListener('click', function() {
                    selectStars(stars, index + 1);
                });
            });
        });
    }

    function highlightStars(stars, count) {
        stars.forEach((star, index) => {
            if (index < count) {
                star.classList.add('text-warning');
                star.classList.remove('text-muted');
            } else {
                star.classList.add('text-muted');
                star.classList.remove('text-warning');
            }
        });
    }

    function resetStars(stars) {
        stars.forEach(star => {
            if (!star.classList.contains('selected')) {
                star.classList.add('text-muted');
                star.classList.remove('text-warning');
            }
        });
    }

    function selectStars(stars, count) {
        stars.forEach((star, index) => {
            if (index < count) {
                star.classList.add('selected', 'text-warning');
                star.classList.remove('text-muted');
            } else {
                star.classList.remove('selected', 'text-warning');
                star.classList.add('text-muted');
            }
        });
    }

    // Initialize components
    initializeStarRatings();

    // Add search functionality (if search input exists)
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            const items = document.querySelectorAll('.searchable-item');
            
            items.forEach(item => {
                const text = item.textContent.toLowerCase();
                if (text.includes(searchTerm)) {
                    item.style.display = '';
                } else {
                    item.style.display = 'none';
                }
            });
        });
    }

    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + K for search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.getElementById('searchInput');
            if (searchInput) {
                searchInput.focus();
            }
        }
        
        // Escape to close modals
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.modal.show');
            modals.forEach(modal => {
                const bsModal = bootstrap.Modal.getInstance(modal);
                if (bsModal) {
                    bsModal.hide();
                }
            });
        }
    });

    // Add progress indicators for long operations
    function showProgress(element, message = 'Processing...') {
        const progressHtml = `
            <div class="d-flex align-items-center">
                <div class="spinner-border spinner-border-sm me-2" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                ${message}
            </div>
        `;
        element.innerHTML = progressHtml;
    }

    // Utility function to format numbers
    function formatNumber(num, decimals = 1) {
        return parseFloat(num).toFixed(decimals);
    }

    // Utility function to capitalize first letter
    function capitalize(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    // Make utility functions globally available
    window.MovieRecommender = {
        showProgress,
        formatNumber,
        capitalize
    };
});