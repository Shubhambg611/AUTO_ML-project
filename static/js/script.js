document.getElementById('login-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const submitButton = document.getElementById('submit-button');
    const errorDisplay = document.getElementById('error-message');
    submitButton.disabled = true;
    submitButton.textContent = 'Signing in...';
    
    try {
        const formData = new FormData(this);
        const response = await fetch('/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams(formData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            window.location.href = data.redirect;
        } else {
            errorDisplay.textContent = data.message;
            errorDisplay.style.display = 'block';
        }
    } catch (error) {
        errorDisplay.textContent = 'Connection error. Please try again.';
        errorDisplay.style.display = 'block';
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = 'Sign In';
    }
});