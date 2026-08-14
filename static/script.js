// --- Screen Routing System ---

// Changes visual card layout from Login to Signup view tracking styles
function switchAuth(target) {
    // Collects DOM element references for the authentication blocks
    const loginCard = document.getElementById('login-card');
    const signupCard = document.getElementById('signup-card');

    if (target === 'signup') {
        // Manipulates layout visibility classes purely for frontend appearance
        loginCard.classList.add('hidden'); // Hides the card  !! - Hidden class already present in css
        signupCard.classList.remove('hidden'); // Reveals the card
        signupCard.classList.add('animate__animated', 'animate__fadeIn'); // animates the card
    } else {
        // Reverts layout classes cleanly to reveal original login box
        signupCard.classList.add('hidden'); // Hides the card
        loginCard.classList.remove('hidden'); // Reveals the card
        loginCard.classList.add('animate__animated', 'animate__fadeIn'); // animates the card
    }
}

// Intercepts gate submission purely to toggle layout panel layers visually
function handleAuth(event, type) {
    // Halts page refresh to let design toggle animations load cleanly
    event.preventDefault(); 
    
    // Updates visibility rules on the gate container interface box
    document.getElementById('auth-gate').classList.add('hidden');
    
    // Grabs active dashboard element node to clear out hidden styles
    const dashboard = document.getElementById('main-dashboard');
    dashboard.classList.remove('hidden');
    
    // Repositions canvas elements to present tracking flow top-down smoothly
    document.body.style.alignItems = 'flex-start';
    document.body.style.paddingTop = '40px';
}

// Completely resets visibility classes back to default startup login status
function logout() {
    // Adjusts visual interface tracking arrays back to secure gated setup
    document.getElementById('main-dashboard').classList.add('hidden');
    document.getElementById('auth-gate').classList.remove('hidden');
    
    // Returns absolute grid positioning back to centered splash view alignment
    document.body.style.alignItems = 'center';
    document.body.style.paddingTop = '0px';
    
    // Visual wipe of active text values across input fields
    document.getElementById('prediction-form').reset();
}

// --- Frontend Visual Overlay Controls ---

// Manages the visual transition of the popup window interface layers
// --- Updated runInference Function ---
async function runInference(event) {
    // 1. Prevent default form reload
    event.preventDefault();

    // 2. DOM Selectors
    const modal = document.getElementById('prediction-modal');
    const loader = modal.querySelector('.loader-wrapper');
    const results = document.getElementById('modal-results');
    const outputElement = document.getElementById('python-output');

    // 3. Extract Input Values using HTML element IDs
    const formData = {
        recency: document.getElementById('recency').value,
        frequency: document.getElementById('frequency').value,
        tenure: document.getElementById('tenure').value,
        total_historical_spend: document.getElementById('total_historical_spend').value,
        avg_order_value: document.getElementById('avg_order_value').value,
        total_items_bought: document.getElementById('total_items_bought').value
    };

    // 4. Reveal Modal & Loader Spinner
    modal.classList.remove('hidden');
    loader.classList.remove('hidden');
    results.classList.add('hidden');

    try {
        // 5. Send Async AJAX Request to Flask Endpoint
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (data.status === 'success') {
            // 6. Update Output Text with Predicted Spend Value
            outputElement.innerText = `$${data.predicted_spend.toFixed(2)}`;
        } else {
            outputElement.innerText = 'Error in prediction';
        }
    } catch (error) {
        console.error('Error fetching prediction:', error);
        outputElement.innerText = 'Server Error';
    } finally {
        // 7. Hide Loader and Display Results
        loader.classList.add('hidden');
        results.classList.remove('hidden');
    }
}

// Safety trigger to close structural card view frame elements
function closeModal() {
    // Injects hide class rules back onto the structural backdrop overlay block
    document.getElementById('prediction-modal').classList.add('hidden');
}