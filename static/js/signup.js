document.getElementById('signupForm').addEventListener('submit', function (e) {
    e.preventDefault();
  
    const name = document.getElementById('name').value.trim();
    const age = document.getElementById('age').value.trim();
    const email = document.getElementById('email').value.trim();
  
    if (!name || !age || !email) {
      alert('Please fill all fields.');
      return;
    }
  
    // Save user info in localStorage
    localStorage.setItem('userName', name);
    localStorage.setItem('userAge', age);
    localStorage.setItem('userEmail', email);
  
    // Redirect to main page
     window.location.href = indexURL;
  });
  