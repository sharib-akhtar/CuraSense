// Handle emotion submission and move to next step
window.submitEmotion = function () {
  const text = document.getElementById('emotionText').value.trim();
  if (!text) {
    alert('Please enter how you’re feeling.');
    return;
  }
  fetch('/detect_emotion', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  })
  .then(res => {
    if (!res.ok) throw new Error('Server error while detecting emotion');
    return res.json();
  })
  .then(data => {
    console.log('Detected Emotion:', data.emotion);
     if (data.emergency) {
      document.getElementById("emergencyModal").style.display = "block";
    }
    document.getElementById('step1').classList.add('hidden');
    document.getElementById('step2').classList.remove('hidden');
    window.detectedEmotion = data.emotion;
  })
  .catch(err => {
    console.error('Error:', err);
    alert('Failed to detect emotion. Make sure the backend is running.');
  });
};
window.closeEmergencyModal = function () {
  document.getElementById("emergencyModal").style.display = "none";
};
// Handle intent form submission
document.getElementById('intentForm').addEventListener('submit', function (e) {
  e.preventDefault();
  const intent = document.querySelector('input[name="intent"]:checked');
  if (!intent) return alert('Please select an intent.');

  window.userIntent = intent.value;
  document.getElementById('step2').classList.add('hidden');
  document.getElementById('step3').classList.remove('hidden');
});

// Handle interests form and get recommendations
document.getElementById('interestForm').addEventListener('submit', function (e) {
  e.preventDefault();
  const checked = Array.from(document.querySelectorAll('input[name="interest"]:checked'));
  if (checked.length === 0) return alert('Please select at least one interest.');

  const interests = checked.map(i => i.value);

  fetch('/get_recommendations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      intent: window.userIntent,
      interests: interests,
      emotion: window.detectedEmotion
    })
  })
  .then(res => {
    if (!res.ok) throw new Error('Server error while getting recommendations');
    return res.json();
  })
  .then(data => {
    document.getElementById('step3').classList.add('hidden');
    document.getElementById('step4').classList.remove('hidden');

    const resultDiv = document.getElementById('recommendationResults');
    resultDiv.innerHTML = data.recommendations.map(r => `<p>${r}</p>`).join('');
  })
  .catch(err => {
    console.error('Error:', err);
    alert('Failed to fetch recommendations. Ensure backend is working.');
  });
});
