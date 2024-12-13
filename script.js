function textToAudio() {
    let msg = document.querySelector('#result').textContent; // Use textContent to get the content
    let speech = new SpeechSynthesisUtterance();

    speech.lang = "en-US";
    speech.text = msg;
    speech.volume = 1.5;
    speech.rate = 1.5;
    speech.pitch = 1.5;

    speechSynthesis.speak(speech);
}

// script.js

// Function to handle form submission
document.getElementById('chat-form').addEventListener('submit', async function (event) {
  event.preventDefault(); // Prevent form submission

  const userInput = document.getElementById('user-input').value.trim();

  if (!userInput) {
    alert('Please enter your query!');
    return;
  }

  // Clear the input field for the next message
  document.getElementById('user-input').value = '';

  // Display the user's query
  const resultDiv = document.getElementById('result');
  const userMessage = document.createElement('div');
  userMessage.classList.add('user-message');
  userMessage.textContent = `You: ${userInput}`;
  resultDiv.appendChild(userMessage);

  try {
    // Make a POST request to the backend API
    const response = await fetch('http://127.0.0.1:5000/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question: userInput }),
    });

    if (!response.ok) {
      throw new Error('Failed to fetch response from server.');
    }

    const data = await response.json();
    const botResponse = data.response;

    // Display the bot's response
    const botMessage = document.createElement('div');
    botMessage.classList.add('bot-message');
    botMessage.textContent = `Chatbot: ${botResponse}`;
    resultDiv.appendChild(botMessage);
  } catch (error) {
    console.error('Error:', error);
    const errorMessage = document.createElement('div');
    errorMessage.classList.add('error-message');
    errorMessage.textContent = 'Something went wrong. Please try again later.';
    resultDiv.appendChild(errorMessage);
  }
});

