from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the pre-trained chatbot model
with open('chatbot.pkl', 'rb') as model_file:
    Pipe = pickle.load(model_file)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '')
    if not question:
        return jsonify({'response': 'Please ask a valid question.'}), 400

    try:
        response = Pipe.predict([question])[0]  # Get model response
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': 'Error occurred while processing your request.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
