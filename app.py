
import os
import zipfile
import torch
import requests
import transformers
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification


app = Flask(__name__)
CORS(app)

# Emotion to uplifting content mapping (fallback if APIs fail)
UPLIFTING_CONTENT = {
    "sadness": [
        "Inspirational video: 'The Power of Hope' - https://example.com/hope",
        "Comedy special to lift your mood: 'Laugh Again'",
        "Guided meditation for healing - 10 min session"
    ],
    "anger": [
        "Calming breathing exercise - Box Breathing Technique",
        "Anger management podcast episode",
        "Stress-relief coloring book pages"
    ],
    "fear": [
        "Soothing nature sounds playlist",
        "Grounding technique: 5-4-3-2-1 method",
        "Motivational talk: 'Facing Your Fears'"
    ],
    "loneliness": [
        "Virtual community meetup event link",
        "Heartwarming movie recommendations",
        "Volunteering opportunities nearby"
    ],
    "disappointment": [
        "Failure-to-success stories compilation",
        "Growth mindset workshop recording",
        "Interactive journaling prompts"
    ]
}

# Initialize model variables
tokenizer = None
model = None
device = None

# Free APIs we'll use for real-time content
CONTENT_APIS = {
    "inspirational": "https://zenquotes.io/api/random",
    "jokes": "https://v2.jokeapi.dev/joke/Any",
    "meditation": "https://mettaton-api.vercel.app/api/random",
    "quotes": "https://api.quotable.io/random",
    "activities": "https://www.boredapi.com/api/activity"
}

# Mapping of emotions to API content types
EMOTION_TO_API_MAPPING = {
    "sadness": ["inspirational", "quotes"],
    "anger": ["meditation", "jokes"],
    "fear": ["meditation", "inspirational"],
    "loneliness": ["activities", "quotes"],
    "disappointment": ["inspirational", "quotes"],
    "joy": ["jokes", "activities"],
    "gratitude": ["quotes", "inspirational"],
    "neutral": ["activities", "jokes"]
}


def load_model():
    global tokenizer, model, device

    zip_path = "C:\\Users\\SHARIB\\Downloads\\emotion-model.zip"
    extract_dir = "D:\\final_year-project\\emotion-model"

    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Model unzipped to:", extract_dir)
    else:
        print("Model already unzipped.")

    model_path = os.path.join(extract_dir, "content", "emotion-model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()


def predict_emotions(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits).squeeze().tolist()
    probs_array = np.array(probs)
    transformed_probs = (probs_array ** 0.5) * 10
    return map_emotion_to_intensity(transformed_probs.tolist())


def map_emotion_to_intensity(intensities):
    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness',
        'optimism', 'pride', 'realization', 'relief', 'remorse',
        'sadness', 'surprise', 'neutral'
    ]

    emotion_intensity_pairs = list(zip(emotion_labels, intensities))
    sorted_emotions = sorted(emotion_intensity_pairs, key=lambda x: x[1], reverse=True)

    print("Top Emotions (sorted by intensity):")
    for emotion, intensity in sorted_emotions:
        print(f"{emotion}: {intensity:.4f}")
    return sorted_emotions


def fetch_real_time_content(emotion):

    api_types = EMOTION_TO_API_MAPPING.get(emotion, ["quotes", "activities"])
    content = []

    for api_type in api_types:
        try:
            if api_type == "inspirational":
                response = requests.get(CONTENT_APIS["inspirational"])
                data = response.json()
                if data and isinstance(data, list):
                    item = data[0]
                    content.append(f"Inspirational quote: {item['q']} - {item['a']}")

            elif api_type == "jokes":
                response = requests.get(CONTENT_APIS["jokes"])
                data = response.json()
                if data.get("joke"):
                    content.append(f"Joke: {data['joke']}")
                elif data.get("setup"):
                    content.append(f"Joke: {data['setup']}... {data['delivery']}")

            elif api_type == "meditation":
                response = requests.get(CONTENT_APIS["meditation"])
                data = response.json()
                if data.get("title"):
                    content.append(f"Meditation: {data['title']} - {data['url']}")

            elif api_type == "quotes":
                response = requests.get(CONTENT_APIS["quotes"])
                data = response.json()
                if data.get("content"):
                    content.append(f"Quote: {data['content']} - {data['author']}")

            elif api_type == "activities":
                response = requests.get(CONTENT_APIS["activities"])
                data = response.json()
                if data.get("activity"):
                    content.append(f"Activity suggestion: {data['activity']}")

        except Exception as e:
            print(f"Error fetching {api_type} content: {str(e)}")
            # Fall back to local content if API fails
            content.extend(UPLIFTING_CONTENT.get(emotion, []))

    # Ensure we return at least 3 items
    if len(content) < 3:
        content.extend(UPLIFTING_CONTENT.get(emotion, []))

    return content[:3]  # Return top 3 recommendations
CUSTOM_RECOMMENDATIONS = {
    "sadness": {
        "comfort": {
            "music": [
                "🎵 Playlist: 'Healing Acoustic Vibes' on Spotify",
                "🎧 Guided meditation with calming background music"
            ],
            "articles": [
                "📖 Article: 'Coping with sadness in healthy ways'",
                "📰 Self-care guide for emotional balance"
            ],
            "videos": [
                "🎥 Uplifting short film: 'The Present'",
                "📺 Relaxing nature video loop"
            ],
            "quotes": [
                "💬 'This too shall pass.'",
                "💬 'Even the darkest night will end and the sun will rise.'"
            ]
        },
        "motivation": {
            "music": [
                "🎶 Playlist: 'Rise & Shine' - Motivational Tracks",
                "🎧 Motivational speeches with instrumental beats"
            ],
            "articles": [
                "📘 How to turn pain into purpose",
                "📖 Real story: Beating the odds"
            ],
            "videos": [
                "🎥 TED Talk: 'The Power of Emotional Courage'",
                "📺 YouTube: 'Motivation in 5 Minutes'"
            ],
            "quotes": [
                "💬 'You are stronger than you think.'",
                "💬 'Stars can’t shine without darkness.'"
            ]
        },
        "distraction": {
            "music": [
                "🎧 Lo-fi chill beats",
                "🎶 Mood-lifting upbeat music"
            ],
            "articles": [
                "📄 Fun reads: Top 10 positive news stories",
                "📚 Fictional short stories to escape"
            ],
            "videos": [
                "🐶 Cute dog compilation",
                "😂 Comedy skits that make you laugh"
            ],
            "quotes": [
                "💬 'Smile, breathe, and go slowly.' - Thich Nhat Hanh",
                "💬 'Don’t let your mind bully your body.'"
            ]
        }
    },

    "anger": {
        "comfort": {
            "music": ["🎵 Calming piano melodies", "🧘 Meditation music for grounding"],
            "articles": ["📖 How to cool down mindfully", "💡 Anger management tips"],
            "videos": ["📺 Breathing exercise demo", "🎥 ASMR stress-relief content"],
            "quotes": ["💬 'Speak when you’re calm, not when you’re angry.'"]
        },
        "motivation": {
            "music": ["🔥 Workout playlist to channel energy"],
            "articles": ["📘 From rage to results: Channeling anger"],
            "videos": ["🎥 'The Mindset of Champions' motivational clip"],
            "quotes": ["💬 'Use anger as fuel, not fire.'"]
        },
        "distraction": {
            "music": ["🎶 Comedy parody songs"],
            "articles": ["📰 Fun facts you didn’t know"],
            "videos": ["🐱 Funny cat videos", "😂 Stand-up comedy"],
            "quotes": ["💬 'Laughter is the shock absorber that eases the blows of life.'"]
        }
    },

    "neutral": {
        "comfort": {
            "music": ["🎵 Lo-fi instrumental playlist"],
            "articles": ["📄 5-minute mindfulness read"],
            "videos": ["📺 Peaceful scenery video"],
            "quotes": ["💬 'Balance is not something you find, it's something you create.'"]
        },
        "motivation": {
            "music": ["🎶 Inspirational background music"],
            "articles": ["📘 Goal-setting for growth"],
            "videos": ["🎥 'Your time is now' - speech clip"],
            "quotes": ["💬 'Believe in your infinite potential.'"]
        },
        "distraction": {
            "music": ["🎧 Random music shuffle"],
            "articles": ["📚 Random Wikipedia article of the day"],
            "videos": ["🎬 Unexpected plot twist scenes"],
            "quotes": ["💬 'When in doubt, take a break and laugh.'"]
        }
    }

    # You can expand for emotions like 'fear', 'disappointment', 'joy', etc.
}

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    try:
        data = request.get_json()
        text = data.get('text', '')
        emotion_results = predict_emotions(text)

        # Get top emotion for general usage
        top_emotion = max(emotion_results, key=lambda x: x[1])[0]

        # Emergency check
        emergency = False
        sadness_score = next((i for emo, i in emotion_results if emo == "sadness"), 0)
        disappointment_score = next((i for emo, i in emotion_results if emo == "disappointment"), 0)

        if sadness_score >= 8 and disappointment_score >= 5:
            emergency = True

        return jsonify({
            "status": "success",
            "emotion": top_emotion,
            "all_emotions": [
                {"emotion": emotion, "intensity": round(intensity, 2)}
                for emotion, intensity in emotion_results
            ],
            "emergency": emergency
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        emotion = data.get("emotion")
        intent = data.get("intent")
        interests = data.get("interests", [])

        if not emotion or not intent or not interests:
            return jsonify({
                "status": "error",
                "message": "Emotion, intent, or interests missing"
            }), 400

        print(f"[INFO] Emotion: {emotion}, Intent: {intent}, Interests: {interests}")

        # Fetch from map
        recommendations = []

        for interest in interests:
            interest_recos = CUSTOM_RECOMMENDATIONS.get(emotion, {}).get(intent, {}).get(interest, [])
            recommendations.extend(interest_recos)

        # Fallbacks
        if not recommendations:
            recommendations = fetch_real_time_content(emotion)

        return jsonify({
            "status": "success",
            "recommendations": recommendations[:5]  # limit to 5
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

"""
@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        text = data.get('text', '')

        # Get emotion predictions
        emotion_results = predict_emotions(text)

        # Filter for high-intensity emotions (>5)
        high_intensity = [(emotion, intensity) for emotion, intensity in emotion_results
                          if intensity > 5 and emotion in EMOTION_TO_API_MAPPING]

        # Get recommendations - try real-time APIs first
        recommendations = []
        for emotion, _ in high_intensity:
            recommendations.extend(fetch_real_time_content(emotion))

        # If no high-intensity emotions, provide general uplifting content
        if not recommendations:
            recommendations = fetch_real_time_content("neutral")

        return jsonify({
            "status": "success",
            "input_text": text,
            "all_emotions": [
                {"emotion": emotion, "intensity": round(intensity, 2)}
                for emotion, intensity in emotion_results
            ],
            "high_intensity_emotions": [
                {"emotion": emotion, "intensity": round(intensity, 2)}
                for emotion, intensity in high_intensity
            ],
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
"""
@app.route('/test_predict', methods=['GET'])
def test_predict():
    result = predict_emotions("I feel terrible and scared")
    return jsonify(result)
@app.route('/')
def home():
    return render_template('signup.html')
@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    load_model()
    app.run(debug=True)
