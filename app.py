
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

tokenizer = None
model = None
device = None
hf_token = os.getenv("HF_TOKEN")
model_name = "Ugly12021/emotion-model"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=hf_token)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

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
'''tokenizer = None
model = None
device = None'''

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

'''
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
    model_path="Ugly12021/emotion-model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    hf_token = os.getenv("HF_TOKEN")
    model_name = "Ugly12021/emotion-model"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=hf_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()'''

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
                "🎵 <a href='https://www.youtube.com/watch?v=e0X5-_rnA0g' target='_blank'>Beautiful Indian Music for Meditation and Yoga</a>",
                "🎧 <a href='https://www.youtube.com/watch?v=mr8GBzTsWqM' target='_blank'>Indian Flute Meditation Music</a>"
            ],
            "articles": [
                "📖 <a href='https://www.thetimes.co.uk/article/indian-children-taught-how-to-coexist-with-tigers-as-numbers-grow-96d62sgk0' target='_blank'>Indian children taught how to coexist with tigers as numbers grow</a>",
                "📰 <a href='https://www.indiatoday.in/good-news' target='_blank'>Good News: Inspiring Stories of Progress and Positivity in India Today</a>"
            ],
            "videos": [
                "🎥 <a href='https://www.youtube.com/watch?v=QqCZcMAC5Hk' target='_blank'>Tiny Things - Award Winning Hindi Short Film</a>",
                "📺 <a href='https://www.youtube.com/playlist?list=PLoIbxxoLFnC7FTE8a-AoQVhKh4awnPzsC' target='_blank'>Recommended Best Indian Short Films</a>"
                "🎥 <a href='https://www.primevideo.com/region/eu/detail/The-Pursuit-of-Happyness/0MW4Y2G0VYCAN9FD9NU3J3RA0U' target='_blank'>The Pursuit Of Happiness</a>",
            ],
            "quotes": [
                "💬 'This too shall pass.'",
                "💬 'Even the darkest night will end and the sun will rise.'"
            ]
        },
        "motivation": {
            "music": [
                "🎶 <a href='https://www.youtube.com/watch?v=2g4TSxxnkvA' target='_blank'>Top Motivational Songs Jukebox | Best of 2024</a>",
                "🎧 <a href='https://www.youtube.com/watch?v=OcmcptbsvzQ' target='_blank'>Feel Good | Motivational Bollywood Songs</a>"
                "🎧 <a href='https://open.spotify.com/track/7tqhbajSfrz2F7E1Z75ASX?si=fee5c0a8f57e43f1&nd=1&dlsi=65ca5ba9234d442b' target='_blank'> '‘Ain’t No Mountain High Enough’ by Marvin Gaye and Tammi Terrell'</a>"
            ],
            "articles": [
                "📘 <a href='https://www.thebetterindia.com/296369/shah-rukh-khan-dolly-singh-laxmi-agarwal-indians-give-inspiring-tedx-talks/' target='_blank'>10 Incredible TED Talks by Indians You Should Not Miss</a>",
                "📖 <a href='https://www.linkedin.com/pulse/turning-pain-purpose-how-use-lifes-challenges-fuel-positive-rivers' target='_blank'>Turning Pain into Purpose: How to Use Life's Challenges to Fuel Positive Change</a>"
            ],
            "videos": [
                "🎥 <a href='https://www.ted.com/playlists/605/the_best_of_ted_talks_india_nayi_soch_episode_8' target='_blank'>The Best of TED Talks India: Nayi Soch</a>",
                "📺 <a href='https://www.youtube.com/watch?v=Yk983AQG7rA' target='_blank'>Live for Yourself! | Chetan Bhagat | TEDxGraphicEraUniversity</a>"
            ],
            "quotes": [
                "💬 'You are stronger than you think.'",
                "💬 'Stars can’t shine without darkness.'"
            ]
        },
        "distraction": {
            "music": [
                "🎧 <a href='https://www.youtube.com/watch?v=rtTI1rh9U5M' target='_blank'>1 Hour Of Night Hindi Lofi Songs</a>",
                "🎶 <a href='https://www.youtube.com/playlist?list=PLHuHXHyLu7BGSLiq97ZYEs1m1IHJKOmgR' target='_blank'>Best Indian Lofi Hits | Sony Music India</a>"
            ],
            "articles": [
                "📄 <a href='https://timesofindia.indiatimes.com/good-news/goodnews.cms' target='_blank'>Good News - Times of India</a>",
                "📚 <a href='https://www.newindianexpress.com/good-news' target='_blank'>Good News - The New Indian Express</a>"
            ],
            "videos": [
                "🐶 <a href='https://www.youtube.com/watch?v=J---aiyznGQ' target='_blank'>Cute Dog Compilation</a>",
                "😂 <a href='https://www.youtube.com/watch?v=AgQ8RV3zn2A' target='_blank'>Attitude | Stand-up Comedy by Ravi Gupta</a>"
            ],
            "quotes": [
                "💬 'Smile, breathe, and go slowly.' - Thich Nhat Hanh",
                "💬 'Don’t let your mind bully your body.'"
            ]
        }
    },
    "anger": {
        "comfort": {
            "music": [
                "🎵 <a href='https://www.youtube.com/watch?v=Qo6eO50oxHc' target='_blank'>SHIVA | Beautiful Indian Background Music</a>",
                "🧘 <a href='https://www.youtube.com/watch?v=mr8GBzTsWqM' target='_blank'>Indian Flute Meditation Music</a>"
            ],
            "articles": [
                "📖 <a href='https://www.helpguide.org/articles/anger/anger-management.htm' target='_blank'>How to Cool Down Mindfully</a>",
                "💡 <a href='https://www.helpguide.org/articles/anger/anger-management.htm' target='_blank'>Anger Management Tips</a>"
            ],
            "videos": [
                "📺 <a href='https://www.youtube.com/watch?v=QZbuj3RJcjI' target='_blank'>Breathing Exercise Demo</a>",
                "🎥 <a href='https://www.youtube.com/watch?v=QZbuj3RJcjI' target='_blank'>ASMR Stress-Relief Content</a>"
            ],
            "quotes": [
                "💬 'Speak when you’re calm, not when you’re angry.'"
            ]
        },
        "motivation": {
            "music": [
                "🔥 <a href='https://www.youtube.com/watch?v=4H_rzcarFnI' target='_blank'>Non Stop Motivation Song | Ft Toppo</a>"
            ],
            "articles": [
                "📘 <a href='https://www.linkedin.com/pulse/turning-pain-purpose-how-use-lifes-challenges-fuel-positive-rivers' target='_blank'>From Rage to Results: Channeling Anger</a>"
            ],
            "videos": [
                "🎥 <a href='https://www.youtube.com/watch?v=NDQ1Mi5I4rg' target='_blank'>The Mindset of Champions Motivational Clip</a>"
            ],
            "quotes": [
                "💬 'Use anger as fuel, not fire.'"
            ]
        },
        "distraction": {
            "music": [
                "🎶 <a href='https://www.youtube.com/watch?v=1nCqRmx3Dnw' target='_blank'>Comedy Parody Songs</a>"
            ],
            "articles": [
                "📰 <a href='https://www.goodnewsnetwork.org/' target='_blank'>Fun Facts You Didn’t Know</a>"
            ],
            "videos": [
                "🐱 <a href='https://www.youtube.com/watch?v=J---aiyznGQ' target='_blank'>Funny Cat Videos</a>",
                "😂 <a href='https://www.youtube.com/watch?v=AgQ8RV3zn2A' target='_blank'>Stand-up Comedy by Ravi Gupta</a>"
            ],
            "quotes": [
                "💬 'Laughter is the shock absorber that eases the blows of life.'"
            ]
        }
    },
    "neutral": {
        "comfort": {
            "music": [
                "🎵 <a href='https://www.youtube.com/watch?v=Qo6eO50oxHc' target='_blank'>Lo-fi Instrumental Playlist</a>"
            ],
            "articles": [
                "📄 <a href='https://www.helpguide.org/articles/stress/relaxation-techniques-for-stress-relief.htm' target='_blank'>5-Minute Mindfulness Read</a>"
            ],
            "videos": [
                "📺 <a href='https://www.youtube.com/watch?v=QZbuj3RJcjI' target='_blank'>Peaceful Scenery Video</a>"
            ],
            "quotes": [
                "💬 'Balance is not something you find, it's something you create.'"
            ]
        },
        "motivation": {
            "music": [
                "🎶 <a href='https://www.youtube.com/watch?v=OcmcptbsvzQ' target='_blank'>Inspirational Background Music</a>"
            ],
            "articles": [
                "📘 <a href='https://www.helpguide.org/articles/healthy-living/setting-goals.htm' target='_blank'>Goal-Setting for Growth</a>"
            ],
            "videos": [
                "🎥 <a href='https://www.youtube.com/watch?v=fUmiD9AtbWc' target='_blank'>'Your Time is Now' - Speech Clip</a>"
            ],
            "quotes": [
                "💬 'Believe in your infinite potential.'"
            ]
        },
        "distraction": {
            "music": [
                "🎧 <a href='https://www.youtube.com/watch?v=rtTI1rh9U5M' target='_blank'>Random Music Shuffle</a>"
            ],
            "articles": [
                "📚 <a href='https://en.wikipedia.org/wiki/Special:Random' target='_blank'>Random Wikipedia Article of the Day</a>"
            ],
            "videos": [
                "🎬 <a href='https://www.youtube.com/watch?v=QZbuj3RJcjI' target='_blank'>Unexpected Plot Twist Scenes</a>"
            ],
            "quotes": [
                "💬 'When in doubt, take a break and laugh.'"
            ]
        }
    }
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
        joy_score = next((i for emo, i in emotion_results if emo == "joy"), 0)
        #disgust_score=next((i for emo, i in emotion_results if emo == "joy"), 0)

        if sadness_score >= 7 and disappointment_score >= 5 and joy_score<=1 :
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
    #load_model()
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),debug=True)

