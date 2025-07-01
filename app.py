# app.py
import datetime
import io
import json
import logging
import pymongo
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from dotenv import load_dotenv
import os
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError
from gtts import gTTS
import base64
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from jsonschema import validate, ValidationError
from huggingface_hub import InferenceClient
from transformers import pipeline
import re
import bcrypt

# Configuration
load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = 'blaa'
CORS(app, resources={r"/*": {"origins": "*"}})

# Rate Limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri=os.getenv("REDIS_URI", "memory://"),
    default_limits=["200 per day", "50 per hour"],
    strategy="fixed-window",
)

# Database setup
client = MongoClient(
    os.getenv("MONGO_URI"),
    connectTimeoutMS=30000,
    socketTimeoutMS=None,
    serverSelectionTimeoutMS=5000,
)
db = client.mentorship_prod

# Database indexes
db.logs.create_index([("timestamp", ASCENDING)])
db.feedback.create_index([("rating", ASCENDING)])
db.users.create_index([("email", ASCENDING)], unique=True)
db.journal.create_index([("timestamp", ASCENDING)])

# AI Configuration
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_output_tokens": 2000,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=gemini_generation_config
)

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MentorAI")

# Validation schemas
CHAT_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string", "maxLength": 500},
    },
    "required": ["message"],
}

FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "rating": {"type": "integer", "minimum": 1, "maximum": 5},
        "user_input": {"type": "string"},
        "bot_response": {"type": "string"},
    },
    "required": ["rating", "user_input", "bot_response"],
}

# Define expected response schema for validation
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "Empathetic Acknowledgement": {"type": "string"},
        "Practical Suggestions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3
        },
        "Encouraging Closing Line": {"type": "string"},
        "CBT-style Reflection Tip": {"type": "string"},
    },
    "required": [
        "Empathetic Acknowledgement",
        "Practical Suggestions",
        "Encouraging Closing Line",
        "CBT-style Reflection Tip",
    ],
}


class MentorAI:
    def __init__(self):
        self.hf_client = InferenceClient(
            provider="hf-inference", api_key=os.getenv("HF_API_KEY")
        )
        self.speech_recognizer = pipeline(
            "automatic-speech-recognition", model="openai/whisper-small", framework="pt"
        )
        self.bot_name = "Ava"
        self.bot_intro = "Hi, I'm Ava, your AI mentor and listener. I'm here for you always—without judgment."

    def analyze_sentiment(self, text):
        try:
            results = self.hf_client.text_classification(
                text=text, model="SamLowe/roberta-base-go_emotions"
            )
            emotions = sorted(
                [{"label": e["label"], "score": float(e["score"])} for e in results],
                key=lambda x: x["score"],
                reverse=True,
            )[:2]
            emotion_mapping = {
                "admiration": "positive",
                "amusement": "positive",
                "anger": "anger",
                "annoyance": "frustration",
                "approval": "positive",
                "caring": "positive",
                "confusion": "confusion",
                "curiosity": "interest",
                "desire": "positive",
                "disappointment": "sadness",
                "disapproval": "negative",
                "disgust": "disgust",
                "embarrassment": "shame",
                "excitement": "excitement",
                "fear": "fear",
                "gratitude": "gratitude",
                "grief": "sadness",
                "joy": "joy",
                "love": "love",
                "nervousness": "anxiety",
                "neutral": "neutral",
                "optimism": "positive",
                "pride": "pride",
                "realization": "awareness",
                "relief": "relief",
                "remorse": "regret",
                "sadness": "sadness",
                "surprise": "surprise",
            }
            for e in emotions:
                e["label"] = emotion_mapping.get(e["label"].lower(), e["label"])
            return emotions
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return [{"label": "neutral", "score": 1.0}]

    def get_recent_history(self, limit=3):
        history = list(
            db.logs.find({"type": "text"}).sort("timestamp", -1).limit(limit)
        )
        conversation = ""
        for h in reversed(history):
            conversation += (
                f"User: {h['user_input']}\n{self.bot_name}: {h['response']}\n"
            )
        return conversation

    def format_response(self, response_dict):
        """Convert the structured response to a natural language format"""
        try:
            acknowledgement = response_dict["Empathetic Acknowledgement"]
            suggestions = response_dict["Practical Suggestions"]
            closing = response_dict["Encouraging Closing Line"]
            reflection = response_dict["CBT-style Reflection Tip"]

            suggestions_text = "\nHere are three suggestions that might help:"
            for i, suggestion in enumerate(suggestions, 1):
                suggestions_text += f"\n  {i}. {suggestion}"

            return f"{acknowledgement}\n{suggestions_text}\n{closing}\n\n{reflection}"
        except KeyError as e:
            logger.error(f"Response missing key: {str(e)}")
            # Fallback to simple concatenation
            return "\n".join(
                f"{k}: {v if not isinstance(v, list) else ', '.join(v)}"
                for k, v in response_dict.items()
            )

    def generate_response(self, user_input, emotions):
        try:
            history = self.get_recent_history()
            prompt = f"""
                {self.bot_name} - Mentorship AI
                Past Conversation:
                {history}

                User Emotions: {', '.join([e['label'] for e in emotions])}
                User Query: {user_input}

                Respond in the following strict JSON format only:
                {{
                    "Empathetic Acknowledgement": "Your empathetic acknowledgement here",
                    "Practical Suggestions": [
                        "Suggestion 1",
                        "Suggestion 2",
                        "Suggestion 3"
                    ],
                    "Encouraging Closing Line": "Your encouraging closing line",
                    "CBT-style Reflection Tip": "Your CBT reflection tip"
                }}

                Guidelines:
                1. Keep total response under 200 words
                2. Make Practical Suggestions exactly 3 items
                3. Be concise and emotionally intelligent
            """
            response = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
            
            # Clean and parse response
            text = response.text.strip()
            
            # Remove markdown code blocks if present
            if text.startswith("```json"):
                text = text[7:-3].strip()
            elif text.startswith("```"):
                text = text[3:-3].strip()
                
            # Parse JSON
            response_dict = json.loads(text)
            
            # Validate structure
            validate(instance=response_dict, schema=RESPONSE_SCHEMA)
            
            # Format to natural language
            formatted_response = self.format_response(response_dict)
            return formatted_response, response_dict
            
        except json.JSONDecodeError:
            logger.error("Failed to parse response as JSON")
            return "I'm having trouble formulating a response. Could you try rephrasing your question?", {}
        except ValidationError as e:
            logger.error(f"Response validation failed: {str(e)}")
            return "I'm working on improving my responses. Please try again.", {}
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return "I'm currently improving my responses. Please try rephrasing your question.", {}

    def process_audio(self, audio_bytes):
        try:
            audio_data = base64.b64decode(audio_bytes.split(",")[1])
            with io.BytesIO(audio_data) as audio_buffer:
                result = self.speech_recognizer(
                    audio_buffer.read(), return_timestamps=False
                )
                return result["text"].strip()
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            return ""


mentor_ai = MentorAI()


@app.route("/", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = db.auth.find_one({"username": username})
        if user and bcrypt.checkpw(password.encode("utf-8"), user["password"]):
            session["user_id"] = str(user["_id"])
            return redirect(url_for("dashboard"))
        else:
            return render_template("index.html", error="Invalid username or password.")

    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        if not username or not email or not password:
            return render_template("signup.html", error="Username, email, and password are required.")

        if db.auth.find_one({"username": username}):
            return render_template("signup.html", error="Username already registered.")
        if db.auth.find_one({"email": email}):
            return render_template("signup.html", error="Email already registered.")

        hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

        result = db.auth.insert_one({
            "username": username,
            "email": email,
            "password": hashed_pw
        })

        session["user_id"] = str(result.inserted_id)
        return redirect(url_for("dashboard"))

    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))


@app.route("/dashboard")
@limiter.limit("10/minute")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")


@app.route("/chat", methods=["POST"])
@limiter.limit("15/minute")
def handle_chat():
    try:
        data = request.get_json()
        validate(instance=data, schema=CHAT_SCHEMA)
        userid = session.get("user_id")
        if not userid:
            return jsonify({"error": "Unauthorized"}), 401

        message = data["message"].strip()[:500]
        if re.search(r"(ignore|forget|pretend|as an ai)", message, re.IGNORECASE):
            return jsonify({"error": "Unsafe input detected"}), 400

        emotions = mentor_ai.analyze_sentiment(message)
        response_text, structured_response = mentor_ai.generate_response(message, emotions)

        # Daily mood reminder
        today = datetime.datetime.now(datetime.timezone.utc).date()
        user_logs = db.logs.find_one({"userid": userid})
        already_chatted_today = False

        if user_logs:
            for log in user_logs.get("messages", []):
                log_time = log["timestamp"]
                if log_time.date() == today:
                    already_chatted_today = True
                    break
        
        if not already_chatted_today:
            response_text = (
                "Hey! Just checking in—how are you feeling today? Remember, sharing a little goes a long way. 😊\n\n"
                + response_text
            )

        # Journal entry detection
        if "write this down" in message.lower():
            db.journal.insert_one(
                {
                    "entry": response_text,
                    "user_input": message,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc),
                }
            )
            response_text += "\n\n📝 I've saved this in your CBT journal."

        # Generate speech
        tts = gTTS(response_text, lang="en")
        audio_io = io.BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        audio_base64 = base64.b64encode(audio_io.read()).decode("utf-8")

        message_doc = {
            "user_input": message,
            "response": response_text,
            "emotions": [e["label"] for e in emotions],
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "type": "text",
        }

        db.logs.update_one(
            {"userid": userid},
            {"$push": {"messages": message_doc}},
            upsert=True
        )

        return jsonify(
            {
                "structured_response": structured_response,
                "audio_response": audio_base64,
                "sentiment": [e["label"] for e in emotions],
            }
        )

    except ValidationError:
        return jsonify({"error": "Invalid input format"}), 400
    except PyMongoError as e:
        logger.critical(f"Database error: {str(e)}")
        return jsonify({"error": "System maintenance in progress"}), 503
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({"error": "Processing failed"}), 500


@app.route("/journal", methods=["GET"])
def get_journal():
    try:
        entries = list(db.journal.find({}, {"_id": 0}).sort("timestamp", -1).limit(50))
        for e in entries:
            e["timestamp"] = e["timestamp"].isoformat()
        return jsonify(entries)
    except Exception as e:
        logger.error(f"Journal fetch error: {str(e)}")
        return jsonify({"error": "Failed to load journal"}), 500


@app.route("/analytics")
def get_analytics():
    try:
        engagement = list(
            db.logs.aggregate(
                [
                    {
                        "$match": {
                            "timestamp": {
                                "$gte": datetime.datetime.now(datetime.timezone.utc)
                                - datetime.timedelta(days=30)
                            }
                        }
                    },
                    {
                        "$group": {
                            "_id": {
                                "$dateToString": {
                                    "format": "%Y-%m-%d",
                                    "date": "$timestamp",
                                }
                            },
                            "count": {"$sum": 1},
                        }
                    },
                    {"$sort": {"_id": 1}},
                ]
            )
        )

        sentiment = list(
            db.logs.aggregate(
                [
                    {"$unwind": "$emotions"},
                    {"$group": {"_id": "$emotions", "count": {"$sum": 1}}},
                ]
            )
        )

        effectiveness = list(
            db.feedback.aggregate(
                [
                    {
                        "$group": {
                            "_id": None,
                            "averageRating": {"$avg": "$rating"},
                            "totalSessions": {"$sum": 1},
                        }
                    }
                ]
            )
        )

        return jsonify(
            {
                "engagement": engagement,
                "sentiment_distribution": sentiment,
                "effectiveness": effectiveness[0] if effectiveness else {},
            }
        )
    except PyMongoError as e:
        logger.error(f"Analytics error: {str(e)}")
        return jsonify({"error": "Failed to generate analytics"}), 500


@app.route("/history")
@limiter.limit("20/minute")
def get_history():
    try:
        history = list(
            db.logs.find(
                {"type": "text"},
                {
                    "_id": 0,
                    "user_input": 1,
                    "response": 1,
                    "timestamp": 1,
                    "emotions": 1,
                },
            )
            .sort("timestamp", -1)
            .limit(50)
        )
        for entry in history:
            entry["timestamp"] = entry["timestamp"].isoformat()
        return jsonify(history)
    except PyMongoError as e:
        logger.error(f"History error: {str(e)}")
        return jsonify({"error": "Failed to retrieve history"}), 500


@app.route("/feedback", methods=["POST"])
@limiter.limit("10/minute")
def handle_feedback():
    try:
        data = request.get_json()
        validate(instance=data, schema=FEEDBACK_SCHEMA)

        feedback_data = {
            "rating": data["rating"],
            "user_input": data.get("user_input", "")[:200],
            "bot_response": data.get("bot_response", "")[:500],
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
        }

        db.feedback.update_one(
            {
                "user_input": feedback_data["user_input"],
                "bot_response": feedback_data["bot_response"],
            },
            {"$set": feedback_data},
            upsert=True,
        )
        return jsonify({"status": "success"})

    except ValidationError as e:
        logger.error(f"Feedback validation failed: {str(e)}")
        return jsonify({"error": "Invalid feedback format"}), 400
    except PyMongoError as e:
        logger.error(f"Feedback error: {str(e)}")
        return jsonify({"error": "Failed to store feedback"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)