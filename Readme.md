# MentorChatbot 🎓

**A personal AI-powered mentorship chatbot (Ava) with CBT style guidance, journaling, and analytics.**

---

## 🚀 Quick Overview

MentorChatbot ("Ava") is an emotion aware, cognitive behavioral therapy (CBT) inspired chatbot built with:

* **Google Gemini** for structured empathetic responses
* **Hugging Face** for sentiment detection (*GoEmotions*) & speech-to-text (*Whisper*)
* **MongoDB** for journaling, logs, feedback & analytics
* **Flask** backend, Jinja2 templates, and a React/Vue (or vanilla JS) frontend

It helps users share how they’re feeling, receive three tailored suggestions, and save journal entries—all with admin dashboards for tracking user engagement and wellbeing trends.

---

## 🌟 Key Features

* **💬 AI Chat Mentor (Ava)**
  Responds empathetically via structured JSON (acknowledgement, 3 suggestions, encouraging close, CBT tip)

* **😊 Emotion Analysis**
  Detects top 2 emotional tones (e.g. sadness, anxiety) for tailored support

* **📝 Smart Journaling**
  Auto-saves entries when user types cues like “write this down”

* **🎧 Speech Support** *(in-progress)*
  Whisper ASR transforms voice to text

* **📊 Analytics Dashboard**
  For users: daily sessions, emotion trends
  For admins: per-user logs, feedback scores, usage stats

* **🔒 Authentication & Security**
  Passwords hashed with bcrypt and sessions managed securely

* **⚙️ Rate Limit + Schema Validation**
  Safeguards APIs & ensures consistent interactions

---

## 🛠 Tech Stack

| Layer           | Tools & Libraries                                        |
| --------------- | -------------------------------------------------------- |
| Backend         | Python, Flask, Flask-Limiter, Flask-CORS                 |
| Frontend        | HTML + Jinja2 Templates, React/Vue or Vanilla JS         |
| Database        | MongoDB via PyMongo (with indices & aggregations)        |
| AI & NLP        | Google Gemini, Transformers, HuggingFace Hub             |
| Speech          | Whisper (Audio-to-text), gTTS (Text-to-speech, optional) |
| Auth & Security | bcrypt, JSONSchema validation                            |
| Deployment      | `.env` env vars, threaded Flask server                   |

---

## 🧩 Installation & Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/krthik777/MentorChatbot.git
   cd MentorChatbot
   ```

2. **Create & activate virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env`:

   ```
   GEMINI_API_KEY=...
   HF_API_KEY=...
   MONGO_URI=...
   ```

5. **Populate MongoDB**
   Your DB will automatically create collections and indexes on startup.

6. **Run the app**

   ```bash
   python app.py
   ```

   Access the app at `http://localhost:5000`

---

## 📡 API Endpoints

| Verb | Endpoint                          | Description                           |
| ---- | --------------------------------- | ------------------------------------- |
| GET  | `/`                               | Login page                            |
| POST | `/signup`                         | Create user account                   |
| POST | `/chat`                           | Chat with Ava (text)                  |
| GET  | `/dashboard`                      | User or admin dashboard               |
| GET  | `/analytics`                      | User-specific analytics               |
| GET  | `/admin/getAnalytics`             | Global analytics (admin only)         |
| GET  | `/admin/user/<user_id>/analytics` | User-specific logs & feedback (admin) |
| GET  | `/journal`                        | Retrieve journal entries              |
| GET  | `/history`                        | Retrieve last 50 chat messages        |
| POST | `/feedback`                       | Leave a rating and feedback           |
| GET  | `/logout`                         | Log out user                          |

---

## 🧠 Ava's Response Format

For every `/chat` message, Ava returns a structured JSON:

```json
{
  "Empathetic Acknowledgement": "I hear how overwhelming that sounds right now.",
  "Practical Suggestions": [
    "Take three deep breaths and pause.",
    "Write down what's on your mind before sleeping.",
    "Talk with a trusted friend or relative."
  ],
  "Encouraging Closing Line": "You're not alone—and you're stronger than you think.",
  "CBT-style Reflection Tip": "Try spotting a thought that caused how you feel."
}
```

---

## 👩‍💼 UI & Frontend

The UI includes:

* **Login / Signup** forms (HTML)
* **Chat window**: text input, buttons for audio input, chat bubbles
* **Dashboard**: shows emotion trend graphs, session counts, journal entries
* **Admin panel**: user list and advanced analytics charts

*(Scripts and styles may vary depending on your frontend tech.)*

---

## 🛡 Security & Safety

* **Rate Limiting**: Prevents usage abuse (e.g., `/chat` capped at 15 requests/min)
* **Input Validation**: JSONSchema for chat and feedback payloads
* **Prompt Hardening**: Blocks attempts to manipulate chatbot behavior via forbidden keywords
* **Password Security**: bcrypt for hashed + salted storage
* **AI Safety Settings**: Gemini configured to allow sensitive conversation patterns responsibly

---

## 📊 Analytics

* **User View**: tracks chat count, sentiment breakdown, average ratings
* **Admin View**: sees total users, message count, feedback rate, average score, individual user breakdowns

---

## 📂 Project Structure

```
MentorChatbot/
│
├── app.py                # Core Flask server with all route logic
├── templates/            # HTML templates for UI
├── requirements.txt
├── .env.example
├── LICENSE
└── README.md             # ← You are here!
```

---

## 📄 License

Licensed under the MIT License. See the `LICENSE` file.

---

## 🎯 Contact

Built by `@krthik777 & @aarontoms`. For feedback, bugs, or questions, feel free to open an issue or reach out to us via GitHub.

---

Thank you for building and using MentorChatbot ✨
