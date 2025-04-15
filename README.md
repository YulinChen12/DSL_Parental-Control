# DSL_Parental-Control
# 🛡️ DSL Parental Control Chat App

This is a local web application that helps parents define and enforce moderation policies for AI chats with their children. It includes moderation logic such as rewriting, blocking, or custom responses to unsafe prompts, and provides tools for parents to monitor and guide AI interaction.

---

### 1. Prerequisites

Before running the app, make sure you have the following:

- ✅ Python 3.8 or newer installed  
- ✅ [Ollama](https://ollama.com) installed and running locally  
- ✅ The `llama3.2` model downloaded via Ollama  
- ✅ Flask installed

To install Flask (if you haven't already):

```bash
pip install flask
```

---

### 2. Install and Run LLM (llama3.2)

#### Step 1: Install Ollama

Download and install Ollama from:  
👉 https://ollama.com/download

Ollama is available for macOS, Windows, and Linux.

#### Step 2: Pull the llama3.2 model

```bash
ollama pull llama3.2
```

> This will download the model to your local machine.

#### Step 3: Run the llama3.2 model

```bash
ollama run llama3.2
```

Leave this terminal running — it will keep the model alive for your app to use.

---

### 3. Running the App

Once Flask is installed and `llama3.2` is running via Ollama:

```bash
python app.py
```

Then open your browser and go to:

```
http://localhost:5000
```

---

### 4. Project Structure

```
DSL_Parental-Control/
├── app.py                  # Main Flask app
├── policy.json             # Stores parent policies
├── history.txt             # Logs prompts and rewrites for the kids
├── templates/              # HTML pages for kid/parent views
│   ├── chat.html
│   ├── policy.html
│   └── parent.html
    └── login.html
└── README.md
```

---

### 5. How It Works

#### For Parents

- Visit `/policy` to define moderation rules:
  - `rewrite`: the model rewrites the prompt to be safer
  - `respond`: the model sends a preset response
  - `block`: the prompt is blocked
- View reports and prompt categories in `/dashboard`

#### For Kids

- Kids type messages in the chat
- Each prompt is moderated according to parent policy
- The model only sees the rewritten (safe) prompt
- Responses are displayed to the child

---

### 6. Example policy.json Format

```json
{
  "123": [
    {
      "type": "rewrite",
      "text": "If the child asks about staying up late, rephrase into a general question about sleep health."
    },
    {
      "type": "respond",
      "text": "If the child says 'lol', respond with 'haha!'"
    }
  ]
}
```

---

