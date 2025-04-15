import os, json, uuid, re, requests, io, base64
from datetime import datetime, timezone, timedelta
from functools import wraps
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, session, send_file
)
import string
from collections import defaultdict

STOPWORDS = set("""
    i me my you your he she it they we us them is are was were do does did doing can could should would will 
    to from at on in for with of by be have has had am the a an how what when where why which who whose this that 
    and or but if not so then also just only all any each every more most many some such no nor too very
    said tell told says use using used get got like want need think see make go come ask said say can't don’t won't 
    couldn’t shouldn’t wasn’t isn’t wasn’t didn’t
""".split())

def clean_and_filter_keywords(text):
    # Remove punctuation and lowercase
    tokens = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    keywords = [
        word for word in tokens
        if word not in STOPWORDS
        and len(word) > 2
        and not word.isnumeric()
    ]
    return keywords



# ── Basic Config ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "replace‑with‑something‑random"

OLLAMA_URL   = "http://localhost:11434/api/generate"   # using /api/generate for kid mode
MODEL_NAME   = "llama3.2"
HISTORY_FILE = "history.txt"
# ─────────────────────────────────────────────────────────────────────────

# ── Helpers for history file ─────────────────────────────────────────────


def utc_iso():
    return datetime.now(timezone.utc).isoformat()

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return {}
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_history(data):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def start_new_thread(kid_id):
    data = load_history()
    tid  = str(uuid.uuid4())
    data.setdefault(kid_id, []).insert(0, {
        "id": tid,
        "started": utc_iso(),
        "turns": []
    })
    save_history(data)
    return tid

def get_threads(kid_id):
    return load_history().get(kid_id, [])

def get_turns(kid_id, tid):
    for t in get_threads(kid_id):
        if t["id"] == tid:
            return t["turns"]
    return []

def append_turn(kid_id, tid, turn):
    data = load_history()
    for t in data.setdefault(kid_id, []):
        if t["id"] == tid:
            t["turns"].append(turn)
            break
    save_history(data)
    
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
# ─────────────────────────────────────────────────────────────────────────

# ── Login Wrapper ─────────────────────────────────────────────────────────
def login_required(role):
    def deco(fn):
        @wraps(fn)
        def wrap(*a, **kw):
            if session.get("role") != role:
                # redirect to login if session role doesn't match
                return redirect(url_for("login"))
            return fn(*a, **kw)
        return wrap
    return deco

# ==========================  ROUTES  ======================================

@app.route("/", methods=["GET", "POST"])
def login():
    # Step 1: Choose role.
    if request.method == "POST" and "role" in request.form:
        r = request.form["role"]
        if r in ("kid", "parent"):
            session.clear()
            session["role"] = r
            return render_template("login.html", need_id=True, role=r)

    # Step 2: Credentials.
    # For Kid: require only user_id (KidsID).
    # For Parent: require both user_id (ParentID) and kid_id.
    if request.method == "POST" and "user_id" in request.form:
        role = session.get("role")
        if role == "kid":
            uid = request.form["user_id"].strip()
            if uid:
                session["user_id"] = uid
                threads = get_threads(uid)
                session["thread_id"] = start_new_thread(uid) if not threads else threads[0]["id"]
                return redirect(url_for("kid_chat"))
        elif role == "parent":
            parent_id = request.form["user_id"].strip()
            kid_id = request.form.get("kid_id", "").strip()
            if parent_id and kid_id:
                session["user_id"] = parent_id
                session["kid_id"] = kid_id
                return redirect(url_for("parent_dash"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# Kid View (unchanged)
@app.route("/kid")
@login_required("kid")
def kid_chat():
    return render_template("chat.html", role="Kid", user_id=session["user_id"])

# Parent Dashboard: Shows ParentID and monitored KidID, with report button.
@app.route("/parent")
@login_required("parent")
def parent_dash():
    return render_template("parent.html", 
                           parent_id=session["user_id"],
                           kid_id=session["kid_id"])
@app.route("/policy_page")
@login_required("parent")
def policy_page():
    kid_id = session["kid_id"]
    policies = load_policies().get(kid_id, [])
    return render_template("policy.html", kid_id=kid_id, policies=policies)

# ── APIs for Kid Mode ─────────────────────────────────────────────────────
@app.route("/history")
def history_list():
    if session.get("role") != "kid":
        return jsonify([])
    return jsonify(get_threads(session["user_id"]))

@app.route("/history/<tid>")
def history_thread(tid):
    if session.get("role") != "kid":
        return jsonify([])
    return jsonify(get_turns(session["user_id"], tid))

@app.route("/new_thread")
def new_thread():
    if session.get("role") != "kid":
        return jsonify({"error": "forbidden"}), 404
    tid = start_new_thread(session["user_id"])
    session["thread_id"] = tid
    return jsonify({"thread_id": tid})

# ── Policy ─────────────────────────────────────────────────────

@app.route("/dashboard")
def dashboard():
    return render_template("parent_dashboard.html", parent_id=session["user_id"], kid_id=session.get("kid_id"))

@app.route("/policy", methods=["GET", "POST"])
def policy():

    parent_id = session.get("user_id")
    kid_id = session.get("kid_id")

    # Load policy file
    try:
        with open("policy.json", "r") as f:
            policy_data = json.load(f)
    except:
        policy_data = {}

    # Flatten all policies across kids to pass into template
    policies = []
    for k_id, plist in policy_data.items():
        for p in plist:
            p["kid_id"] = k_id
            policies.append(p)

    # Handle policy creation
    if request.method == "POST":
        new_policy = {
            "type": request.form["category"],
            "text": request.form["policy"]
        }

        # Insert into correct kid's policy list
        if kid_id not in policy_data:
            policy_data[kid_id] = []
        policy_data[kid_id].append(new_policy)

        # Save back
        with open("policy.json", "w") as f:
            json.dump(policy_data, f, indent=2)

        return redirect("/policy")

    return render_template("policy.html", policies=policies, parent_id=parent_id, kid_id=kid_id)


@app.route("/delete_policy", methods=["POST"])
def delete_policy():
    index = int(request.form["index"])
    try:
        with open("policy.json", "r") as f:
            policy_data = json.load(f)

        # Flatten into list and locate which kid the policy belongs to
        all_policies = []
        for k_id, plist in policy_data.items():
            for p in plist:
                all_policies.append((k_id, p))

        target_kid_id, _ = all_policies[index]
        policy_data[target_kid_id].pop(
            next(i for i, p in enumerate(policy_data[target_kid_id]) if p == all_policies[index][1])
        )

        # Save updated file
        with open("policy.json", "w") as f:
            json.dump(policy_data, f, indent=2)

    except Exception as e:
        print("Error deleting policy:", e)

    return redirect("/policy")


# POLICY_FILE = "policy.json"

# def load_policies():
#     if not os.path.exists(POLICY_FILE):
#         return {}
#     with open(POLICY_FILE, "r", encoding="utf-8") as f:
#         try:
#             return json.load(f)
#         except json.JSONDecodeError:
#             return {}

# def save_policy(kid_id, policy_type, policy_text):
#     data = load_policies()
#     data.setdefault(kid_id, [])

#     new_policy = {"type": policy_type, "text": policy_text.strip()}
#     if new_policy not in data[kid_id]:
#         data[kid_id].append(new_policy)

#     with open(POLICY_FILE, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)
# @app.route("/policy", methods=["POST"])
# @login_required("parent")
# def update_policy():
#     kid_id = session["kid_id"]
#     policy_type = request.form.get("policy_type", "").strip().lower()
#     policy_text = request.form.get("policy_text", "").strip()

#     if policy_type not in {"rewrite", "respond", "block"} or not policy_text:
#         return "Invalid input", 400

#     save_policy(kid_id, policy_type, policy_text)
#     return redirect(url_for("parent_dash"))
# def format_policies_for_prompt(kid_id):
#     policies = load_policies().get(kid_id, [])
#     grouped = {"rewrite": [], "respond": [], "block": []}
#     for p in policies:
#         grouped[p["type"]].append(p["text"])

#     return f"""
# REWRITE Policies:
# {chr(10).join(f"- {p}" for p in grouped["rewrite"])}

# RESPOND Policies:
# {chr(10).join(f"- {p}" for p in grouped["respond"])}

# BLOCK Policies:
# {chr(10).join(f"- {p}" for p in grouped["block"])}
# """.strip()


# ── /ask route for Parent mode ─────────────────────────────────────────────

# @app.route("/ask", methods=["POST"])
# def ask():
#     req = request.json
#     user_input = req["prompt"]
#     tid = req.get("thread_id")
#     role = session.get("role")
#     uid = session.get("user_id", "anon")

#     if not tid or not get_turns(uid, tid):
#         tid = start_new_thread(uid)

#     kid_id = uid  # In kid mode, uid is the kid ID
#     policies_text = format_policies_for_prompt(kid_id)

#        screen_prompt = f"""
# You are a child safety assistant. Your ONLY job is to enforce parent-defined moderation policies.

# DO NOT use your own judgment.
# DO NOT generate friendly responses or explanations unless a parent explicitly told you to.
# DO NOT classify or block anything unless a parent policy matches.

# ---

# There are three allowed actions:

# 1. "respond": Return a specific message. Do not generate your own response.
# 2. "rewrite": Rephrase the prompt to make it safer without changing the intent.
# 3. "block": Deny the prompt completely. Do not explain, rephrase, or help.

# If NO policy matches the prompt:
# → Set "action" to "none" and pass the prompt through unchanged.

# ---

# Respond with this exact JSON format:
# {{
#   "action": "rewrite" | "respond" | "block" | "none",
#   "rephrased_prompt": "...",     # required only if action is rewrite
#   "response_text": "...",        # required only if action is respond
#   "block_reason": "...",         # required only if action is block
#   "semantic_category": "Homework" | "Social" | "Games" | "Other"
# }}

# ---

# Parent-defined policies:
# \"\"\"
# {format_policies_for_prompt(kid_id)}
# \"\"\"

# Child's input:
# \"\"\"
# {user_input}
# \"\"\"
# """


#     try:
#         screen_res = requests.post(OLLAMA_URL, json={
#             "model": MODEL_NAME,
#             "prompt": screen_prompt,
#             "stream": False
#         }, timeout=60)
#         raw = screen_res.json().get("response", "")
#         match = re.search(r"\{.*\}", raw, re.DOTALL)
#         screen_data = json.loads(match.group()) if match else {}
#     except Exception:
#         screen_data = {
#             "action": "none",
#             "semantic_category": "Uncategorized"
#         }

#     action = screen_data.get("action", "none")
#     semantic_category = screen_data.get("semantic_category", "Uncategorized")
#     rewritten_prompt = screen_data.get("rephrased_prompt", user_input)

#     # Step 3: Log user prompt
#     append_turn(uid, tid, {
#         "role": "user",
#         "text": user_input,
#         "original_prompt": user_input,
#         "action": action,
#         "semantic_category": semantic_category,
#         "time": utc_iso()
#     })

#     # Step 4: Handle each moderation action
#     if action == "respond":
#         return jsonify({
#             "response": screen_data.get("response_text", "[Policy Response]"),
#             "thread_id": tid,
#             "was_rewritten": False,
#             "rewrite_display": "[Policy: Direct Response]"
#         })

#     elif action == "block":
#         return jsonify({
#             "response": f"[BLOCKED] {screen_data.get('block_reason', 'Blocked by parent policy.')}",
#             "thread_id": tid,
#             "was_rewritten": False,
#             "rewrite_display": "[Policy: Blocked]"
#         })

#     elif action == "rewrite":
#         append_turn(uid, tid, {
#             "role": "user",
#             "text": f"REWRITTEN: {rewritten_prompt}",
#             "original_prompt": user_input,
#             "rewrited_prompt": rewritten_prompt,
#             "action": "rewrite",
#             "semantic_category": semantic_category,
#             "time": utc_iso()
#         })

#     # Step 5: Ask the model using the moderated or original prompt
#     turns = get_turns(uid, tid)[-20:]
#     context = "".join(f"{t['role']}: {t['text']}\n" for t in turns)
#     context += f"user: {rewritten_prompt}"

#     try:
#         res = requests.post(OLLAMA_URL, json={
#             "model": MODEL_NAME,
#             "prompt": context,
#             "stream": False
#         }, timeout=120)
#         response = res.json().get("response", "")
#     except Exception as e:
#         response = f"Error contacting Ollama: {e}"

#     append_turn(uid, tid, {
#         "role": "model",
#         "text": response,
#         "time": utc_iso()
#     })

#     return jsonify({
#         "response": response,
#         "thread_id": tid,
#         "was_rewritten": action == "rewrite",
#         "rewrite_display": f"REWRITTEN: {rewritten_prompt}" if action == "rewrite" else ""
#     })

@app.route("/ask", methods=["POST"])
def ask():
    req = request.json
    user_input = req["prompt"]
    tid = req.get("thread_id")
    role = session.get("role")
    uid = session.get("user_id", "anon")

    if not tid or not get_turns(uid, tid):
        tid = start_new_thread(uid)

    # Step 1: Moderate Prompt
    screen_prompt = f"""You are a safety assistant reviewing messages from children.

        Your job is to:
            1. Detect only clearly inappropriate or unsafe prompts (e.g. violence, deception, privacy violations) Especially when you see the user mention that their parent.
            2. If the prompt is inappropriate, rewrite it in a safer way **without changing the user's intent**.REMEMBER! Dont change the central idea of what the user is asking. Just change a way to ask it. Keep in mind you are a children assistant,the most important goal is to let the children explore but you have to guide them!
            3. If the prompt is appropriate or neutral, do NOT rewrite it.
            4. Classify the nature of the prompt into one high-level category, like:
   "Homework", "Social", "Family", "Entertainment", "Games", "Emotional", "Curiosity", "Math", "Science", or "Other".

        Respond ONLY with valid JSON in this format:
        {{
          "rewrite": "yes" or "no",
          "rephrased_prompt": "...",
          "category": "Deceptive Behavior", "Safety Issue",or "None"
          "semantic_category": "Homework", "Social", "Games", etc.
        }}

        Input: {user_input}
        """

    try:
        screen_res = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": screen_prompt,
            "stream": False
        }, timeout=60)
        raw = screen_res.json().get("response", "")
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        screen_data = json.loads(match.group()) if match else {}
        semantic_category = screen_data.get("semantic_category", "Uncategorized")
    except Exception:
        screen_data = {
            "rewrite": "no",
            "rephrased_prompt": user_input,
            "category": "None"
        }
        semantic_category = "Uncategorized"

    is_rewrite       = screen_data.get("rewrite", "no").lower() == "yes"
    rewritten_prompt = screen_data.get("rephrased_prompt", user_input)
    category         = screen_data.get("category", "None")

    # Only record history if it's a kid
    if role == "kid":
        append_turn(uid, tid, {
            "role": "user",
            "text": user_input,
            "original_prompt": user_input,
            "rewrite": "no",
            "semantic_category": semantic_category,
            "time": utc_iso()
        })

        if is_rewrite:
            append_turn(uid, tid, {
                "role": "user",
                "text": f"REWRITTEN ({category}): {rewritten_prompt}",
                "original_prompt": user_input,
                "rewrited_prompt": rewritten_prompt,
                "rewrite_category": category,
                "rewrite": "yes",
                "semantic_category": semantic_category,
                "time": utc_iso()
            })

    # Step 2: Generate model response using rewritten or original prompt
    turns = get_turns(uid, tid)[-20:]
    context = "".join(f"{t['role']}: {t['text']}\n" for t in turns)
    context += f"user: {rewritten_prompt}"

    try:
        res = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": context,
            "stream": False
        }, timeout=120)
        response = res.json().get("response", "")
    except Exception as e:
        response = f"Error contacting Ollama: {e}"

    if role == "kid":
        append_turn(uid, tid, {
            "role": "model",
            "text": response,
            "time": utc_iso()
        })

    display_prompt = f"REWRITTEN ({category}): {rewritten_prompt}" if is_rewrite else user_input
    return jsonify({
        "response": response,
        "thread_id": tid,
        "was_rewritten": is_rewrite,
        "display_prompt": display_prompt,
        "original_prompt": user_input if is_rewrite else None,
        "rewritten_prompt": rewritten_prompt if is_rewrite else None,
        "rewrite_category": category if is_rewrite else None
    })

# ── /ask route for kids mode ─────────────────────────────────────────────

    # Step 1: Moderate Prompt
    screen_prompt = f"""You are a safety assistant reviewing messages from children.

        Your job is to:
            1. Detect only clearly inappropriate or unsafe prompts (e.g. violence, deception, privacy violations) Especially when you see the user mention that their parent.
            2. If the prompt is inappropriate, rewrite it in a safer way **without changing the user's intent**.REMEMBER! Dont change the central idea of what the user is asking. Just change a way to ask it. Keep in mind you are a children assistant,the most important goal is to let the children explore but you have to guide them!
            3. If the prompt is appropriate or neutral, do NOT rewrite it.
            4. lassify the nature of the prompt into one high-level category, like:
   "Homework", "Social", "Family", "Entertainment", "Games", "Emotional", "Curiosity", "Math", "Science", or "Other".

        Respond ONLY with valid JSON in this format:
        {{
          "rewrite": "yes" or "no",
          "rephrased_prompt": "...",
          "category": "Deceptive Behavior", "Safety Issue",or "None"
          "semantic_category": "Homework", "Social", "Games", etc.
        }}

        Input: {user_input}
        """

    try:
        screen_res = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": screen_prompt,
            "stream": False
        }, timeout=60)
        raw = screen_res.json().get("response", "")
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        screen_data = json.loads(match.group()) if match else {}
        semantic_category = screen_data.get("semantic_category", "Uncategorized")
    except Exception:
        screen_data = {
            "rewrite": "no",
            "rephrased_prompt": user_input,
            "category": "None"
        }
        semantic_category = "Uncategorized"

    is_rewrite       = screen_data.get("rewrite", "no").lower() == "yes"
    rewritten_prompt = screen_data.get("rephrased_prompt", user_input)
    category         = screen_data.get("category", "None")

    # Step 2: Append the original prompt to history
    append_turn(uid, tid, {
        "role": "user",
        "text": user_input,
        "original_prompt": user_input,
        "rewrite": "no",
        "semantic_category": semantic_category,
        "time": utc_iso()
    })

    # Step 3: If rewritten, append the rewritten version as a separate turn
    if is_rewrite:
        append_turn(uid, tid, {
            "role": "user",
            "text": f"REWRITTEN ({category}): {rewritten_prompt}",
            "original_prompt": user_input,
            "rewrited_prompt": rewritten_prompt,
            "rewrite_category": category,
            "rewrite": "yes",
            "semantic_category": semantic_category,
            "time": utc_iso()
        })

    # Step 4: Generate model response using the moderated prompt (rewritten if available)
    turns = get_turns(uid, tid)[-20:]
    context = "".join(f"{t['role']}: {t['text']}\n" for t in turns)
    context += f"user: {rewritten_prompt}"
    try:
        res = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": context,
            "stream": False
        }, timeout=120)
        response = res.json().get("response", "")
    except Exception as e:
        response = f"Error contacting Ollama: {e}"

    append_turn(uid, tid, {
        "role": "model",
        "text": response,
        "time": utc_iso()
    })

    display_prompt = f"REWRITTEN ({category}): {rewritten_prompt}" if is_rewrite else user_input
    return jsonify({
    "response": response,
    "thread_id": tid,
    "was_rewritten": is_rewrite,
    "rewrite_display": f"REWRITTEN ({category}): {rewritten_prompt}" if is_rewrite else user_input
})

# ── Parent Report endpoints ─────────────────────────────────────────────
from flask import request
from datetime import timedelta

...
@app.route("/report")
@login_required("parent")
def report():
    kid_id = session["kid_id"]
    days = int(request.args.get("days", 7))
    one_week_ago = datetime.now(timezone.utc) - timedelta(days=days)

    data = load_history().get(kid_id, [])
    total_prompt_count = 0
    rewrite_count = 0
    cat_counter = Counter()
    rewrite_breakdown = Counter()
    keyword_counter = Counter()

    for thread in data:
        turns = thread.get("turns", [])
        i = 0
        while i < len(turns):
            turn = turns[i]
            if turn["role"] != "user":
                i += 1
                continue

            t = datetime.fromisoformat(turn["time"])
            if t < one_week_ago:
                i += 1
                continue

            # Count prompt (original + rewrite) as one unit
            total_prompt_count += 1

            if i + 1 < len(turns) and turns[i + 1].get("rewrite", "no") == "yes":
                rewrite_count += 1
                rewrite_turn = turns[i + 1]
                rewrite_breakdown[rewrite_turn.get("rewrite_category", "Unknown")] += 1
                cat_counter[rewrite_turn.get("semantic_category", "Other")] += 1

                # keyword extraction from original prompt
                prompt_text = turn.get("original_prompt") or turn.get("text")
                for word in clean_and_filter_keywords(prompt_text):
                    keyword_counter[word] += 1

                i += 2
                continue

            # No rewrite
            cat_counter[turn.get("semantic_category", "Other")] += 1
            prompt_text = turn.get("original_prompt") or turn.get("text")
            for word in clean_and_filter_keywords(prompt_text):
                keyword_counter[word] += 1

            i += 1

    # Chart: Prompt categories
    fig1, ax1 = plt.subplots()
    ax1.bar(cat_counter.keys(), cat_counter.values(), color="#42a5f5")
    ax1.set_title(f"Prompt Categories (Last {days} Days)")
    ax1.set_ylabel("Count")
    plt.xticks(rotation=45)
    fig1.tight_layout()
    chart1_b64 = fig_to_base64(fig1)

    # Chart: Rewrite categories (pie chart only if data)
    chart2_b64 = ""
    if rewrite_breakdown:
        fig2, ax2 = plt.subplots()
        ax2.pie(rewrite_breakdown.values(), labels=rewrite_breakdown.keys(), autopct='%1.1f%%', colors=plt.cm.Pastel1.colors)
        ax2.set_title(f"Rewrite Categories (Last {days} Days)")
        fig2.tight_layout()
        chart2_b64 = fig_to_base64(fig2)

    top_keywords = [kw for kw, _ in keyword_counter.most_common(10)]

    return jsonify({
        "total": total_prompt_count,
        "rewrite_total": rewrite_count,
        "categories": cat_counter,
        "rewrite_breakdown": [{"cat": k, "cnt": v} for k, v in rewrite_breakdown.items()],
        "chart_semantic": chart1_b64,
        "chart_rewrite": chart2_b64,
        "keywords": top_keywords
    })
...


@app.route("/report/pdf")
@login_required("parent")
def report_pdf():
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    kid_id = session["kid_id"]
    days = 7
    one_week_ago = datetime.now(timezone.utc) - timedelta(days=days)
    data = load_history().get(kid_id, [])
    total_prompt_count = 0
    rewrite_count = 0
    cat_counter = Counter()
    rewrite_breakdown = Counter()
    keyword_counter = Counter()

    for thread in data:
        turns = thread.get("turns", [])
        i = 0
        while i < len(turns):
            turn = turns[i]
            if turn["role"] != "user":
                i += 1
                continue
            t = datetime.fromisoformat(turn["time"])
            if t < one_week_ago:
                i += 1
                continue

            is_rewritten = turn.get("rewrite", "no") == "yes"
            total_prompt_count += 1
            if is_rewritten:
                rewrite_count += 1
                rewrite_breakdown[turn.get("rewrite_category", "Unknown")] += 1
            cat_counter[turn.get("semantic_category", "Other")] += 1

            tokens = clean_and_filter_keywords(turn.get("original_prompt") or turn.get("text"))
            for token in tokens:
                keyword_counter[token] += 1

            if i + 1 < len(turns) and turns[i + 1].get("rewrite", "no") == "yes":
                i += 2
            else:
                i += 1

    # Generate figures
    def chart_to_image(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        return ImageReader(buf)

    fig1, ax1 = plt.subplots()
    ax1.bar(cat_counter.keys(), cat_counter.values(), color="#42a5f5")
    ax1.set_title(f"Prompt Categories (Last {days} Days)")
    ax1.set_ylabel("Count")
    plt.xticks(rotation=45)
    fig1.tight_layout()
    img1 = chart_to_image(fig1)

    fig2, ax2 = plt.subplots()
    if rewrite_breakdown:
        ax2.pie(rewrite_breakdown.values(), labels=rewrite_breakdown.keys(),
                autopct='%1.1f%%', colors=plt.cm.Pastel1.colors)
        ax2.set_title(f"Rewrite Categories (Last {days} Days)")
    fig2.tight_layout()
    img2 = chart_to_image(fig2)

    # Create PDF
    pdf = io.BytesIO()
    c = canvas.Canvas(pdf, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, f"Weekly Report for Kid {kid_id}")

    c.setFont("Helvetica", 12)
    c.drawString(72, height - 100, f"Total prompts: {total_prompt_count}")
    c.drawString(72, height - 120, f"Rewritten prompts: {rewrite_count}")

    c.drawImage(img1, 72, height - 400, width=470, preserveAspectRatio=True)
    c.showPage()
    if rewrite_breakdown:
        c.drawImage(img2, 72, height - 400, width=470, preserveAspectRatio=True)

    # Keywords
    keywords = ", ".join([kw for kw, _ in keyword_counter.most_common(10)])
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, height - 420, "Top Keywords:")
    c.setFont("Helvetica", 11)
    c.drawString(72, height - 440, keywords)

    c.showPage()
    c.save()
    pdf.seek(0)

    return send_file(pdf, as_attachment=True,
                     download_name=f"{kid_id}_weekly_report.pdf",
                     mimetype="application/pdf")


# ======================================================================

if __name__ == "__main__":
    app.run(debug=True)
