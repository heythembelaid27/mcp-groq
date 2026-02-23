from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Any
import requests
import json
import os
import psycopg2
from psycopg2.extras import RealDictCursor

# ─── Config ─────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"
DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI()


# ─── Postgres ───────────────────────────────────────────────
def get_db():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            chat_id BIGINT PRIMARY KEY,
            step VARCHAR(50) DEFAULT 'idle',
            emails JSONB,
            selected_email JSONB,
            draft TEXT,
            search_query TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


@app.on_event("startup")
def startup():
    init_db()


# ─── Modèles ────────────────────────────────────────────────
class Email(BaseModel):
    id: str
    threadId: Optional[str] = None
    from_: Optional[str] = None
    subject: Optional[str] = None
    date: Optional[str] = None
    snippet: Optional[str] = None

    class Config:
        fields = {"from_": "from"}


class ChatRequest(BaseModel):
    chat_id: int
    message: str
    emails: Optional[List[Email]] = None
    events: Optional[List[Any]] = None


class ChatResponse(BaseModel):
    type: str
    text: str
    buttons: Optional[List[Any]] = None
    action: Optional[str] = None
    action_data: Optional[Any] = None


# ─── Groq ───────────────────────────────────────────────────
def call_groq(system_prompt: str, user_prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=60)
    data = r.json()
    content = data["choices"][0]["message"]["content"].strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].strip()
    return json.loads(content)


def emails_to_text(emails: List[Email]) -> str:
    text = ""
    for i, e in enumerate(emails, start=1):
        text += f"Email {i}:\nDe: {e.from_}\nSujet: {e.subject}\nSnippet: {e.snippet}\n\n"
    return text


# ─── Session ────────────────────────────────────────────────
def get_session(chat_id: int) -> dict:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions WHERE chat_id = %s", (chat_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return dict(row) if row else {"chat_id": chat_id, "step": "idle"}


def save_session(chat_id: int, **kwargs):
    conn = get_db()
    cur = conn.cursor()
    fields = ", ".join([f"{k} = %s" for k in kwargs])
    fields += ", updated_at = NOW()"
    values = list(kwargs.values())
    cur.execute(
        f"INSERT INTO sessions (chat_id) VALUES (%s) ON CONFLICT (chat_id) DO UPDATE SET {fields}",
        [chat_id] + values
    )
    conn.commit()
    cur.close()
    conn.close()


# ─── Health ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ─── Chat ───────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session = get_session(req.chat_id)
    step = session.get("step", "idle")
    msg = req.message.strip().lower()

    if msg.startswith("/"):
        save_session(req.chat_id, step="idle", selected_email=None, draft=None)
        step = "idle"

    if msg == "/inbox":
        return handle_inbox(req)
    elif msg == "/important":
        return handle_important(req)
    elif msg == "/search":
        return handle_search_start(req.chat_id)
    elif msg.startswith("/search "):
        return handle_search(req, req.message[8:].strip())
    elif msg == "/reply":
        return handle_reply_start(req)
    elif msg == "/today":
        return handle_today(req)
    elif msg == "/help":
        return handle_help()
    elif step == "search_waiting":
        return handle_search(req, req.message)
    elif step == "reply_select":
        return handle_reply_select(req, session)
    elif step == "reply_instruction":
        return handle_reply_instruction(req, session)
    else:
        return ChatResponse(type="text", text="❓ Commande inconnue. Tape /help pour voir les commandes.")


# ─── Handlers ───────────────────────────────────────────────
def handle_help() -> ChatResponse:
    return ChatResponse(
        type="text",
        text=(
            "🤖 *Commandes disponibles :*\n\n"
            "📬 /inbox — Résumé de ta boîte mail\n"
            "⭐ /important — Emails importants\n"
            "🔍 /search — Rechercher un email\n"
            "✉️ /reply — Répondre à un email\n"
            "📅 /today — Agenda du jour\n"
        )
    )


def handle_inbox(req: ChatRequest) -> ChatResponse:
    if not req.emails:
        return ChatResponse(type="action", text="", action="get_emails", action_data={"filter": "inbox"})
    system_prompt = (
        "Analyse ces emails et réponds en JSON : "
        "`summary` (string), `urgent` (array), `tasks` (array). "
        "JSON brut uniquement."
    )
    parsed = call_groq(system_prompt, emails_to_text(req.emails))
    urgent = "\n".join([f"🔥 {u}" for u in parsed.get("urgent", [])]) or "Aucun"
    tasks = "\n".join([f"✅ {t}" for t in parsed.get("tasks", [])]) or "Aucune"
    return ChatResponse(
        type="text",
        text=f"📬 *Résumé de ta boîte mail*\n\n{parsed.get('summary','')}\n\n🔥 *Urgents*\n{urgent}\n\n✅ *Tâches*\n{tasks}"
    )


def handle_important(req: ChatRequest) -> ChatResponse:
    if not req.emails:
        return ChatResponse(type="action", text="", action="get_emails", action_data={"filter": "important"})
    system_prompt = (
        "Identifie les emails vraiment importants (pas pubs ni newsletters). "
        "Réponds en JSON : `summary` (string), `emails` (array de strings). "
        "JSON brut uniquement."
    )
    parsed = call_groq(system_prompt, emails_to_text(req.emails))
    emails_list = "\n".join([f"📌 {e}" for e in parsed.get("emails", [])]) or "Aucun email important"
    return ChatResponse(
        type="text",
        text=f"⭐ *Emails importants*\n\n{parsed.get('summary','')}\n\n{emails_list}"
    )


def handle_search_start(chat_id: int) -> ChatResponse:
    save_session(chat_id, step="search_waiting")
    return ChatResponse(type="text", text="🔍 Que veux-tu rechercher ?")


def handle_search(req: ChatRequest, query: str) -> ChatResponse:
    if not req.emails:
        return ChatResponse(type="action", text="", action="get_emails", action_data={"filter": query})
    system_prompt = (
        f"Recherche les emails correspondant à : '{query}'. "
        "Réponds en JSON : `results` (array de strings), `summary` (string). "
        "JSON brut uniquement."
    )
    parsed = call_groq(system_prompt, emails_to_text(req.emails))
    results = "\n".join([f"📧 {r}" for r in parsed.get("results", [])]) or "Aucun résultat"
    save_session(req.chat_id, step="idle")
    return ChatResponse(
        type="text",
        text=f"🔍 *Résultats pour \"{query}\"*\n\n{parsed.get('summary','')}\n\n{results}"
    )


def handle_reply_start(req: ChatRequest) -> ChatResponse:
    if not req.emails:
        return ChatResponse(type="action", text="", action="get_emails", action_data={"filter": "inbox"})
    emails = req.emails[:10]
    save_session(req.chat_id, step="reply_select", emails=json.dumps([e.dict() for e in emails]))
    buttons = [[{
        "text": f"{i+1}. {(e.from_ or '')[:20]} — {(e.subject or '')[:25]}",
        "callback_data": f"reply_select|{i}"
    }] for i, e in enumerate(emails)]
    return ChatResponse(
        type="buttons",
        text="✉️ *Choisis un email pour répondre :*",
        buttons=buttons
    )


def handle_reply_select(req: ChatRequest, session: dict) -> ChatResponse:
    try:
        index = int(req.message.split("|")[1]) if "|" in req.message else int(req.message) - 1
        emails = json.loads(session.get("emails") or "[]")
        selected = emails[index]
        save_session(req.chat_id, step="reply_instruction", selected_email=json.dumps(selected))
        return ChatResponse(
            type="text",
            text=(
                f"✉️ *Email sélectionné :*\n"
                f"De : {selected.get('from_') or selected.get('from', '')}\n"
                f"Sujet : {selected.get('subject', '')}\n\n"
                f"✏️ Quelle est ton instruction ?\n"
                f"Ex: \"Réponds poliment que je suis absent\""
            )
        )
    except Exception:
        return ChatResponse(type="text", text="❌ Sélection invalide. Tape le numéro de l'email.")


def handle_reply_instruction(req: ChatRequest, session: dict) -> ChatResponse:
    selected = json.loads(session.get("selected_email") or "{}")
    system_prompt = (
        "Tu es un assistant email professionnel. Rédige une réponse email. "
        "Réponds en JSON : `draft` (string, texte complet). JSON brut uniquement."
    )
    user_prompt = (
        f"Email original :\nDe : {selected.get('from_') or selected.get('from', '')}\n"
        f"Sujet : {selected.get('subject', '')}\nContenu : {selected.get('snippet', '')}\n\n"
        f"Instruction : {req.message}"
    )
    parsed = call_groq(system_prompt, user_prompt)
    draft = parsed.get("draft", "")
    save_session(req.chat_id, step="reply_confirm", draft=draft)
    return ChatResponse(
        type="buttons",
        text=f"📝 *Brouillon rédigé :*\n\n{draft}",
        buttons=[[
            {"text": "✅ Envoyer", "callback_data": "reply_confirm|yes"},
            {"text": "❌ Annuler", "callback_data": "reply_confirm|no"}
        ]],
        action="confirm_reply",
        action_data={"email_id": selected.get("id"), "draft": draft}
    )


def handle_today(req: ChatRequest) -> ChatResponse:
    if not req.events:
        return ChatResponse(type="action", text="", action="get_calendar", action_data={"range": "today"})
    events_text = "\n".join([
        f"- {e.get('summary', 'Sans titre')} à {e.get('start', '')}"
        for e in req.events
    ])
    return ChatResponse(
        type="text",
        text=f"📅 *Agenda du jour :*\n\n{events_text or 'Aucun événement aujourd hui'}"
    )