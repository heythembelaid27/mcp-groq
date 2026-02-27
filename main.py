from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Any
import requests
import json
import os
import psycopg2
import xml.etree.ElementTree as ET
from psycopg2.extras import RealDictCursor
from datetime import datetime

# ─── Config ─────────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
GROQ_URL            = "https://api.groq.com/openai/v1/chat/completions"
MODEL               = "llama-3.3-70b-versatile"
DATABASE_URL        = os.getenv("DATABASE_URL")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
API_SECRET          = os.getenv("API_SECRET")
DEFAULT_CITY        = "Paris"
GOOGLE_NEWS_RSS     = "https://news.google.com/rss?hl=fr&gl=FR&ceid=FR:fr"
GOOGLE_NEWS_SEARCH_RSS = "https://news.google.com/rss/search?hl=fr&gl=FR&ceid=FR:fr&q={query}"

app = FastAPI()


# ─── Middleware : vérification header secret ─────────────────
@app.middleware("http")
async def check_secret(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    if API_SECRET:
        token = request.headers.get("X-API-Secret", "")
        if token != API_SECRET:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


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

    if "error" in data:
        raise RuntimeError(f"Groq error: {data['error']}")
    if "choices" not in data:
        raise RuntimeError(f"Groq missing choices: {data}")

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


def parse_session_json(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


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


# ─── Weather ────────────────────────────────────────────────
def fetch_weather(city: str) -> dict:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
        "lang": "fr"
    }
    r = requests.get(url, params=params, timeout=10)
    return r.json()


def weather_emoji(condition: str) -> str:
    condition = condition.lower()
    if "clear" in condition:
        return "☀️"
    elif "cloud" in condition:
        return "☁️"
    elif "rain" in condition or "drizzle" in condition:
        return "🌧️"
    elif "storm" in condition or "thunder" in condition:
        return "⛈️"
    elif "snow" in condition:
        return "❄️"
    elif "fog" in condition or "mist" in condition:
        return "🌫️"
    return "🌤️"


# ─── News (Google News RSS) ─────────────────────────────────
def fetch_news(topic: str = "", max_results: int = 5) -> list:
    if topic:
        url = GOOGLE_NEWS_SEARCH_RSS.format(query=requests.utils.quote(topic))
    else:
        url = GOOGLE_NEWS_RSS

    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()

    root = ET.fromstring(r.content)
    channel = root.find("channel")
    items = channel.findall("item") if channel is not None else []

    articles = []
    for item in items[:max_results]:
        title = item.findtext("title", "Sans titre")
        link = item.findtext("link", "")
        source_el = item.find("source")
        source = source_el.text if source_el is not None else "Google News"
        pub_date = item.findtext("pubDate", "")
        articles.append({
            "title": title,
            "url": link,
            "source": source,
            "pubDate": pub_date
        })
    return articles


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

    if msg == "/start":
        return handle_start()
    elif msg == "/inbox":
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
    elif msg.startswith("reply_select|"):
        return handle_reply_select(req, session)
    elif msg.startswith("reply_confirm|"):
        return handle_reply_confirm(req, session)
    elif step == "search_waiting":
        return handle_search(req, req.message)
    elif step == "reply_instruction":
        return handle_reply_instruction(req, session)
    else:
        return ChatResponse(type="text", text="❓ Commande inconnue. Tape /help pour voir les commandes.")


# ─── Handlers ───────────────────────────────────────────────
def handle_start() -> ChatResponse:
    return ChatResponse(
        type="text",
        text=(
            "👋 *Bonjour ! Je suis ton assistant personnel.*\n\n"
            "Je peux t'aider à gérer tes emails, ton agenda, la météo et les actualités.\n\n"
            "Tape /help pour voir toutes les commandes disponibles 🚀"
        )
    )


def handle_help() -> ChatResponse:
    return ChatResponse(
        type="text",
        text=(
            "🤖 *Commandes disponibles :*\n\n"
            "📅 /today — Briefing du jour (météo + agenda + actus)\n"
            "📬 /inbox — Résumé de ta boîte mail\n"
            "⭐ /important — Emails importants\n"
            "🔍 /search — Rechercher un email\n"
            "✉️ /reply — Répondre à un email\n"
        )
    )


def handle_inbox(req: ChatRequest) -> ChatResponse:
    if not req.emails:
        return ChatResponse(
            type="action",
            text="",
            action="get_emails",
            action_data={"filter": "inbox"}
        )
    try:
        system_prompt = (
            "Analyse ces emails et réponds en JSON : "
            "`summary` (string), `urgent` (array), `tasks` (array). "
            "JSON brut uniquement, aucun texte avant ou après."
        )
        parsed = call_groq(system_prompt, emails_to_text(req.emails))
        urgent = "\n".join([f"🔥 {u}" for u in parsed.get("urgent", [])]) or "Aucun"
        tasks = "\n".join([f"✅ {t}" for t in parsed.get("tasks", [])]) or "Aucune"
        return ChatResponse(
            type="text",
            text=f"📬 *Résumé de ta boîte mail*\n\n{parsed.get('summary','')}\n\n🔥 *Urgents*\n{urgent}\n\n✅ *Tâches*\n{tasks}"
        )
    except Exception as e:
        return ChatResponse(type="text", text=f"❌ Erreur lors de l'analyse : {str(e)}")


def handle_important(req: ChatRequest) -> ChatResponse:
    if not req.emails:
        return ChatResponse(
            type="action",
            text="",
            action="get_emails",
            action_data={"filter": "important"}
        )
    try:
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
    except Exception as e:
        return ChatResponse(type="text", text=f"❌ Erreur : {str(e)}")


def handle_search_start(chat_id: int) -> ChatResponse:
    save_session(chat_id, step="search_waiting")
    return ChatResponse(type="text", text="🔍 Que veux-tu rechercher ?")


def handle_search(req: ChatRequest, query: str) -> ChatResponse:
    if not req.emails:
        return ChatResponse(
            type="action",
            text="",
            action="get_emails",
            action_data={"filter": query}
        )
    try:
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
    except Exception as e:
        return ChatResponse(type="text", text=f"❌ Erreur : {str(e)}")


def handle_reply_start(req: ChatRequest) -> ChatResponse:
    if not req.emails:
        return ChatResponse(
            type="action",
            text="",
            action="get_emails",
            action_data={"filter": "inbox"}
        )
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
        index = int(req.message.split("|")[1])
        emails = parse_session_json(session.get("emails"), [])
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
    except Exception as e:
        print(f"DEBUG handle_reply_select error: {e}")
        return ChatResponse(type="text", text="❌ Sélection invalide. Tape le numéro de l'email.")


def handle_reply_instruction(req: ChatRequest, session: dict) -> ChatResponse:
    selected = parse_session_json(session.get("selected_email"), {})
    try:
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
    except Exception as e:
        return ChatResponse(type="text", text=f"❌ Erreur lors de la rédaction : {str(e)}")


def handle_reply_confirm(req: ChatRequest, session: dict) -> ChatResponse:
    action = req.message.split("|")[1] if "|" in req.message else ""
    if action == "no":
        save_session(req.chat_id, step="idle", draft=None, selected_email=None)
        return ChatResponse(type="text", text="❌ Brouillon annulé.")
    selected = parse_session_json(session.get("selected_email"), {})
    draft = session.get("draft", "")
    save_session(req.chat_id, step="idle", draft=None, selected_email=None)
    return ChatResponse(
        type="action",
        text="✅ Envoi en cours...",
        action="send_email",
        action_data={
            "email_id": selected.get("id"),
            "thread_id": selected.get("threadId"),
            "draft": draft
        }
    )


def handle_today(req: ChatRequest) -> ChatResponse:
    # Étape 1 — demander le calendar à n8n
    if req.events is None:
        return ChatResponse(
            type="action",
            text="",
            action="get_calendar",
            action_data={"range": "today"}
        )

    # Étape 2 — on a les events, on assemble météo + news + calendar ici
    now = datetime.now()
    days_fr   = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
    months_fr = ["janvier","février","mars","avril","mai","juin",
                 "juillet","août","septembre","octobre","novembre","décembre"]
    date_str = f"{days_fr[now.weekday()]} {now.day} {months_fr[now.month-1]} {now.year}"

    sections = [f"🗓️ *Briefing du jour — {date_str}*"]

    # ── Météo ──────────────────────────────────────────────
    try:
        w = fetch_weather(DEFAULT_CITY)
        if w.get("cod") == 200:
            temp  = round(w["main"]["temp"])
            feels = round(w["main"]["feels_like"])
            desc  = w["weather"][0]["description"].capitalize()
            wind  = round(w["wind"]["speed"] * 3.6)
            emoji = weather_emoji(w["weather"][0]["main"])
            sections.append(
                f"{emoji} *Météo à {w['name']}*\n"
                f"{temp}°C (ressenti {feels}°C) — {desc}\n"
                f"💨 {wind} km/h · 💧 {w['main']['humidity']}%"
            )
        else:
            sections.append("🌤️ *Météo* : données indisponibles")
    except Exception as e:
        sections.append(f"🌤️ *Météo* : erreur ({e})")

    # ── Agenda ─────────────────────────────────────────────
    if req.events:
        lines = []
        for e in req.events:
            title = e.get("summary", "Sans titre")
            start = e.get("start", {})
            time_str = start.get("dateTime", start.get("date", "")) if isinstance(start, dict) else str(start)
            try:
                dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                time_display = dt.strftime("%H:%M")
            except Exception:
                time_display = "toute la journée"
            lines.append(f"  🕐 *{time_display}* — {title}")
        sections.append("📅 *Agenda*\n" + "\n".join(lines))
    else:
        sections.append("📅 *Agenda* : Aucun événement aujourd'hui 🎉")

    # ── News ───────────────────────────────────────────────
    try:
        articles = fetch_news(max_results=4)
        if articles:
            lines = [f"  {i}. [{a['title']}]({a['url']})" for i, a in enumerate(articles, 1)]
            sections.append("📰 *Actus du jour*\n" + "\n".join(lines))
        else:
            sections.append("📰 *Actus* : aucune disponible")
    except Exception as e:
        sections.append(f"📰 *Actus* : erreur ({e})")

    return ChatResponse(type="text", text="\n\n".join(sections))


def handle_weather(city: str) -> ChatResponse:
    try:
        data = fetch_weather(city)
        if data.get("cod") != 200:
            return ChatResponse(type="text", text=f"❌ Ville introuvable : *{city}*")

        name = data["name"]
        country = data["sys"]["country"]
        temp = round(data["main"]["temp"])
        feels_like = round(data["main"]["feels_like"])
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"].capitalize()
        condition = data["weather"][0]["main"]
        wind = round(data["wind"]["speed"] * 3.6)  # m/s → km/h
        emoji = weather_emoji(condition)

        text = (
            f"{emoji} *Météo à {name}, {country}*\n\n"
            f"🌡️ Température : *{temp}°C* (ressenti {feels_like}°C)\n"
            f"💧 Humidité : {humidity}%\n"
            f"💨 Vent : {wind} km/h\n"
            f"🌥️ Ciel : {description}"
        )
        return ChatResponse(type="text", text=text)
    except Exception as e:
        return ChatResponse(type="text", text=f"❌ Erreur météo : {str(e)}")


def handle_news(topic: str) -> ChatResponse:
    try:
        articles = fetch_news(topic=topic, max_results=5)
        if not articles:
            return ChatResponse(type="text", text="📰 Aucune actualité trouvée.")

        header = f"📰 *Actualités{' sur *' + topic + '*' if topic else ' du jour'} :*\n\n"
        lines = []
        for i, a in enumerate(articles, start=1):
            title = a.get("title", "Sans titre")
            source = a.get("source", "")
            url = a.get("url", "")
            lines.append(f"{i}\\. [{title}]({url})\n   _{source}_")

        return ChatResponse(type="text", text=header + "\n\n".join(lines))
    except Exception as e:
        return ChatResponse(type="text", text=f"❌ Erreur news : {str(e)}")
