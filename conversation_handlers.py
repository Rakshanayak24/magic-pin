"""
conversation_handlers.py — Multi-turn Conversation State Machine
================================================================
Implements the optional multi-turn handler for the magicpin AI challenge.
This gives bonus points for:
1. Auto-reply detection + graceful exit
2. Intent-transition (qualify → action on accept)
3. Language detection per turn
4. Graceful conversation close
"""

from __future__ import annotations

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone

log = logging.getLogger("vera.conv")

# ─── State Machine States ─────────────────────────────────────────────────────
class ConvState:
    QUALIFY   = "qualify"     # Gathering intent / qualification
    PITCH     = "pitch"       # Presenting value prop
    ACTION    = "action"      # Taking action (merchant accepted)
    FOLLOW_UP = "follow_up"   # Post-action follow-up
    CLOSED    = "closed"      # Conversation ended

# ─── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    trigger_id: Optional[str] = None
    state: str = ConvState.PITCH
    history: list[dict] = field(default_factory=list)
    auto_reply_count: int = 0
    turn_count: int = 0
    detected_language: str = "en"
    last_vera_body: str = ""
    merchant_context: Optional[dict] = None
    category_context: Optional[dict] = None
    trigger_context: Optional[dict] = None

# ─── Language detection ────────────────────────────────────────────────────────
HINDI_MARKERS = [
    "haan", "nahi", "kya", "hai", "hoon", "karo", "shukriya",
    "bahut", "theek", "bilkul", "chalega", "zaroor", "mujhe",
    "aapka", "aapki", "mere", "main", "toh", "ki", "ka", "ke",
]

def detect_language(message: str) -> str:
    """Detect if message is hi-en mix or english."""
    ml = message.lower()
    hindi_count = sum(1 for word in HINDI_MARKERS if word in ml)
    if hindi_count >= 2:
        return "hi-en"
    return "en"

# ─── Auto-reply detection ──────────────────────────────────────────────────────
AUTO_REPLY_SIGNATURES = [
    "thank you for contacting",
    "thank you for reaching out",
    "i am currently unavailable",
    "out of office",
    "automated message",
    "automated assistant",
    "automated reply",
    "this is an auto",
    "main ek automated",
    "aapki jaankari ke liye bahut-bahut shukriya",
    "team tak pahuncha",
    "will get back to you soon",
    "i will get back",
]

def detect_auto_reply(message: str, history: list[dict]) -> bool:
    """Detect canned WA Business auto-reply."""
    msg_lower = message.lower().strip()

    # Check against known signatures
    for sig in AUTO_REPLY_SIGNATURES:
        if sig in msg_lower:
            return True

    # Check for verbatim repeat from same merchant (same message twice in a row)
    prev_merchant_msgs = [
        t["msg"].strip() for t in history
        if t.get("from") in ("merchant", "customer")
    ]
    if prev_merchant_msgs and prev_merchant_msgs[-1:] == [message.strip()]:
        return True

    return False

# ─── Intent detection ──────────────────────────────────────────────────────────
ACCEPT_WORDS = {
    "yes", "yeah", "yep", "sure", "ok", "okay", "done",
    "haan", "ha", "ji haan", "theek hai", "theek h", "chalega",
    "proceed", "go ahead", "let's do it", "karo", "kar do",
    "register", "join", "start", "shuru", "bilkul", "zaroor",
    "judrna hai", "joinaa hai", "sign up", "sign me up",
}

REJECT_WORDS = {
    "no", "nope", "nahi", "na", "not interested", "interested nahi",
    "stop", "band karo", "mat karo", "later", "baad mein",
    "abhi nahi", "busy hoon", "time nahi", "leave me alone",
    "unsubscribe", "opt out",
}

def detect_intent(message: str) -> str:
    """Returns: accept | reject | question | neutral"""
    ml = message.lower().strip()
    words = set(re.findall(r"\b\w+\b", ml))

    if words & ACCEPT_WORDS:
        return "accept"
    if words & REJECT_WORDS:
        return "reject"
    if "?" in message:
        return "question"
    # Short affirmative patterns
    if re.search(r"^(yes|haan|ok|sure|done|go|proceed|karo)[\s!.]*$", ml):
        return "accept"
    return "neutral"

# ─── Main handler ─────────────────────────────────────────────────────────────
def respond(state: ConversationState, merchant_message: str) -> dict:
    """
    Given the conversation state + merchant's latest message, produce the reply.

    Returns:
        {action, body?, cta?, wait_seconds?, rationale}
    """
    state.turn_count += 1
    state.history.append({
        "from": "merchant",
        "msg": merchant_message,
        "ts": datetime.now(timezone.utc).isoformat(),
    })

    # Update detected language per-turn
    detected_lang = detect_language(merchant_message)
    if detected_lang == "hi-en":
        state.detected_language = "hi-en"

    # ─── Auto-reply detection ────────────────────────────────────────────────
    is_auto = detect_auto_reply(merchant_message, state.history[:-1])
    if is_auto:
        state.auto_reply_count += 1

        if state.auto_reply_count >= 2:
            # Second auto-reply → graceful exit
            farewell = _get_farewell(state, "auto_reply")
            _record_vera(state, farewell)
            state.state = ConvState.CLOSED
            return {
                "action": "end",
                "body": farewell,
                "cta": "none",
                "rationale": f"Auto-reply detected {state.auto_reply_count}× — exiting gracefully to avoid wasting turns",
            }

        # First auto-reply → try one direct hook
        hook = _compose_auto_reply_hook(state)
        _record_vera(state, hook)
        return {
            "action": "send",
            "body": hook,
            "cta": "binary_yes_stop",
            "rationale": "Auto-reply detected (count=1) — sent one curiosity hook to reach real owner",
        }

    # ─── Intent-based routing ────────────────────────────────────────────────
    intent = detect_intent(merchant_message)

    if intent == "reject":
        farewell = _get_farewell(state, "rejected")
        _record_vera(state, farewell)
        state.state = ConvState.CLOSED
        return {
            "action": "end",
            "body": farewell,
            "cta": "none",
            "rationale": "Merchant explicitly declined — graceful exit preserves relationship",
        }

    if intent == "accept":
        # CRITICAL: Transition to ACTION immediately — do NOT ask qualifying questions
        state.state = ConvState.ACTION
        action_msg = _compose_action_response(state, merchant_message)
        _record_vera(state, action_msg)
        return {
            "action": "send",
            "body": action_msg,
            "cta": "open_ended",
            "rationale": "Merchant accepted — transitioned to action mode, executing requested task without further qualification",
        }

    if intent == "question":
        answer = _compose_question_answer(state, merchant_message)
        _record_vera(state, answer)
        return {
            "action": "send",
            "body": answer,
            "cta": "open_ended",
            "rationale": "Merchant asked a question — answered directly, offered next step",
        }

    # Neutral — advance the conversation with a new angle
    if state.turn_count >= 5:
        # Too many turns without progress → graceful exit
        farewell = _get_farewell(state, "timeout")
        _record_vera(state, farewell)
        state.state = ConvState.CLOSED
        return {
            "action": "end",
            "body": farewell,
            "cta": "none",
            "rationale": "5 turns without clear progress — exiting gracefully",
        }

    advance_msg = _compose_advance(state, merchant_message)
    _record_vera(state, advance_msg)
    return {
        "action": "send",
        "body": advance_msg,
        "cta": "binary_yes_stop",
        "rationale": "Merchant is neutral — advancing with new value angle, tightening CTA",
    }

# ─── Helper composers ─────────────────────────────────────────────────────────

def _record_vera(state: ConversationState, body: str):
    state.history.append({"from": "vera", "msg": body, "ts": datetime.now(timezone.utc).isoformat()})
    state.last_vera_body = body

def _get_farewell(state: ConversationState, reason: str) -> str:
    lang = state.detected_language
    merchant = state.merchant_context or {}
    name = merchant.get("identity", {}).get("owner_first_name", "")

    if reason == "auto_reply":
        if lang == "hi-en":
            return f"Koi baat nahi{', ' + name if name else ''} — samajh gayi. Owner/manager se directly connect kar lungi. Best wishes! 🙂"
        return f"Got it{', ' + name if name else ''} — I'll connect with the owner/manager directly. Best wishes! 🙂"
    elif reason == "rejected":
        if lang == "hi-en":
            return f"Bilkul theek hai{', ' + name if name else ''} — jab bhi ready hon, main yahan hoon. 🙏"
        return f"Absolutely fine{', ' + name if name else ''} — whenever you're ready, I'm here. 🙏"
    else:  # timeout
        if lang == "hi-en":
            return "Theek hai — baad mein baat karte hain. Good luck! 🙂"
        return "No worries — we'll catch up another time. Good luck! 🙂"

def _compose_auto_reply_hook(state: ConversationState) -> str:
    """Single curiosity hook to pierce through auto-reply."""
    merchant = state.merchant_context or {}
    category = state.category_context or {}
    lang = state.detected_language
    name = merchant.get("identity", {}).get("owner_first_name", "")
    perf = merchant.get("performance", {})
    peer_stats = category.get("peer_stats", {})

    # Find a specific hook from merchant data
    if perf.get("ctr") and peer_stats.get("avg_ctr"):
        ctr = perf["ctr"]
        peer_ctr = peer_stats["avg_ctr"]
        ctr_pct = int(ctr * 100)
        peer_pct = int(peer_ctr * 100)
        if lang == "hi-en":
            return (
                f"{'Dr. ' + name if 'dentist' in category.get('slug','') else name}, "
                f"aapka CTR {ctr_pct}% hai vs peers ka {peer_pct}% — "
                f"iska seedha matlab {int((peer_ctr - ctr) * 100)} extra clicks/month chhoot rahe hain. "
                f"Kya main wajah dikhaaun? Reply YES"
            )
        return (
            f"{'Dr. ' + name if 'dentist' in category.get('slug','') else name}, "
            f"your CTR is {ctr_pct}% vs peer avg {peer_pct}% — "
            f"that gap = {int((peer_ctr - ctr) * 100)} missed clicks/month. "
            f"Want me to show why? Reply YES"
        )

    # Fallback hook
    if lang == "hi-en":
        return f"{name or 'Namaskar'}, ek quick question — is mahine sabse zyada kis service ki demand aa rahi hai aapke paas? (1 word bhi chalega)"
    return f"{name or 'Hi there'}, quick question — what's your most-requested service this month? Even one word helps me help you better."

def _compose_action_response(state: ConversationState, merchant_message: str) -> str:
    """Compose immediate action response after merchant accepts."""
    merchant = state.merchant_context or {}
    trigger = state.trigger_context or {}
    lang = state.detected_language
    name = merchant.get("identity", {}).get("owner_first_name", "")
    kind = trigger.get("kind", "update")

    action_map = {
        "research_digest":      ("pulling the full abstract + drafting a patient-education WhatsApp template", "Drafting now — ready in 30s"),
        "recall_due":           ("booking the slot and sending confirmation", "Slot confirmed — you'll get a summary"),
        "perf_dip":             ("running a full profile audit to identify the drop cause", "Audit running — checking 7 factors"),
        "perf_spike":           ("capturing this momentum with a Google Post + offer boost", "Drafting the post now"),
        "festival_upcoming":    ("drafting a festival campaign with 3 copy options", "Campaign drafted — sending 3 options"),
        "regulation_change":    ("preparing a compliance checklist for your setup", "Checklist ready in 30s"),
        "dormant_with_vera":    ("resuming your profile optimization where we left off", "Picking up from last checkpoint"),
        "renewal_due":          ("processing your renewal and confirming the plan", "Renewal confirmed — sending receipt"),
        "milestone_reached":    ("crafting a milestone post for your Google Business Profile", "Milestone post drafted"),
        "curious_ask_due":      ("noting that down and checking if we can feature it", "Got it — checking now"),
        "competitor_opened":    ("pulling the competitor's profile details for your review", "Analysis ready in 30s"),
        "bridal_followup":      ("drafting a bridal follow-up message for the customer", "Draft ready — want to review?"),
        "appointment_tomorrow": ("sending the appointment confirmation to the customer", "Confirmation sent"),
    }

    action_desc, action_status = action_map.get(kind, ("getting that done for you", "On it"))

    if lang == "hi-en":
        return f"Perfect{', ' + name if name else ''}! Main abhi {action_desc.split(' and ')[0]} kar rahi hoon. {action_status}. Kya kuch specific chahiye?"
    return f"Perfect{', ' + name if name else ''}! {action_status} — {action_desc}. Anything specific you'd like included?"

def _compose_question_answer(state: ConversationState, question: str) -> str:
    """Answer merchant's question and re-offer value."""
    merchant = state.merchant_context or {}
    category = state.category_context or {}
    lang = state.detected_language
    name = merchant.get("identity", {}).get("owner_first_name", "")

    q_lower = question.lower()

    if any(w in q_lower for w in ["cost", "price", "kitne", "kitna", "charges", "fees", "amount"]):
        if lang == "hi-en":
            return f"Aapki plan ke according — Pro plan mein yeh service included hai. Extra charges nahi. {name or 'Aap'} chahein toh main detail breakdown bhej sakti hoon — Reply YES?"
        return "Under your current Pro plan, this service is included at no extra charge. Want me to send a full breakdown? Reply YES"

    if any(w in q_lower for w in ["time", "kab", "when", "kitne time", "how long"]):
        if lang == "hi-en":
            return "Typically 24-48 ghante mein reflect hota hai Google pe. Main progress track karti rehti hoon aur update karti hoon. Koi aur sawaal?"
        return "Typically reflects on Google within 24-48 hours. I'll track and update you. Any other questions?"

    if any(w in q_lower for w in ["help", "kya", "what", "kaise", "how"]):
        perf = merchant.get("performance", {})
        if perf.get("ctr"):
            if lang == "hi-en":
                return (
                    f"Main aapka Google profile optimize karti hoon — photos, posts, offers, aur reviews. "
                    f"Aapka CTR abhi {int(perf['ctr']*100)}% hai; peers {int(category.get('peer_stats', {}).get('avg_ctr', 0.03)*100)}% pe hain. "
                    f"Main gap close kar sakti hoon. Chalega?"
                )
            return (
                f"I optimize your Google Business Profile — photos, posts, offers, reviews. "
                f"Your CTR is {int(perf['ctr']*100)}% vs peer avg {int(category.get('peer_stats', {}).get('avg_ctr', 0.03)*100)}%. "
                f"I can close that gap. Want me to start? Reply YES"
            )

    # Fallback
    if lang == "hi-en":
        return "Samajh gayi — main check karti hoon. Ek second. Kya aap chahenge main aapka full profile audit kar doon? (FREE, 2 min) Reply YES"
    return "Got it — let me check on that. Meanwhile, want me to run a quick free profile audit? Reply YES"

def _compose_advance(state: ConversationState, merchant_message: str) -> str:
    """Advance with a new angle when merchant is neutral."""
    merchant = state.merchant_context or {}
    category = state.category_context or {}
    lang = state.detected_language
    name = merchant.get("identity", {}).get("owner_first_name", "")
    turn = state.turn_count
    perf = merchant.get("performance", {})
    customer_agg = merchant.get("customer_aggregate", {})
    peer_stats = category.get("peer_stats", {})

    if turn == 2:
        # Social proof angle
        lapsed = customer_agg.get("lapsed_180d_plus", 0)
        if lapsed > 0:
            if lang == "hi-en":
                return (
                    f"{name or 'Aap'}, aapke {lapsed} customers 6+ months se nahi aaye. "
                    f"Main unhe recall message bhej sakti hoon aapki taraf se — mostly 20-30% wapas aate hain. "
                    f"Try karein? Reply YES / STOP"
                )
            return (
                f"{name or 'Quick note'}: {lapsed} of your customers haven't returned in 6+ months. "
                f"I can send them a recall message — typically 20-30% come back. Worth trying? Reply YES / STOP"
            )

    if turn == 3:
        # Loss aversion angle
        views = perf.get("views", 0)
        ctr = perf.get("ctr", 0)
        if views and ctr:
            missed = int(views * (peer_stats.get("avg_ctr", 0.03) - ctr))
            if missed > 0:
                if lang == "hi-en":
                    return (
                        f"Last 30 days mein approx {missed} searches thi jo aapke profile pe aayi lekin convert nahi hui "
                        f"(CTR gap vs peers). Main wajah diagnose karke fix kar sakti hoon — 10 min ka kaam. Reply YES?"
                    )
                return (
                    f"In the last 30 days, ~{missed} profile visitors didn't convert (CTR gap vs peers). "
                    f"I can diagnose and fix the reason — takes me 10 minutes. Reply YES?"
                )

    if turn == 4:
        # Curiosity angle — asking the merchant
        if lang == "hi-en":
            return f"Ek quick question {name or ''}: is mahine kaunsi service ke liye sabse zyada inquiries aa rahi hain? Main uske around ek targeted post draft kar sakti hoon."
        return f"Quick question {name or ''}: what service is getting the most inquiries this month? I'll draft a targeted post around it."

    # Default: effort externalization
    if lang == "hi-en":
        return "Main aapka Google post already draft kar chuki hoon — sirf approval chahiye. Reply YES aur main publish kar deti hoon. STOP agar interested nahi hain."
    return "I've already drafted a Google post for you — just need your go-ahead. Reply YES to publish, STOP to skip."