# Vera Bot — Submission README

## Approach

**Core architecture**: A FastAPI HTTP server with Claude-powered 4-context composition, trigger-kind routing, and a conversation state machine.

### Message Composition
Every message is composed by a Claude inference call receiving all four context layers (category, merchant, trigger, customer) in a structured prompt. The system prompt encodes all 8 compulsion levers and all anti-patterns from the challenge brief as hard constraints. Output is validated JSON; on parse failure, we retry once with an explicit reminder.

**Routing**: Triggers are scored by kind (urgency × kind_priority) and deduplicated via suppression keys before each tick. A `TRIGGER_KIND_PRIORITY` table ensures regulation changes (score 9) fire before festival promo (score 4).

### Auto-Reply Detection
Two signals: (a) regex match against 14 known WA Business canned reply patterns (Hindi and English), (b) exact-match repeat detection across conversation history. On first auto-reply: one curiosity hook using real merchant stats. On second: graceful exit.

### Intent-Transition State Machine
`detect_intent()` checks for accept / reject / question / neutral on every incoming message. On explicit accept → immediate `ACTION` state, composing "doing it now" response without any qualifying question. On reject → warm exit. This eliminates the Pattern D failure documented in the brief.

### What Additional Context Would Have Helped Most
1. **Live WhatsApp Business reply templates** — knowing the exact Meta-approved template structure would improve `template_params` precision
2. **Real merchant conversation logs** — the 9 anonymized examples are great; 50 more would let me fine-tune the compulsion lever selection per trigger kind
3. **Category-wise conversion rates** — knowing which levers actually work per vertical (dentists vs salons vs gyms) would enable per-category routing logic beyond what peer_stats provide

## Files

| File | Description |
|---|---|
| `bot.py` | FastAPI server with all 5 endpoints + compose() entrypoint |
| `conversation_handlers.py` | Multi-turn state machine (auto-reply, intent-transition, language detection) |
| `submission.jsonl` | 30 test-pair messages, all 5 dimensions optimized |
| `requirements.txt` | Python dependencies |

## Running

```bash
pip install -r requirements.txt
pip install fastapi uvicorn pydantic
uvicorn bot:app --host 0.0.0.0 --port 8080
```
To view it in your browser:
Open this URL: http://localhost:8080/docs

Set `ANTHROPIC_API_KEY` in environment before starting.

## Key Design Decisions

- **Temperature 0** on all Claude calls for determinism
- **Suppression log** in-memory: prevents double-firing same trigger key
- **Context versioning**: idempotent on `(scope, context_id, version)` — higher version atomically replaces
- **30s budget**: `/v1/tick` processes triggers sorted by priority, capping at 20 actions; returns immediately if 0 actions available
- **Language auto-detection per turn** in conversation_handlers: updates `detected_language` dynamically as merchant switches languages mid-conversation
