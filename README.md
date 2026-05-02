# Vera Bot — magicpin AI Challenge Submission v3.0

## Approach

**Claude-powered 4-context composer** with trigger-kind-specific prompting, full multi-turn state machine, and role-aware reply routing.

### Architecture

1. **Composition engine** (`bot.py → compose_message`): Every `/v1/tick` trigger is composed via a structured prompt that feeds all 4 context layers (category, merchant, trigger, customer) to Claude Sonnet. The prompt includes trigger-kind-specific instructions (e.g. for `regulation_change`: lead with the regulatory fact + deadline; for `research_digest`: cite trial_n + % improvement + source; for `perf_dip`: name the exact metric and offer to diagnose).

2. **State machine** (`conversation_handlers.py`): Each conversation has a state (PITCH → ACTION → CLOSED). Fast-path regex handles clear intents (accept/reject/hostile/auto-reply) without an LLM call. Claude handles question/neutral turns with full context.

3. **Auto-reply detection**: Detects on first hit, sends one direct curiosity hook. Exits gracefully on second auto-reply hit — no wasted turns.

4. **Role routing**: `/v1/reply` checks `from_role` — customer replies go to `compose_customer_reply` which addresses the customer by name from their context, confirms specific booking details (date/time from trigger payload), and uses `send_as: merchant_on_behalf`.

5. **Anti-repetition guard**: After every composition, checks if body matches any prior Vera message in the conversation. Re-prompts with temperature=0.3 if duplicate detected.

### Key design decisions

- **No expiry filtering on triggers**: Trust the judge's `available_triggers` list — the judge only sends active triggers. Filtering by `expires_at` against server wall-clock caused all triggers to be dropped (clock mismatch).
- **Trigger-kind-specific system prompt sections**: Each of the 20+ trigger kinds gets its own instruction block rather than a generic "compose a message". This drives specificity scores.
- **Suppression via judge's key, not our own**: We use the trigger's `suppression_key` directly to prevent re-firing, not our own derived key.
- **Synthetic trigger shells for context-not-yet-loaded**: If a trigger ID arrives in `available_triggers` before its context push, we infer the kind from the ID string and still compose rather than skip.

### Tradeoffs

- Using Claude Sonnet for every composition adds latency (~3-5s/message) but dramatically improves specificity and category fit vs. template-based approaches.
- Full context in prompt (~2000 tokens) vs. retrieval: with 5 categories and 50 merchants, full-context fits comfortably. At 500+ merchants, retrieval would be needed.
- Temperature=0 for determinism on composition; 0.3 for retry-on-duplicate only.

### What additional context would have helped

- Real merchant phone numbers / WhatsApp Business API response patterns (for better auto-reply signature matching)
- Historical engagement rates by trigger kind (to tune which triggers to fire vs. suppress)
- The judge's exact scoring rubric weights per trigger kind
