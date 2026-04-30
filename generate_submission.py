#!/usr/bin/env python3
"""
Generate submission.jsonl for the 30 test pairs.
Uses the same Claude-powered composition logic as bot.py.
"""

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

EXPANDED = Path("/tmp/magicpin/expanded")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

SYSTEM_PROMPT = """You are Vera, magicpin's AI merchant assistant. You compose WhatsApp messages that engage Indian merchants.

RULES (non-negotiable):
1. Body must be concise and WhatsApp-native — no markdown headers, no bullet points in the message, plain text with line breaks only.
2. ONE primary CTA at the end. Binary (Reply YES / STOP) for action triggers. Open-ended for information triggers. No CTA for pure digest.
3. Service+price framing ALWAYS beats percentage discounts. "Haircut @ ₹99" > "10% off".
4. Never fabricate data. Only use what's in the contexts.
5. No taboo words from the category voice (guaranteed, cure, 100% safe, miracle, best in city).
6. No long preambles ("I hope you're doing well"). Get to value in sentence 1.
7. Anchor on at least ONE verifiable concrete fact (number, date, source, stat).
8. Peer/colleague tone, not promotional. Clinical vocabulary allowed in medical categories.
9. Hindi-English code-mix when merchant language includes "hi". Match the merchant's language register.

COMPULSION LEVERS (use 1-3 per message):
- Specificity / verifiability (numbers, dates, source citations)
- Loss aversion ("you're missing X", "before this closes")
- Social proof ("3 dentists in your area did Y this month")
- Effort externalization ("I've already drafted it — just say go")
- Curiosity ("want to see who?", "want the full breakdown?")
- Reciprocity (I noticed X, thought you'd want to know)
- Asking the merchant (what's most in demand this week?)
- Single binary commit (Reply YES / STOP)

ANTI-PATTERNS (judge penalizes these):
- Generic offers ("Flat 30% off") — use service+price instead
- Multiple CTAs
- Buried CTA (must be last sentence)
- Promotional tone for clinical categories (dentists, doctors)
- Hallucinated data
- Long preambles

OUTPUT FORMAT (JSON only, no preamble, no markdown fences):
{
  "body": "<the WhatsApp message>",
  "cta": "open_ended" | "binary_yes_stop" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "<key>",
  "rationale": "<2-3 sentence explanation of compulsion levers + why this message>"
}"""

def call_claude(system: str, user: str) -> str:
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 800,
        "temperature": 0.0,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=28) as resp:
        result = json.loads(resp.read())
    return result["content"][0]["text"].strip()

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

def find_file(directory: Path, name: str) -> Path:
    """Find a JSON file by merchant/trigger/customer id."""
    matches = list(directory.glob(f"{name}.json"))
    if matches:
        return matches[0]
    # Try partial match
    all_files = list(directory.glob("*.json"))
    for f in all_files:
        if name in f.stem:
            return f
    raise FileNotFoundError(f"Cannot find {name} in {directory}")

def compose_for_pair(test_id: str, merchant_id: str, trigger_id: str, customer_id: str | None) -> dict:
    """Compose a message for one test pair."""
    
    # Load merchant
    merchant = load_json(find_file(EXPANDED / "merchants", merchant_id))
    category_slug = merchant.get("category_slug", "dentists")
    category = load_json(EXPANDED / "categories" / f"{category_slug}.json")
    
    # Load trigger
    trigger = load_json(find_file(EXPANDED / "triggers", trigger_id))
    
    # Load customer if present
    customer = None
    if customer_id:
        try:
            customer = load_json(find_file(EXPANDED / "customers", customer_id))
        except FileNotFoundError:
            pass
    
    # Find digest item for trigger
    trigger_payload = trigger.get("payload", {})
    top_item_id = trigger_payload.get("top_item_id")
    digest_item = None
    if top_item_id:
        for d in category.get("digest", []):
            if d.get("id") == top_item_id:
                digest_item = d
                break
    
    # Customer section
    customer_section = ""
    if customer:
        customer_section = f"\nCUSTOMER CONTEXT:\n{json.dumps(customer, ensure_ascii=False, indent=2)}\nNote: send_as MUST be 'merchant_on_behalf'"

    prompt = f"""TEST PAIR: {test_id}
TRIGGER KIND: {trigger.get("kind")}
TRIGGER SCOPE: {trigger.get("scope")}
TRIGGER URGENCY: {trigger.get("urgency")}/5

CATEGORY CONTEXT:
slug: {category.get("slug")}
display_name: {category.get("display_name")}
voice_tone: {category.get("voice", {}).get("tone")}
vocab_taboo: {category.get("voice", {}).get("vocab_taboo", [])}
vocab_allowed: {category.get("voice", {}).get("vocab_allowed", [])}
peer_stats: {json.dumps(category.get("peer_stats", {}), ensure_ascii=False)}
offer_catalog (use these price anchors, never invent prices): {json.dumps(category.get("offer_catalog", [])[:5], ensure_ascii=False)}
seasonal_beats: {json.dumps(category.get("seasonal_beats", []), ensure_ascii=False)}
trend_signals: {json.dumps(category.get("trend_signals", []), ensure_ascii=False)}
digest_item_for_trigger: {json.dumps(digest_item, ensure_ascii=False) if digest_item else "None"}

MERCHANT CONTEXT:
merchant_id: {merchant.get("merchant_id")}
name: {merchant.get("identity", {}).get("name")}
owner_first_name: {merchant.get("identity", {}).get("owner_first_name")}
city: {merchant.get("identity", {}).get("city")}
locality: {merchant.get("identity", {}).get("locality")}
languages: {merchant.get("identity", {}).get("languages", ["en"])}
verified_gbp: {merchant.get("identity", {}).get("verified")}
subscription: {json.dumps(merchant.get("subscription", {}), ensure_ascii=False)}
performance_30d: {json.dumps(merchant.get("performance", {}), ensure_ascii=False)}
active_offers: {json.dumps([o for o in merchant.get("offers", []) if o.get("status") == "active"], ensure_ascii=False)}
customer_aggregate: {json.dumps(merchant.get("customer_aggregate", {}), ensure_ascii=False)}
signals: {merchant.get("signals", [])}
review_themes: {json.dumps(merchant.get("review_themes", []), ensure_ascii=False)}
conversation_history_last_3: {json.dumps(merchant.get("conversation_history", [])[-3:], ensure_ascii=False)}

TRIGGER CONTEXT:
{json.dumps(trigger, ensure_ascii=False, indent=2)}
{customer_section}

SUPPRESSION KEY: {trigger.get("suppression_key", f"auto:{trigger_id}")}

Now compose the optimal message for this pair. Output valid JSON only."""

    raw = call_claude(SYSTEM_PROMPT, prompt)
    raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    
    try:
        result = json.loads(raw.strip())
    except json.JSONDecodeError as e:
        print(f"  JSON parse error for {test_id}: {e}")
        # Try to extract JSON from response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
        else:
            raise
    
    return result

def main():
    pairs_data = load_json(EXPANDED / "test_pairs.json")
    pairs = pairs_data["pairs"]
    
    output_lines = []
    
    for i, pair in enumerate(pairs):
        test_id = pair["test_id"]
        merchant_id = pair["merchant_id"]
        trigger_id = pair["trigger_id"]
        customer_id = pair.get("customer_id")
        
        print(f"[{i+1:02d}/30] Composing {test_id}: {merchant_id[:30]} + {trigger_id[:35]}")
        
        try:
            result = compose_for_pair(test_id, merchant_id, trigger_id, customer_id)
            
            line = {
                "test_id": test_id,
                "body": result.get("body", ""),
                "cta": result.get("cta", "open_ended"),
                "send_as": result.get("send_as", "vera"),
                "suppression_key": result.get("suppression_key", f"auto:{trigger_id}"),
                "rationale": result.get("rationale", ""),
            }
            output_lines.append(line)
            print(f"  ✓ {result.get('cta')} | {len(result.get('body',''))} chars | {result.get('send_as')}")
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            # Fallback entry
            output_lines.append({
                "test_id": test_id,
                "body": f"[Composition error for {test_id}]",
                "cta": "open_ended",
                "send_as": "vera",
                "suppression_key": f"error:{test_id}",
                "rationale": f"Error: {str(e)[:100]}",
            })
        
        time.sleep(0.5)  # Rate limit courtesy
    
    # Write JSONL
    out_path = Path("/home/claude/vera_bot/submission.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Written {len(output_lines)} lines to {out_path}")
    return output_lines

if __name__ == "__main__":
    main()
