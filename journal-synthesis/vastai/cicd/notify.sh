#!/usr/bin/env bash
# =====================================================================
# notify.sh  (PHASE 0 - status pings, optional)
# =====================================================================
# POST a short JSON status line to $NOTIFY_WEBHOOK on start/success/
# failure. INFRASTRUCTURE ONLY: it reports run lifecycle, NEVER a
# scientific result, NEVER a paper number, NEVER a verdict to act on.
#
# Usage:
#   notify.sh <event> [message...]
#     event   : one of  start | success | failure | info
#     message : free text (kept short; no secrets)
#
# Behaviour:
#   * $NOTIFY_WEBHOOK unset  -> NO-OP with a single clear log line.
#     The pipeline must work perfectly with no webhook configured.
#   * Webhook set            -> one best-effort POST. A failed notify
#     NEVER fails the run (notifications are not load-bearing).
#
# SECRETS: the webhook URL itself comes from the env var only; it is
# never echoed in full. No API key is used or printed here.
# =====================================================================
set -uo pipefail

EVENT="${1:-info}"
shift || true
MSG="${*:-}"

log() { echo "[notify] $*" >&2; }

# Redact a URL for logs: keep scheme+host, drop the secret path/token.
_redact() {
  printf '%s' "$1" | sed -E 's#(https?://[^/]+/).*#\1<redacted>#'
}

WEBHOOK="${NOTIFY_WEBHOOK:-}"
if [ -z "$WEBHOOK" ]; then
  log "NOTIFY_WEBHOOK unset -> skipping notification (no-op). "\
"event=$EVENT msg='$MSG'"
  exit 0
fi

HOST="$(hostname 2>/dev/null || echo unknown-host)"
TS="$(date -u +%FT%TZ)"

# A generic JSON shape that also satisfies Slack/Discord (both accept a
# top-level "text"/"content"; we send "text" + structured fields so a
# generic endpoint gets everything and Slack renders "text").
PAYLOAD="$(EVENT="$EVENT" MSG="$MSG" HOST="$HOST" TS="$TS" python3 - <<'PY'
import json, os
text = f"[journal-experiments] {os.environ['EVENT'].upper()}: " \
       f"{os.environ['MSG']}".strip()
print(json.dumps({
    "text": text,            # generic + Slack
    "content": text,         # Discord
    "event": os.environ["EVENT"],
    "message": os.environ["MSG"],
    "host": os.environ["HOST"],
    "utc": os.environ["TS"],
    "kind": "infra-status",  # NOT a scientific result
}))
PY
)"

log "POST status event=$EVENT to $(_redact "$WEBHOOK")"
if command -v curl >/dev/null 2>&1; then
  curl -fsS -m 15 -X POST -H "Content-Type: application/json" \
       --data "$PAYLOAD" "$WEBHOOK" >/dev/null 2>&1 \
    && log "notification sent" \
    || log "notification POST failed (non-fatal; run continues)"
else
  python3 - "$WEBHOOK" "$PAYLOAD" <<'PY' 2>/dev/null \
    && log "notification sent" \
    || log "notification POST failed (non-fatal; run continues)"
import sys, urllib.request
url, payload = sys.argv[1], sys.argv[2].encode()
req = urllib.request.Request(
    url, data=payload, method="POST",
    headers={"Content-Type": "application/json"})
urllib.request.urlopen(req, timeout=15).read()
PY
fi
exit 0
