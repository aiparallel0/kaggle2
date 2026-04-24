/*
 * api/dispatch.c — HTTP request dispatcher with per-endpoint rate limiting.
 *
 * Fix (Bug 1 — rate limiter):
 *
 *   Original code used a SINGLE rate-limit bucket whose key was the raw
 *   X-Forwarded-For value (or the literal string "unknown" when the header
 *   was absent).  This had two failure modes that composed:
 *
 *   a) All four auth endpoints shared one bucket.  A click-storm on
 *      /auth/login burned the budget for /auth/register and /auth/forgot
 *      too, so legitimate registration attempts were blocked while the
 *      account already existed in the DB.
 *
 *   b) When nginx did not set X-Forwarded-For (e.g. direct TCP connection,
 *      or a misconfigured proxy), every client collapsed onto the key
 *      "unknown" — a single global 10-request budget for the whole site.
 *
 * This version:
 *   - Builds the rate-limit key as  "<endpoint_tag>:<ip>"  so each endpoint
 *     has its own independent quota.
 *   - Falls back to req.peer_addr (the TCP peer IP, always available) when
 *     req.fwd_for is empty, instead of using the string "unknown".
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "../core/types.h"

/* ── Stubs for surrounding infrastructure ────────────────────────────────
 *
 * In the real build these are provided by their respective translation units.
 * They are declared here only so that this file compiles stand-alone.
 */
extern int  rl_check(const char *key, int limit);
extern Response json_error(int status, const char *code);
extern Response handle_register(Request req);
extern Response handle_login(Request req);
extern Response handle_forgot(Request req);
extern Response handle_reset(Request req);
extern Response handle_goals(Request req);
extern Response not_found(void);

/* ── Rate-limit helpers ──────────────────────────────────────────────── */

/*
 * rl_key_for — build a namespaced rate-limit key: "<tag>:<ip>".
 *
 * Using an endpoint tag prevents one endpoint's traffic from consuming
 * another endpoint's budget.  Using the TCP peer address as the final
 * fallback (rather than the sentinel "unknown") keeps per-client isolation
 * even when the reverse proxy omits X-Forwarded-For.
 *
 * buf must be at least 128 bytes.
 */
static void rl_key_for(char *buf, size_t bufsz,
                       const char *tag, const Request *req)
{
    /*
     * Prefer the first address in X-Forwarded-For (set by nginx's
     * $proxy_add_x_forwarded_for directive).  Fall back to the direct
     * TCP peer address, which is always populated by the HTTP layer.
     */
    const char *ip = (req->fwd_for[0] != '\0') ? req->fwd_for
                                                : req->peer_addr;
    int n = snprintf(buf, bufsz, "%s:%s", tag, ip);
    /*
     * If the formatted key would overflow buf, use a safe sentinel that
     * still carries the endpoint tag so different endpoints keep separate
     * buckets even in this degenerate case.
     */
    if (n < 0 || (size_t)n >= bufsz) {
        snprintf(buf, bufsz, "%s:overflow", tag);
    }
}

/*
 * auth_rate_limited — return 1 if this request should be rejected.
 *
 * Each auth endpoint has its own bucket (tag differs) so that a burst of
 * failed logins cannot starve register/forgot/reset.
 */
static int auth_rate_limited(const char *tag, const Request *req, int limit)
{
    char key[128];
    rl_key_for(key, sizeof(key), tag, req);
    return !rl_check(key, limit);
}

/* ── Dispatcher ─────────────────────────────────────────────────────────── */

Response dispatch(Request req)
{
    /*
     * Auth endpoints — each gets its own rate-limit bucket.
     *
     * Limits (per IP, per endpoint, per window — window size is defined
     * inside rl_check):
     *   register : 10  (a human can create ≤10 accounts in a rate window)
     *   login    : 10  (failed-login click-storms stay in their own bucket)
     *   forgot   : 5   (lower — reset emails are expensive to send)
     *   reset    : 5   (same)
     */
    if (strcmp(req.path, "/auth/register") == 0) {
        if (auth_rate_limited("reg", &req, 10))
            return json_error(429, "rate_limited");
        return handle_register(req);
    }

    if (strcmp(req.path, "/auth/login") == 0) {
        if (auth_rate_limited("login", &req, 10))
            return json_error(429, "rate_limited");
        return handle_login(req);
    }

    if (strcmp(req.path, "/auth/forgot") == 0) {
        if (auth_rate_limited("forgot", &req, 5))
            return json_error(429, "rate_limited");
        return handle_forgot(req);
    }

    if (strcmp(req.path, "/auth/reset") == 0) {
        if (auth_rate_limited("reset", &req, 5))
            return json_error(429, "rate_limited");
        return handle_reset(req);
    }

    /* Authenticated endpoints */
    if (strcmp(req.path, "/goals") == 0) {
        return handle_goals(req);
    }

    return not_found();
}
