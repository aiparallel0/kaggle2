/*
 * core/types.h — shared type definitions for the teb2 auth backend.
 *
 * Fix (Bug 2): password_hash and HashResult.hash are widened from 128 → 256
 * bytes.  A SHA-512 crypt(3) output is at most 106 bytes (15-byte prefix +
 * 24-char base64 salt + "$" + 86-char base64 digest + NUL = 127), but
 * larger rounds values or longer salts could push the encoded output past
 * 127 bytes.  Using 256 bytes gives a 2× safety margin and costs < 0.5 KB
 * per User record on the stack.
 */

#ifndef TEB2_TYPES_H
#define TEB2_TYPES_H

#include <stdint.h>

/* ── Roles ──────────────────────────────────────────────────────────────── */

typedef enum {
    ROLE_USER  = 0,
    ROLE_ADMIN = 1,
} UserRole;

/* ── User record ─────────────────────────────────────────────────────────
 *
 * CHANGE vs. original: password_hash widened 128 → 256.
 */
typedef struct {
    int64_t  id;
    char     user_id[64];
    char     email[128];
    char     password_hash[256];  /* was 128 — see file header */
    UserRole role;
} User;

/* ── Errors ──────────────────────────────────────────────────────────────── */

typedef enum {
    ERR_OK         = 0,
    ERR_NOT_FOUND  = 1,
    ERR_DUPLICATE  = 2,
    ERR_CRYPTO     = 3,   /* hash overflow or crypt(3) failure */
    ERR_INTERNAL   = 4,
} Err;

/* ── Hash config / result ────────────────────────────────────────────────
 *
 * CHANGE vs. original: HashResult.hash widened 128 → 256, matching User.
 */
typedef struct {
    unsigned int rounds;
    char         salt[32];
} HashConfig;

typedef struct {
    Err  err;
    char hash[256];  /* was 128 — must match User.password_hash */
    int  match;
} HashResult;

/* ── HTTP request/response primitives ───────────────────────────────────── */

typedef struct {
    const char *method;
    const char *path;
    const char *body;
    size_t      body_len;
    char        fwd_for[64];   /* first value from X-Forwarded-For, or "" */
    char        peer_addr[64]; /* TCP peer IP (always available)          */
} Request;

typedef struct {
    int         status;
    const char *body;
} Response;

#endif /* TEB2_TYPES_H */
