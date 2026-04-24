/*
 * auth/hash.c — password hashing and verification.
 *
 * Fix (Bug 2): hash_password now returns ERR_CRYPTO when the crypt(3) output
 * would overflow HashResult.hash, instead of silently truncating it via
 * snprintf.  Silent truncation caused verify_password to always fail for
 * affected users: the stored hash was truncated, but crypt(password, truncated)
 * reproduced the *full-length* hash → strcmp mismatch → 401 on every login.
 */

#include <string.h>
#include <stdio.h>
#include <unistd.h>      /* crypt(3) on Linux with _GNU_SOURCE / -lcrypt */

#define _GNU_SOURCE
#include <crypt.h>

#include "../core/types.h"

/* ── Internal helpers ─────────────────────────────────────────────────── */

/* Default hash configuration: SHA-512 with 5 000 rounds. */
HashConfig default_hash_config(void)
{
    HashConfig cfg;
    cfg.rounds = 5000;
    /* 16 bytes of base-64 encoded salt, generated at call site */
    cfg.salt[0] = '\0';
    return cfg;
}

/*
 * make_salt — write the full salt prefix into buf.
 *
 * Format: "$6$rounds=<N>$<raw_salt>$"
 * Maximum length for the formatted salt prefix:
 *   "$6$rounds=" (10) + 10-digit rounds + "$" + 31-char salt + "$" + NUL
 *   = 10 + 10 + 1 + 31 + 1 + 1 = 54 bytes
 * buf must be at least 64 bytes.
 */
static int make_salt(char *buf, size_t bufsz, const HashConfig *cfg)
{
    int n = snprintf(buf, bufsz, "$6$rounds=%u$%s$",
                     cfg->rounds ? cfg->rounds : 5000u,
                     cfg->salt);
    if (n < 0 || (size_t)n >= bufsz) return -1;
    return 0;
}

/* ── Public API ───────────────────────────────────────────────────────── */

/*
 * hash_password — derive a salted SHA-512 hash for the given password.
 *
 * Returns HashResult with:
 *   .err   = ERR_OK on success, ERR_CRYPTO on overflow or crypt failure.
 *   .hash  = the printable hash string (fits in HashResult.hash[256]).
 *   .match = 0 (not applicable during hash creation).
 *
 * CHANGE vs. original:
 *   Added explicit length check before copying into r.hash.  If crypt(3)
 *   returns a string that is >= sizeof(r.hash) bytes the function now sets
 *   r.err = ERR_CRYPTO and leaves r.hash empty, instead of silently
 *   truncating by one or more bytes (which caused verify_password to always
 *   return match=0 for the affected user).
 */
HashResult hash_password(const char *password, HashConfig cfg)
{
    HashResult r;
    memset(&r, 0, sizeof(r));

    char salt_buf[64];
    if (make_salt(salt_buf, sizeof(salt_buf), &cfg) != 0) {
        r.err = ERR_CRYPTO;
        return r;
    }

    const char *h = crypt(password, salt_buf);
    if (!h) {
        r.err = ERR_CRYPTO;
        return r;
    }

    size_t hlen = strlen(h);

    /*
     * SAFETY CHECK (the fix): if the crypt output does not fit — including
     * the NUL terminator — inside HashResult.hash, refuse to store it.
     * Callers must treat ERR_CRYPTO as a configuration error (rounds / salt
     * too large) rather than retrying with the same parameters.
     */
    if (hlen >= sizeof(r.hash)) {
        r.err = ERR_CRYPTO;
        return r;
    }

    memcpy(r.hash, h, hlen + 1); /* +1 for NUL */
    r.err   = ERR_OK;
    r.match = 0;
    return r;
}

/*
 * verify_password — compare a plaintext password against a stored hash.
 *
 * Returns HashResult with:
 *   .err   = ERR_OK if the comparison completed (even if passwords differ).
 *            ERR_CRYPTO if crypt fails or the result would overflow.
 *   .match = 1 if passwords match, 0 otherwise.
 *
 * Note: the stored hash string itself serves as the salt for crypt(3);
 * crypt extracts the algorithm, rounds, and salt automatically from the
 * "$6$rounds=N$salt$…" prefix.
 */
HashResult verify_password(const char *password, HashResult stored)
{
    HashResult r;
    memset(&r, 0, sizeof(r));

    if (stored.err != ERR_OK || stored.hash[0] == '\0') {
        r.err = ERR_CRYPTO;
        return r;
    }

    const char *h = crypt(password, stored.hash);
    if (!h) {
        r.err = ERR_CRYPTO;
        return r;
    }

    /*
     * crypt() produces the full-length hash even when stored.hash was
     * (wrongly) truncated in a previous version.  The explicit length check
     * in hash_password prevents new truncated hashes from being stored, but
     * this guard also protects verify_password from a silent mismatch in
     * case a legacy truncated hash is still in the database.
     */
    size_t hlen = strlen(h);
    if (hlen >= sizeof(r.hash)) {
        r.err = ERR_CRYPTO;
        return r;
    }

    r.err   = ERR_OK;
    r.match = (strcmp(h, stored.hash) == 0) ? 1 : 0;
    return r;
}
