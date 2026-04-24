/**
 * ui/app.js — teb2 single-page application.
 *
 * Fix (Bug 3): after a 401 response on the auto-sign-in that immediately
 * follows a successful registration, the original code fell through to the
 * generic teb.errmsg() handler which displayed "Please log in" — confusing
 * because the user literally just registered.
 *
 * This version:
 *   - Distinguishes "post-register 401" from "normal session-expired 401".
 *   - Shows an actionable banner: "Account created, but sign-in failed —
 *     please use the form below or reset your password."
 *   - Surfaces a password-reset link directly in the banner so the user has
 *     a clear recovery path without needing to understand why the login
 *     failed.
 */

/* global teb */

(function () {
    "use strict";

    var T = "";          /* JWT / session token, empty = unauthenticated */
    var _justRegistered = false;  /* set true after doRegister succeeds */

    /* ── Utilities ──────────────────────────────────────────────────────── */

    /**
     * errmsg — map an API response object to a human-readable string.
     *
     * Preserves the original status-code mapping from the legacy version.
     */
    teb.errmsg = function (r) {
        var s = r && r._status;
        if (s === 429) return "Rate limited — try again shortly.";
        if (s === 401) return "Please log in.";
        if (s === 403) return "You don't have permission to do that.";
        if (s === 404) return "Not found.";
        return (r && r.message) ? r.message : "Something went wrong.";
    };

    /* ── Auth ────────────────────────────────────────────────────────────── */

    /**
     * signIn — attempt to authenticate with the server.
     *
     * @param {string}   email
     * @param {string}   password
     * @param {Function} onOk   called with the token string on success
     * @param {Function} onFail called with the response object on failure
     */
    function signIn(email, password, onOk, onFail) {
        teb.post("/auth/login", { email: email, password: password },
            function (r) {
                T = r.token || "";
                if (onOk) onOk(T);
            },
            function (r) {
                T = "";
                if (onFail) onFail(r);
            }
        );
    }

    /**
     * doRegister — submit the registration form.
     *
     * On success, automatically attempts to sign in.  If the auto-sign-in
     * returns a 401 (the server accepted the registration but will not issue
     * a token — e.g. because of a hash-overflow bug in old deployments),
     * we show a specific recovery banner rather than the generic "Please log
     * in" message.
     */
    window.doRegister = function () {
        var email    = document.getElementById("reg-email").value.trim();
        var password = document.getElementById("reg-password").value;

        teb.post("/auth/register", { email: email, password: password },
            function (/* r */) {
                teb.info("Account created — signing in\u2026");
                _justRegistered = true;

                signIn(email, password,
                    function (/* token */) {
                        _justRegistered = false;
                        showView("goals");
                    },
                    function (r2) {
                        _justRegistered = false;

                        /*
                         * FIX (Bug 3): distinguish "post-register 401" from
                         * a normal session-expired 401.
                         *
                         * Original code called teb.errmsg(r2) here, which
                         * displayed "Please log in" — misleading when the
                         * user just finished registering.
                         *
                         * We now show a dedicated banner that:
                         *   1. Confirms the account WAS created.
                         *   2. Tells the user sign-in failed (server error,
                         *      not user error).
                         *   3. Provides a password-reset link as a recovery
                         *      path without requiring any technical knowledge.
                         */
                        if (r2 && r2._status === 401) {
                            showPostRegisterFailBanner(email);
                        } else {
                            teb.err(teb.errmsg(r2));
                        }
                    }
                );
            },
            function (r) {
                teb.err(teb.errmsg(r));
            }
        );
    };

    /**
     * showPostRegisterFailBanner — display an actionable error when auto-
     * sign-in fails immediately after registration.
     *
     * The user's account exists; they just cannot log in right now.  The
     * banner provides a reset link so they are never stuck.
     *
     * @param {string} email — pre-populated into the reset link
     */
    function showPostRegisterFailBanner(email) {
        var encodedEmail = encodeURIComponent(email || "");
        var resetHref    = "/auth/forgot?email=" + encodedEmail;

        /* Reuse whatever notification container the app already has.
         * Fall back to a simple alert if the container is absent. */
        var container = document.getElementById("auth-message");
        if (!container) {
            window.alert(
                "Your account was created, but automatic sign-in failed.\n" +
                "Please use the login form, or reset your password."
            );
            return;
        }

        container.className = "auth-message error";
        container.innerHTML =
            "Your account was created, but the server could not sign you in " +
            "automatically. " +
            "Please <strong>try logging in below</strong>, or " +
            "<a href=\"" + resetHref + "\">reset your password</a> " +
            "if that doesn\u2019t work.";
        container.style.display = "block";
    }

    /* ── Views ───────────────────────────────────────────────────────────── */

    /**
     * showView — render the named view if the user is authenticated.
     *
     * Unchanged from the original except that the "not authenticated" path
     * now uses the specific errmsg for 401 rather than a raw string literal,
     * keeping the wording consistent.
     */
    window.showView = function (name) {
        if (!T) {
            teb.err(teb.errmsg({ _status: 401 }));
            return;
        }
        var fn = window["load_" + name];
        if (fn) fn();
    };

    /**
     * doLogin — submit the login form directly (not via registration flow).
     */
    window.doLogin = function () {
        var email    = document.getElementById("login-email").value.trim();
        var password = document.getElementById("login-password").value;

        signIn(email, password,
            function (/* token */) {
                showView("goals");
            },
            function (r) {
                teb.err(teb.errmsg(r));
            }
        );
    };

}());
