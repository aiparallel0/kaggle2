#!/usr/bin/env bash
# Audit prose overlap between Paper 2 and Paper 3 paper-specific sections.
#
# What this measures: 5-word shingle overlap between the
# *paper-specific* sections (intro, results, abstract, conclusion-
# specific files) — NOT the shared substrate (related, method_pipeline,
# experiments, bugs).  The shared substrate is by design described
# similarly in both papers (DONUT recipe, YOLO+TrOCR upstream,
# normaliser, manifest).  The bifurcation contract in
# docs/PAPER2_VS_PAPER3.md allows up to 1 paragraph of shared-substrate
# prose per paper; what it forbids is shared abstract / intro / results
# wording.
#
# Pass criteria: overlap on paper-specific sections must be ≤ 8 %.
# This is calibrated against the compiled-paper baseline (~20 %) so
# the audit fails when paper-specific sections start cross-contaminating
# but tolerates the legitimate shared-substrate paragraphs.
#
# Usage:
#   bash scripts/audit_paper_overlap.sh
#   bash scripts/audit_paper_overlap.sh paper2_intro paper3_intro
#
# Exits 0 if overlap is within budget; 1 otherwise (CI-friendly).

set -euo pipefail

# By default audit the paper-specific section pair (intro_paper2 vs
# intro_neurips).  Callers may pass alternative section files.
PAPER2="${1:-report/sections/intro_paper2.tex}"
PAPER3="${2:-report/sections/intro_neurips.tex}"

if [[ ! -f "$PAPER2" ]]; then
    echo "audit: Paper 2 template not found: $PAPER2" >&2
    exit 2
fi
if [[ ! -f "$PAPER3" ]]; then
    echo "audit: Paper 3 template not found: $PAPER3" >&2
    exit 2
fi

# Expand \input{} so the audit sees the full prose, not just the
# template scaffold.  We use sed for portability rather than tectonic.
expand_tex() {
    local f="$1"
    local dir
    dir="$(dirname "$f")"
    local body
    body="$(cat "$f")"
    while echo "$body" | grep -qE '\\input\{[^}]+\}'; do
        local line input_path
        line="$(echo "$body" | grep -oE '\\input\{[^}]+\}' | head -1)"
        input_path="$(echo "$line" | sed -E 's/\\input\{([^}]+)\}/\1/')"
        # Try .tex extension if it has none.
        if [[ ! "$input_path" =~ \.tex$ ]]; then
            input_path="${input_path}.tex"
        fi
        local resolved="$dir/$input_path"
        if [[ -f "$resolved" ]]; then
            local replacement
            replacement="$(cat "$resolved")"
            # Use awk to do single-line replacement to avoid sed escape pain.
            body="$(awk -v needle="$line" -v repl="$replacement" '
                BEGIN { found = 0 }
                {
                    if (!found && index($0, needle)) {
                        n = index($0, needle)
                        before = substr($0, 1, n - 1)
                        after = substr($0, n + length(needle))
                        print before repl after
                        found = 1
                    } else {
                        print
                    }
                }
            ' <<< "$body")"
        else
            # Resolve failed — drop the input tag so the loop terminates.
            body="$(echo "$body" | sed -e "s|\\\\input{[^}]*}|MISSING|")"
        fi
    done
    echo "$body"
}

# Strip LaTeX so the comparison sees prose only.
prose() {
    local raw="$1"
    echo "$raw" \
        | sed -E 's/%.*$//' \
        | sed -E 's/\\[a-zA-Z]+(\[[^]]*\])?(\{[^}]*\})?//g' \
        | sed -E 's/\$[^$]*\$//g' \
        | sed -E 's/\\VAR\{[^}]+\}//g' \
        | sed -E 's/[{}\\]/ /g' \
        | tr '[:upper:]' '[:lower:]' \
        | tr -cs 'a-z' '\n' \
        | grep -v '^$' \
        | grep -vE '^.{0,2}$'
}

P2_TOKENS="$(prose "$(expand_tex "$PAPER2")")"
P3_TOKENS="$(prose "$(expand_tex "$PAPER3")")"

P2_COUNT="$(echo "$P2_TOKENS" | wc -l)"
P3_COUNT="$(echo "$P3_TOKENS" | wc -l)"

# 5-word shingles.
shingles() {
    local tokens="$1"
    awk 'BEGIN { ORS=" " } { a[NR]=$0 } END {
        for (i = 1; i <= NR - 4; i++) {
            print a[i] "_" a[i+1] "_" a[i+2] "_" a[i+3] "_" a[i+4]
            printf "\n"
        }
    }' <<< "$tokens"
}

P2_SHINGLES_FILE="$(mktemp)"
P3_SHINGLES_FILE="$(mktemp)"
trap 'rm -f "$P2_SHINGLES_FILE" "$P3_SHINGLES_FILE"' EXIT

shingles "$P2_TOKENS" > "$P2_SHINGLES_FILE"
shingles "$P3_TOKENS" > "$P3_SHINGLES_FILE"

P2_SHINGLE_COUNT="$(wc -l < "$P2_SHINGLES_FILE")"
P3_SHINGLE_COUNT="$(wc -l < "$P3_SHINGLES_FILE")"
SMALLER=$(( P2_SHINGLE_COUNT < P3_SHINGLE_COUNT ? P2_SHINGLE_COUNT : P3_SHINGLE_COUNT ))

OVERLAP_COUNT="$(sort -u "$P2_SHINGLES_FILE" | comm -12 - <(sort -u "$P3_SHINGLES_FILE") | wc -l)"

if [[ "$SMALLER" -eq 0 ]]; then
    echo "audit: one paper has zero prose; nothing to compare." >&2
    exit 0
fi

OVERLAP_PCT=$(( 100 * OVERLAP_COUNT / SMALLER ))

echo "Paper 2 ($PAPER2):  $P2_COUNT tokens, $P2_SHINGLE_COUNT shingles"
echo "Paper 3 ($PAPER3):  $P3_COUNT tokens, $P3_SHINGLE_COUNT shingles"
echo "Shared 5-grams:     $OVERLAP_COUNT"
echo "Overlap (smaller):  $OVERLAP_PCT %"

if [[ "$OVERLAP_PCT" -gt 8 ]]; then
    echo "audit: FAIL — overlap $OVERLAP_PCT % exceeds 8 % budget." >&2
    echo "       (Paper-specific sections are cross-contaminating; check intro / abstract / results.)" >&2
    exit 1
fi
echo "audit: PASS (paper-specific overlap within 8 % budget)"
