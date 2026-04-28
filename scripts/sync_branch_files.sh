#!/usr/bin/env bash
# scripts/sync_branch_files.sh
#
# Copy every source file changed on this branch into an existing kaggle2
# checkout that already has `./results/`, `./trocr/`, `./yolo/` artifacts
# from a prior training run, WITHOUT touching those artifacts.
#
# Usage:
#   # from inside this branch checkout:
#   bash scripts/sync_branch_files.sh /workspace/kaggle2
#
# Safety:
#   * Overwritten files are backed up to  <target>/.sync_backup_<timestamp>/
#   * Nothing under  results/ trocr/ yolo/ datasets/ .venv/ venv/  is touched.
#   * Dry-run:  bash scripts/sync_branch_files.sh /workspace/kaggle2 --dry-run

set -euo pipefail

TARGET="${1:-}"
DRY_RUN="${2:-}"

if [[ -z "${TARGET}" || "${TARGET}" == "-h" || "${TARGET}" == "--help" ]]; then
    echo "usage: bash scripts/sync_branch_files.sh <target_kaggle2_dir> [--dry-run]"
    exit 1
fi

if [[ ! -d "${TARGET}" ]]; then
    echo "error: target directory does not exist: ${TARGET}" >&2
    exit 1
fi

SRC="$(cd "$(dirname "$0")/.." && pwd)"
if [[ ! -f "${SRC}/main.py" ]]; then
    echo "error: this script must live at <branch>/scripts/sync_branch_files.sh" >&2
    exit 1
fi

# The exact files changed on this branch vs. origin/main. Keep this list in
# sync with `git diff --name-only origin/main...HEAD` whenever the branch adds
# new files — this is the single source of truth for what the sync copies.
FILES=(
    configs/default.json
    core/config.py
    core/types.py
    models/assigner_train.py
    models/attention_model.py
    models/donut_eval.py
    models/pipeline_assign.py
    models/pipeline_consensus.py
    models/pipeline_corrections.py
    models/pipeline_eval.py
    models/pipeline_miss_tracker.py
    models/pipeline_normalize.py
    models/rule_eval.py
    models/rule_regex.py
    scripts/eval_only.py
    scripts/sync_branch_files.sh
    stages/_common.py
    stages/eval.py
    tests/test_total_normalizer.py
)

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="${TARGET}/.sync_backup_${STAMP}"

echo "sync_branch_files.sh"
echo "  source : ${SRC}"
echo "  target : ${TARGET}"
echo "  backup : ${BACKUP}"
echo "  files  : ${#FILES[@]}"
if [[ "${DRY_RUN}" == "--dry-run" ]]; then
    echo "  mode   : DRY RUN (no writes)"
fi
echo

copied=0
new=0
skipped=0
for rel in "${FILES[@]}"; do
    src_path="${SRC}/${rel}"
    dst_path="${TARGET}/${rel}"

    if [[ ! -f "${src_path}" ]]; then
        echo "  SKIP   (missing in source) ${rel}"
        skipped=$((skipped + 1))
        continue
    fi

    if [[ -f "${dst_path}" ]] && cmp -s "${src_path}" "${dst_path}"; then
        echo "  ok     (identical)         ${rel}"
        continue
    fi

    if [[ "${DRY_RUN}" == "--dry-run" ]]; then
        if [[ -f "${dst_path}" ]]; then
            echo "  would update               ${rel}"
        else
            echo "  would create               ${rel}"
        fi
        continue
    fi

    mkdir -p "$(dirname "${dst_path}")"

    if [[ -f "${dst_path}" ]]; then
        mkdir -p "$(dirname "${BACKUP}/${rel}")"
        cp -p "${dst_path}" "${BACKUP}/${rel}"
        cp -p "${src_path}" "${dst_path}"
        echo "  updated (backup kept)      ${rel}"
        copied=$((copied + 1))
    else
        cp -p "${src_path}" "${dst_path}"
        echo "  created                    ${rel}"
        new=$((new + 1))
    fi
done

echo
echo "done. ${new} created, ${copied} updated, ${skipped} skipped."
if [[ "${DRY_RUN}" != "--dry-run" && -d "${BACKUP}" ]]; then
    echo "rollback with:  rsync -a '${BACKUP}/' '${TARGET}/'"
fi
