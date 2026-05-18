# CI/CD automation for the journal GPU experiments

This directory automates the **infrastructure** around the journal
experiments: rent a GPU, run the existing pipeline, collect raw
results, run severe tests, generate an audit checklist, and open a
human-gated pull request. It then tears the rented GPU down.

---

## **INTEGRITY GUARDRAIL (NON-NEGOTIABLE)**

> **This automation is INFRASTRUCTURE ONLY. It NEVER edits
> `journal-synthesis/main.tex`, NEVER writes or "fixes" a paper claim
> or number, NEVER decides a scientific outcome, and NEVER auto-merges
> anything.** Every generated pull request is labelled
> `needs-human-audit` and carries a prominent banner:
> **"DO NOT MERGE INTO PAPER WITHOUT HUMAN SIGN-OFF vs
> PREREGISTRATION.md"**. The scientific decision stays human. No
> fabrication anywhere.

How this is enforced in code:

- `controller.py` and `audit_checklist_gen.py` never construct a write
  handle to `main.tex`. `audit_checklist_gen.py` explicitly **refuses**
  if its output path is `main.tex`.
- The GitHub workflow and `local_runner.sh` both **abort** if
  `git diff` shows `main.tex` changed, and stage only
  `results/*.json`, `SEVERE.json`, `AUDIT_CHECKLIST.md`.
- The PR is opened as a **draft**, labelled `needs-human-audit`, with
  the audit checklist (which begins with the DO-NOT-MERGE banner) as
  the PR body. There is **no** auto-merge step anywhere.
- `audit_checklist_gen.py` *interprets* results into PASS / FAIL /
  UNDERPOWERED / ARTIFACT / HUMAN-REVIEW tags and a mandatory human
  sign-off checklist; it states explicitly that it does **not** decide.

---

## Phase map

| Phase | File(s) | Delivers |
|-------|---------|----------|
| 0 | `vast_api.py`, `teardown.sh`, `notify.sh`, run_parallel.sh wiring | lifecycle hygiene: stdlib vast.ai client, idempotent teardown + wall-clock cost-cap watchdog, opt-in self-destroy, status pings |
| 1 | `controller.py` | one command: search cheapest offer -> create -> wait for SSH -> bootstrap + run_parallel -> rsync results back -> ALWAYS destroy -> notify; hard `--max-cost-usd` / `--max-minutes`; `--dry-run` |
| 2 | `.github/workflows/journal-experiments.yml`, `audit_checklist_gen.py` | scheduled hands-off cycle: dispatch (schedule commented out), single-concurrency, controller -> checklist -> human-gated draft PR |
| 3 | `controller.py --loop` | self-repeating WITH BRAKES: mandatory global cost ceiling, `--max-iters`, `STOP` kill switch, stop on CONCLUSIVE verdict, stop on unchanged results hash |
| local | `local_runner.sh` | same pipeline minus rental: local CUDA GPU, opens the same human-gated PR via `gh`; cron / systemd-timer snippets included |

---

## Secrets handling

- **Never commit secrets.** The vast.ai API key is read **only** from
  `$VAST_API_KEY` (no CLI flag exists for it). The webhook is read only
  from `$NOTIFY_WEBHOOK`. The SSH key path comes from `$SSH_KEY_PATH` /
  `--ssh-key`.
- In GitHub Actions these are `${{ secrets.VAST_API_KEY }}`,
  `${{ secrets.NOTIFY_WEBHOOK }}`, `${{ secrets.SSH_PRIVATE_KEY }}`,
  and the built-in `${{ secrets.GITHUB_TOKEN }}`.
- Any local `.env` is **gitignored** (root `.gitignore` covers
  `.env`, `*.env`, `cicd/.env`). `cicd/state/` is gitignored.

---

## Cost safety

- `controller.py` enforces **hard** `--max-cost-usd` and
  `--max-minutes`: exceeding either force-destroys the instance and
  exits non-zero. The instance is destroyed on **every** exit path
  (finally / atexit / signal handlers), even on exception or Ctrl-C.
- `teardown.sh --cost-cap-watchdog SECONDS` is an independent
  background wall-clock killer.
- `run_parallel.sh` self-destroy is **OFF by default**: it arms only if
  `SELF_DESTROY=1` **and** an instance id **and** `VAST_API_KEY` are
  present (so a dev box is never destroyed by surprise).
  `MAX_RUN_MINUTES` is unset (= off) by default.
- The workflow's `schedule:` cron is **commented out by default** -
  unattended spend must be a deliberate choice. `concurrency:` ensures
  only one run at a time; the job has a `timeout-minutes` ceiling.
- **Phase 3 loop brakes:** `--global-cost-ceiling-usd` is **mandatory**
  (the loop refuses to start without it); `--max-iters`; a kill switch
  file `cicd/STOP` (create it to stop the loop); auto-stop when
  `SEVERE.json`'s S3 verdict is already `CONCLUSIVE`; auto-stop when
  the results hash is unchanged versus the last iteration. State (cost
  ledger, last hash) lives in the gitignored `cicd/state/`.

### Kill switch

```bash
touch journal-synthesis/vastai/cicd/STOP   # the loop stops next check
```

### Deliberately enabling the schedule

Editing the workflow's commented `schedule:` block is the **only** way
to enable unattended recurring spend. Do this only after a `--dry-run`
and one supervised run, and only with cost caps you have reviewed.

---

## First-use sequence (do this in order)

1. **Dry run, zero network / zero spend:**
   ```bash
   python3 cicd/vast_api.py search --dry-run
   python3 cicd/controller.py --dry-run
   ```
   Confirm the printed plan looks right. No instance is created.
2. **One supervised single run** (real spend, you watching):
   ```bash
   export VAST_API_KEY=...   # never commit
   export SSH_KEY_PATH=~/.ssh/id_vast
   python3 cicd/controller.py --max-cost-usd 2.00 --max-minutes 45
   python3 cicd/audit_checklist_gen.py
   ```
   Verify the instance was destroyed (vast.ai console) and inspect
   `results/AUDIT_CHECKLIST.md`.
3. **Optionally** enable the loop (with brakes) only after steps 1-2
   succeed:
   ```bash
   python3 cicd/controller.py --loop \
       --global-cost-ceiling-usd 20.00 --max-iters 5 \
       --max-cost-usd 2.00 --max-minutes 45
   ```
4. Only after a HUMAN has audited the PR against
   `journal-synthesis/PREREGISTRATION.md` may any number be
   transcribed into the paper, by a human.

---

## Honest limitation

**None of this was live-tested here.** There are no vast.ai / GPU /
GitHub Actions credentials in this environment, so only static checks
were possible (`py_compile`, `bash -n`, YAML parse, `--dry-run` with
zero network). The **first real use must be a `--dry-run`, then a
single supervised run**, before any unattended loop or scheduled cron.
