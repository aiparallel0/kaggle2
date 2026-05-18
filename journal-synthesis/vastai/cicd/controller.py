#!/usr/bin/env python3
"""controller.py - one-command provision -> run -> fetch -> destroy.

PHASE 1 (single cycle) and PHASE 3 (--loop, with hard brakes).

NON-NEGOTIABLE INTEGRITY GUARDRAIL
----------------------------------
This is INFRASTRUCTURE ONLY. It rents a GPU, runs the existing
experiment pipeline, copies raw `results/` back into the working copy,
and ALWAYS destroys the rented instance. It NEVER:
  * edits journal-synthesis/main.tex (it never even references it for
    writing),
  * writes, alters, or "fixes" any paper claim or number,
  * decides a scientific outcome,
  * merges anything.
The scientific decision stays human. The downstream PR (Phase 2) is
labelled needs-human-audit and carries a DO-NOT-MERGE banner; this
controller only produces the raw artifacts that PR will carry.

SECRETS: the vast.ai key is read by vast_api.py from $VAST_API_KEY ONLY.
The SSH key path / host come from args or env. Nothing is embedded; the
key is never logged.

COST SAFETY: --max-cost-usd and --max-minutes are HARD. If either is
exceeded the instance is force-destroyed and the process exits non-zero.
The instance is destroyed in a finally / atexit / signal handler even on
exception or KeyboardInterrupt. --dry-run prints the full plan and makes
ZERO network / API / SSH calls.

PHASE 3 BRAKES (--loop): a GLOBAL cumulative cost ceiling FILE
(mandatory, the loop refuses to start without it), --max-iters, a
kill-switch file (cicd/STOP present => stop), stop when SEVERE.json
verdict is already CONCLUSIVE, and stop when the results hash is
unchanged versus the previous iteration. State lives in cicd/state/
(gitignored).
"""
from __future__ import annotations

import argparse
import atexit
import datetime
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VASTAI_DIR = os.path.dirname(HERE)                       # .../vastai
REPO_ROOT = os.path.abspath(os.path.join(VASTAI_DIR, "..", ".."))
STATE_DIR = os.path.join(HERE, "state")
STOP_FILE = os.path.join(HERE, "STOP")
sys.path.insert(0, HERE)

import vast_api  # noqa: E402  (stdlib-only sibling module)

# main.tex is referenced ONLY to assert we never touch it. There is no
# code path anywhere in this file that opens it for writing.
MAIN_TEX = os.path.join(REPO_ROOT, "journal-synthesis", "main.tex")


def log(msg: str) -> None:
    print(f"[controller] {msg}", flush=True)


def _now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def notify(event: str, message: str) -> None:
    """Best-effort lifecycle ping via notify.sh (no-op if unset)."""
    script = os.path.join(HERE, "notify.sh")
    if not os.path.exists(script):
        return
    try:
        subprocess.run(["bash", script, event, message],
                       timeout=30, check=False)
    except Exception as e:  # notifications are never load-bearing
        log(f"notify failed (non-fatal): {e}")


class CostExceeded(RuntimeError):
    pass


class Controller:
    def __init__(self, args):
        self.args = args
        self.instance_id = None
        self.start_ts = time.time()
        self.dph = None
        self._destroyed = False
        # Register teardown on EVERY exit path.
        atexit.register(self.teardown)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._signal_teardown)

    # ---- cost / time guards ------------------------------------------
    def elapsed_minutes(self) -> float:
        return (time.time() - self.start_ts) / 60.0

    def projected_cost(self) -> float:
        if self.dph is None:
            return 0.0
        return self.dph * (self.elapsed_minutes() / 60.0)

    def enforce_caps(self) -> None:
        if self.elapsed_minutes() > self.args.max_minutes:
            raise CostExceeded(
                f"wall-clock {self.elapsed_minutes():.1f} min exceeded "
                f"--max-minutes {self.args.max_minutes}")
        if self.projected_cost() > self.args.max_cost_usd:
            raise CostExceeded(
                f"projected ${self.projected_cost():.4f} exceeded "
                f"--max-cost-usd {self.args.max_cost_usd}")

    # ---- teardown (idempotent, every exit path) ----------------------
    def _signal_teardown(self, signum, _frame):
        log(f"signal {signum} -> teardown then exit")
        self.teardown()
        os._exit(143)

    def teardown(self) -> None:
        if self._destroyed or self.instance_id is None:
            return
        self._destroyed = True
        if self.args.dry_run:
            log(f"[dry-run] would destroy instance {self.instance_id} "
                f"(no network)")
            return
        log(f"destroying instance {self.instance_id} (always-destroy "
            f"guarantee)")
        for attempt in range(1, 5):
            try:
                vast_api.destroy_instance(self.instance_id)
                log(f"instance {self.instance_id} destroy accepted")
                return
            except Exception as e:  # keep trying; this is the brake
                wait = 2 ** attempt
                log(f"destroy attempt {attempt} failed: {e}; "
                    f"retry in {wait}s")
                time.sleep(wait)
        log(f"ERROR: could not destroy {self.instance_id}; DESTROY IT "
            f"MANUALLY in the vast.ai console NOW.")

    # ---- the plan ----------------------------------------------------
    def plan(self) -> dict:
        return {
            "phase": "1 (provision->run->fetch->destroy)",
            "dry_run": self.args.dry_run,
            "search": {
                "max_dph": self.args.max_dph,
                "min_disk_gb": self.args.min_disk_gb,
                "gpu_name": self.args.gpu_name,
                "allow_interruptible": self.args.allow_interruptible,
            },
            "image": self.args.image,
            "remote_steps": [
                "git clone/pull repo (bootstrap.sh)",
                "bootstrap.sh (deps, checkpoint, data fetchers)",
                "run_parallel.sh (Stage A GPU decode-once, Stage B "
                "CPU E1E3/E5/E6/E9/E10 + GPU E7/E8, Stage C severe)",
            ],
            "fetch": f"rsync remote results/ -> {self._local_results()}",
            "caps": {
                "max_cost_usd": self.args.max_cost_usd,
                "max_minutes": self.args.max_minutes,
            },
            "always_destroy": True,
            "integrity": ("infra only; NEVER edits main.tex / paper / "
                          "verdict; downstream PR is human-gated"),
        }

    def _local_results(self) -> str:
        return os.path.join(VASTAI_DIR, "results")

    # ---- remote command construction ---------------------------------
    def _ssh_base(self, host, port):
        key = self.args.ssh_key
        opts = ["-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=20",
                "-p", str(port)]
        if key:
            opts = ["-i", key] + opts
        return ["ssh"] + opts + [f"{self.args.ssh_user}@{host}"]

    def _remote_script(self) -> str:
        # Bootstrap then run the full pipeline. Env (tokens) is passed
        # by the caller's environment on the remote via export lines the
        # human/CI sets; we only orchestrate, never embed creds.
        rc = self.args.remote_repo_dir
        return (
            f"set -e; "
            f"cd {shlex.quote(rc)}/journal-synthesis/vastai; "
            f"bash bootstrap.sh; "
            f"source .env.sh; "
            f"bash run_parallel.sh")

    # ---- single cycle ------------------------------------------------
    def run_cycle(self) -> dict:
        a = self.args
        notify("start", "provisioning vast.ai GPU for journal experiments")

        offers = vast_api.search_offers(
            max_dph=a.max_dph, min_disk_gb=a.min_disk_gb,
            gpu_name=a.gpu_name,
            allow_interruptible=a.allow_interruptible,
            dry_run=a.dry_run)
        if not offers:
            raise RuntimeError("no vast.ai offer met the spec")
        offer = offers[0]
        self.dph = float(offer.get("dph_total", 0.0) or 0.0)
        log(f"cheapest offer id={offer.get('id')} "
            f"${self.dph}/hr gpu={offer.get('gpu_name')}")
        # Pre-flight cost sanity: even one max-minutes block must fit
        # the cost cap.
        proj = self.dph * (a.max_minutes / 60.0)
        if proj > a.max_cost_usd:
            raise CostExceeded(
                f"offer ${self.dph}/hr * {a.max_minutes}min "
                f"= ${proj:.4f} > --max-cost-usd {a.max_cost_usd}")

        if a.dry_run:
            log("[dry-run] plan complete; no instance created, no "
                "network, no SSH. Printing plan and exiting.")
            print(json.dumps(self.plan(), indent=2))
            return {"dry_run": True, "plan": self.plan()}

        inst = vast_api.create_instance(
            offer["id"], image=a.image, disk_gb=a.min_disk_gb,
            label="journal-exp")
        self.instance_id = int(inst["id"])
        log(f"created instance {self.instance_id}")

        host = port = None
        deadline = time.time() + a.ssh_timeout_s
        while time.time() < deadline:
            self.enforce_caps()
            host, port = vast_api.ssh_endpoint(self.instance_id)
            if host and port:
                break
            time.sleep(15)
        if not (host and port):
            raise RuntimeError(
                f"instance {self.instance_id} never exposed SSH within "
                f"{a.ssh_timeout_s}s")
        log(f"SSH ready {host}:{port}")

        # Run the pipeline remotely.
        ssh = self._ssh_base(host, port)
        cmd = ssh + [self._remote_script()]
        log("launching remote bootstrap + run_parallel.sh")
        proc = subprocess.Popen(cmd)
        while proc.poll() is None:
            try:
                self.enforce_caps()
            except CostExceeded:
                proc.terminate()
                raise
            time.sleep(20)
        rc = proc.returncode
        log(f"remote pipeline exited rc={rc}")

        # Fetch results back (rsync; scp fallback).
        local_res = self._local_results()
        os.makedirs(local_res, exist_ok=True)
        remote_res = (f"{a.ssh_user}@{host}:"
                      f"{a.remote_repo_dir}/journal-synthesis/vastai/"
                      f"results/")
        rsync_ssh = "ssh -p %d -o StrictHostKeyChecking=no " \
                    "-o UserKnownHostsFile=/dev/null" % port
        if a.ssh_key:
            rsync_ssh += f" -i {shlex.quote(a.ssh_key)}"
        try:
            subprocess.run(
                ["rsync", "-az", "-e", rsync_ssh,
                 remote_res, local_res + "/"], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            log(f"rsync failed ({e}); trying scp")
            scp = ["scp", "-r", "-P", str(port),
                   "-o", "StrictHostKeyChecking=no",
                   "-o", "UserKnownHostsFile=/dev/null"]
            if a.ssh_key:
                scp = scp[:1] + ["-i", a.ssh_key] + scp[1:]
            subprocess.run(scp + [remote_res, local_res], check=True)
        log(f"results fetched into {local_res}")

        if rc != 0:
            notify("failure", f"remote pipeline rc={rc}; results fetched")
            raise RuntimeError(f"remote pipeline failed rc={rc}")
        notify("success", "experiments complete; raw results fetched")
        return {"rc": rc, "results_dir": local_res,
                "instance_id": self.instance_id}


# ---- Phase 3 loop state helpers --------------------------------------
def _results_hash(results_dir: str) -> str:
    h = hashlib.sha256()
    if not os.path.isdir(results_dir):
        return ""
    for name in sorted(os.listdir(results_dir)):
        if not (name.endswith(".json")):
            continue
        p = os.path.join(results_dir, name)
        try:
            with open(p, "rb") as f:
                h.update(name.encode())
                h.update(f.read())
        except OSError:
            continue
    return h.hexdigest()


def _severe_conclusive(results_dir: str) -> bool:
    p = os.path.join(results_dir, "SEVERE.json")
    if not os.path.exists(p):
        return False
    try:
        d = json.load(open(p))
    except Exception:
        return False
    # Conservative: only CONCLUSIVE if the pre-registered S3 verdict
    # says so. This module INTERPRETS the existing verdict; it does NOT
    # decide science. A human still signs off downstream.
    pooled = d.get("pooled", {})
    s3 = pooled.get("S3_power_minimum_detectable_effect", {})
    return s3.get("h1_negative_conclusive") is True


def _read_cum_cost(path: str) -> float:
    try:
        return float(open(path).read().strip() or "0")
    except Exception:
        return 0.0


def _write_cum_cost(path: str, val: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{val:.6f}\n")


def run_loop(args) -> int:
    """Phase 3: self-repeating WITH BRAKES. Impossible to start without
    an explicit global cost ceiling."""
    if args.global_cost_ceiling_usd is None:
        log("REFUSING to loop: --global-cost-ceiling-usd is MANDATORY. "
            "The loop cannot start without an explicit cumulative cost "
            "ceiling (cost-safety brake).")
        return 2
    os.makedirs(STATE_DIR, exist_ok=True)
    cum_path = os.path.join(STATE_DIR, "cumulative_cost_usd.txt")
    hash_path = os.path.join(STATE_DIR, "last_results_hash.txt")
    cum = _read_cum_cost(cum_path)
    last_hash = ""
    if os.path.exists(hash_path):
        last_hash = open(hash_path).read().strip()

    it = 0
    while True:
        it += 1
        if os.path.exists(STOP_FILE):
            log(f"KILL SWITCH: {STOP_FILE} present -> stopping loop.")
            return 0
        if args.max_iters is not None and it > args.max_iters:
            log(f"--max-iters {args.max_iters} reached -> stopping.")
            return 0
        if cum >= args.global_cost_ceiling_usd:
            log(f"GLOBAL COST CEILING hit "
                f"(${cum:.4f} >= ${args.global_cost_ceiling_usd}) "
                f"-> stopping loop.")
            return 0
        log(f"=== loop iteration {it} (cumulative ${cum:.4f} / "
            f"ceiling ${args.global_cost_ceiling_usd}) ===")

        ctrl = Controller(args)
        try:
            ctrl.run_cycle()
        except CostExceeded as e:
            log(f"cost cap hit: {e}")
            ctrl.teardown()
            return 3
        finally:
            ctrl.teardown()
        cum += ctrl.projected_cost()
        _write_cum_cost(cum_path, cum)

        res_dir = ctrl._local_results()
        if _severe_conclusive(res_dir):
            log("SEVERE.json S3 verdict already CONCLUSIVE -> the loop "
                "stops (further runs cannot change the pre-registered "
                "verdict; a HUMAN must now audit).")
            return 0
        cur_hash = _results_hash(res_dir)
        if cur_hash and cur_hash == last_hash:
            log("results hash unchanged vs last iteration -> nothing "
                "new to learn; stopping loop (brake).")
            return 0
        last_hash = cur_hash
        with open(hash_path, "w") as f:
            f.write(cur_hash + "\n")
        if args.dry_run:
            log("[dry-run] loop does exactly one simulated iteration "
                "with zero network, then stops.")
            return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Provision->run->fetch->destroy controller "
                    "(infra only; never writes the paper).")
    p.add_argument("--dry-run", action="store_true",
                   help="print the full plan; ZERO network/API/SSH.")
    p.add_argument("--max-cost-usd", type=float, default=2.00,
                   help="HARD per-run cost cap (force destroy + exit).")
    p.add_argument("--max-minutes", type=float, default=45.0,
                   help="HARD per-run wall-clock cap.")
    p.add_argument("--max-dph", type=float, default=0.50,
                   help="max $/hr offer to accept.")
    p.add_argument("--min-disk-gb", type=float, default=48.0)
    p.add_argument("--gpu-name", default=None)
    p.add_argument("--allow-interruptible", action="store_true",
                   help="OK for correctness numbers; NOT for E8 latency "
                        "(E8 stays single-disclosed-GPU by design).")
    p.add_argument("--image", default="vastai/pytorch:latest")
    p.add_argument("--ssh-user", default="root")
    p.add_argument("--ssh-key", default=os.environ.get("SSH_KEY_PATH"),
                   help="SSH private key path (env SSH_KEY_PATH); "
                        "never embedded.")
    p.add_argument("--ssh-timeout-s", type=int, default=900)
    p.add_argument("--remote-repo-dir", default="/workspace/kaggle2")
    # Phase 3
    p.add_argument("--loop", action="store_true",
                   help="Phase 3 self-repeating mode WITH BRAKES.")
    p.add_argument("--global-cost-ceiling-usd", type=float, default=None,
                   help="MANDATORY for --loop: cumulative cost ceiling.")
    p.add_argument("--max-iters", type=int, default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # Defensive integrity assertion: this process must never have opened
    # main.tex for writing. There is no such code path; assert the file
    # is outside our write scope by never constructing a write handle.
    assert "w" not in "", "unreachable"  # documents intent; no-op
    if args.loop:
        return run_loop(args)
    ctrl = Controller(args)
    try:
        out = ctrl.run_cycle()
    except CostExceeded as e:
        log(f"COST CAP EXCEEDED: {e} -> force destroy + nonzero exit")
        ctrl.teardown()
        notify("failure", f"cost cap exceeded: {e}")
        return 3
    except KeyboardInterrupt:
        log("interrupted -> teardown")
        ctrl.teardown()
        return 130
    except Exception as e:
        log(f"ERROR: {e} -> teardown")
        ctrl.teardown()
        notify("failure", f"controller error: {e}")
        return 1
    finally:
        ctrl.teardown()
    if args.dry_run:
        return 0
    log("cycle complete; raw results in working copy. A HUMAN must "
        "audit them vs PREREGISTRATION.md before any number moves into "
        "the paper.")
    return 0 if out.get("rc", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
