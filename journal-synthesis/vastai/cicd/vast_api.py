#!/usr/bin/env python3
"""Stdlib-only vast.ai REST client (PHASE 0).

NON-NEGOTIABLE: this is INFRASTRUCTURE only. It provisions / destroys
rented GPUs. It NEVER touches the paper, never decides a scientific
result, never writes a number anywhere.

Design rules enforced here:
  * Pure standard library (urllib). No third-party deps so it runs on a
    bare GitHub Actions / vast.ai image with zero install.
  * The API key is read ONLY from the environment variable
    VAST_API_KEY. It is never a CLI arg, never logged, never written to
    disk. There is intentionally no flag to pass it.
  * --dry-run returns canned, clearly-fake data and makes ZERO network
    calls. The whole pipeline can be exercised with no creds / no spend.

This file does not import anything from the experiment package and has
no concept of main.tex / results; it cannot mutate them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://console.vast.ai/api/v0"
ENV_KEY = "VAST_API_KEY"  # the ONLY accepted source of the API key

# Canned, OBVIOUSLY-synthetic data returned in --dry-run so no caller
# ever mistakes it for a real instance. Costs are deliberately tiny.
_DRY_OFFER = {
    "id": 999000111,
    "dph_total": 0.123,
    "gpu_name": "DRYRUN_GPU",
    "disk_space": 64.0,
    "cuda_max_good": 12.1,
    "reliability2": 0.99,
    "inet_down": 500.0,
    "rentable": True,
    "_dry_run": True,
}
_DRY_INSTANCE = {
    "id": 888000222,
    "actual_status": "running",
    "ssh_host": "dryrun.invalid",
    "ssh_port": 22022,
    "dph_total": 0.123,
    "_dry_run": True,
}


class VastAPIError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        raise VastAPIError(
            f"{ENV_KEY} is not set. Refusing to make any vast.ai call. "
            f"Set it via an environment variable / GitHub Actions secret "
            f"(NEVER commit it)."
        )
    return key


def _request(method: str, path: str, *, params=None, body=None,
             dry_run: bool = False):
    """Single choke-point for every network call.

    When dry_run is True this raises if it is ever reached: callers must
    short-circuit BEFORE here, guaranteeing zero network in dry-run.
    """
    if dry_run:
        raise VastAPIError(
            "INTERNAL: _request reached in dry-run mode (this is a bug; "
            "dry-run must never perform network I/O)."
        )
    key = _api_key()
    params = dict(params or {})
    url = f"{API_BASE}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    data = None
    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method,
                                 headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
    except urllib.error.HTTPError as e:  # pragma: no cover - network
        detail = e.read().decode(errors="replace")[:500]
        raise VastAPIError(f"vast.ai HTTP {e.code} for {method} {path}: "
                           f"{detail}") from None
    except urllib.error.URLError as e:  # pragma: no cover - network
        raise VastAPIError(f"vast.ai network error for {method} {path}: "
                           f"{e.reason}") from None
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise VastAPIError(f"vast.ai returned non-JSON for {path}: "
                           f"{raw[:200]}") from None


def search_offers(*, max_dph: float, min_disk_gb: float,
                  gpu_name: str | None = None,
                  allow_interruptible: bool = False,
                  cuda_min: float | None = None,
                  dry_run: bool = False):
    """Return a list of candidate offers, cheapest first.

    Correctness numbers are GPU-independent so heterogeneous / cheap /
    interruptible offers are acceptable for everything EXCEPT E8 latency
    (the controller keeps E8 single-disclosed-GPU; this just finds the
    cheapest box that runs the workload).
    """
    if dry_run:
        return [dict(_DRY_OFFER)]

    # vast.ai query language: list of [field, op, value] style filters
    # serialised as a JSON "q". We keep it conservative & documented.
    q = {
        "rentable": {"eq": True},
        "dph_total": {"lte": float(max_dph)},
        "disk_space": {"gte": float(min_disk_gb)},
        "order": [["dph_total", "asc"]],
        "type": "bid" if allow_interruptible else "on-demand",
    }
    if gpu_name:
        q["gpu_name"] = {"eq": gpu_name}
    if cuda_min is not None:
        q["cuda_max_good"] = {"gte": float(cuda_min)}
    resp = _request("GET", "/bundles/",
                    params={"q": json.dumps(q)})
    offers = resp.get("offers", []) if isinstance(resp, dict) else []
    offers = [o for o in offers
              if o.get("dph_total") is not None
              and float(o["dph_total"]) <= float(max_dph)]
    offers.sort(key=lambda o: float(o["dph_total"]))
    return offers


def create_instance(offer_id: int, *, image: str, disk_gb: float,
                     onstart_cmd: str = "", label: str = "journal-exp",
                     dry_run: bool = False):
    if dry_run:
        return dict(_DRY_INSTANCE)
    body = {
        "client_id": "me",
        "image": image,
        "disk": float(disk_gb),
        "label": label,
        "runtype": "ssh",
    }
    if onstart_cmd:
        body["onstart"] = onstart_cmd
    resp = _request("PUT", f"/asks/{int(offer_id)}/", body=body)
    if not (isinstance(resp, dict) and resp.get("success", True)):
        raise VastAPIError(f"create_instance failed: {resp}")
    new_id = resp.get("new_contract") or resp.get("id")
    if new_id is None:
        raise VastAPIError(f"create_instance: no instance id in {resp}")
    return {"id": int(new_id), "raw": resp}


def instance_status(instance_id: int, *, dry_run: bool = False):
    if dry_run:
        return dict(_DRY_INSTANCE)
    resp = _request("GET", f"/instances/{int(instance_id)}/")
    inst = resp.get("instances", resp) if isinstance(resp, dict) else resp
    if isinstance(inst, list):
        inst = inst[0] if inst else {}
    return inst or {}


def ssh_endpoint(instance_id: int, *, dry_run: bool = False):
    """Return (host, port) once the instance exposes SSH, else (None, None)."""
    if dry_run:
        return (_DRY_INSTANCE["ssh_host"], _DRY_INSTANCE["ssh_port"])
    inst = instance_status(instance_id)
    host = inst.get("ssh_host") or inst.get("public_ipaddr")
    port = inst.get("ssh_port")
    status = (inst.get("actual_status") or "").lower()
    if host and port and status == "running":
        return (host, int(port))
    return (None, None)


def destroy_instance(instance_id: int, *, dry_run: bool = False):
    """Idempotent destroy. Safe to call on an already-gone instance."""
    if dry_run:
        return {"destroyed": int(instance_id), "_dry_run": True}
    try:
        resp = _request("DELETE", f"/instances/{int(instance_id)}/")
    except VastAPIError as e:
        # An already-destroyed / unknown instance must NOT raise: this
        # function is a safety teardown and has to be call-twice safe.
        msg = str(e)
        if "404" in msg or "not found" in msg.lower():
            return {"destroyed": int(instance_id), "already_gone": True}
        raise
    return {"destroyed": int(instance_id), "raw": resp}


def _main(argv=None):
    p = argparse.ArgumentParser(
        description="vast.ai REST client (infra only; API key from "
                    f"${ENV_KEY} env var ONLY).")
    p.add_argument("action",
                   choices=["search", "status", "ssh", "destroy",
                            "create"],
                   help="which call to demo")
    p.add_argument("--dry-run", action="store_true",
                   help="ZERO network: return canned synthetic data")
    p.add_argument("--max-dph", type=float, default=0.50)
    p.add_argument("--min-disk-gb", type=float, default=32.0)
    p.add_argument("--gpu-name", default=None)
    p.add_argument("--allow-interruptible", action="store_true")
    p.add_argument("--instance-id", type=int, default=None)
    p.add_argument("--offer-id", type=int, default=None)
    p.add_argument("--image", default="vastai/pytorch:latest")
    args = p.parse_args(argv)

    if args.action == "search":
        out = search_offers(max_dph=args.max_dph,
                             min_disk_gb=args.min_disk_gb,
                             gpu_name=args.gpu_name,
                             allow_interruptible=args.allow_interruptible,
                             dry_run=args.dry_run)
        print(json.dumps(out[:5], indent=2))
    elif args.action == "status":
        print(json.dumps(instance_status(args.instance_id or 0,
                                         dry_run=args.dry_run), indent=2))
    elif args.action == "ssh":
        print(json.dumps(ssh_endpoint(args.instance_id or 0,
                                      dry_run=args.dry_run)))
    elif args.action == "destroy":
        print(json.dumps(destroy_instance(args.instance_id or 0,
                                          dry_run=args.dry_run)))
    elif args.action == "create":
        print(json.dumps(create_instance(args.offer_id or 0,
                                          image=args.image,
                                          disk_gb=args.min_disk_gb,
                                          dry_run=args.dry_run), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
