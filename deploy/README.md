# kaggle2 deploy/ — portearchive.com `/teb2/`

This directory holds the production-deployment files for the DONUT demo at
`https://portearchive.com/teb2/`. The droplet is shared with two other
apps (the History Network at `/`, Fixie at `/fixie/` and `/api/`) so the
routing contract is documented centrally in
[`aiparallel0/site` → `PORTEARCHIVE_ROUTING.md`](https://github.com/aiparallel0/site/blob/main/PORTEARCHIVE_ROUTING.md).
Read that first.

## Install on the droplet

From a fresh checkout under `/var/www/kaggle2`:

```bash
# 1. systemd unit — binds uvicorn to 127.0.0.1:8000 with --root-path /teb2
sudo install -m 0644 deploy/kaggle2.service /etc/systemd/system/kaggle2.service
sudo systemctl daemon-reload
sudo systemctl enable --now kaggle2
sudo systemctl status kaggle2     # should show "active (running)"

# 2. nginx snippet — dropped into the apex apps directory so the site
#    repo's template auto-includes it. No manual edit of the apex vhost.
sudo install -m 0644 deploy/nginx-teb2.conf \
    /etc/nginx/snippets/portearchive-apps/teb2.conf
sudo nginx -t && sudo systemctl reload nginx
```

The snippet file's comment header is the source of truth for the install
path. If the apex `site` repo evolves the contract (e.g. moves the apps
directory), update the comment header here and the table in
`PORTEARCHIVE_ROUTING.md` together.

## What lives where on the droplet

| File                                                          | Owner repo                  |
|---------------------------------------------------------------|-----------------------------|
| `/etc/systemd/system/kaggle2.service`                         | this repo → `deploy/kaggle2.service` |
| `/etc/nginx/snippets/portearchive-apps/teb2.conf`             | this repo → `deploy/nginx-teb2.conf` |
| `/etc/nginx/snippets/portearchive-security-headers.conf`      | `aiparallel0/site` repo     |
| `/etc/nginx/sites-available/portearchive.com`                 | `aiparallel0/site` repo (via setup.sh) |

Do not edit anything in the right column from this repo — changes will be
clobbered on the next `site` deploy. Edit them in their owner repo and
open a PR if a cross-repo change is needed.

## Ports

- `127.0.0.1:8000` — uvicorn, bound loopback only. Public traffic is
  proxied via nginx at `/teb2/`. Do not bind to `0.0.0.0` — doing so
  bypasses nginx security headers, CSP, and TLS.

## Auto-deploy

This repo's `.github/workflows/deploy.yml` SSHes into the droplet on
push to `main` and runs `git pull && systemctl restart kaggle2`. The
workflow does NOT touch nginx — nginx is reloaded only when the snippet
file's content changes (manual step).
