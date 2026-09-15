# Making the data viewable online

Two ways, both already wired up.

## 1. GitHub (static snapshot, no setup)

Double-click `push_data.bat` after a session. The `dataset/` folder on
https://github.com/greatroboticslab/3DAI/tree/main/dataset updates with the
images by material class, `metadata.xlsx` and `laser_features.csv`. This is
what a reviewer without any account on our network sees.

## 2. Live GUI over Tailscale (browse the database as it grows)

`go_online.bat` starts the scanner GUI in read-only mode (Samples and
Dataset export pages only; Capture and Hardware are hidden) and publishes it
through Tailscale.

### One-time install (needs an administrator account, about 5 minutes)

1. Download https://pkgs.tailscale.com/stable/tailscale-setup-latest-amd64.msi
   and run it (right-click, Run as administrator). Accept defaults.
2. Double-click `go_online.bat`. The first time it prints a
   `https://login.tailscale.com/a/...` link; open it and sign in (a Google or
   GitHub login is fine; this creates our lab's "tailnet").
3. In https://login.tailscale.com/admin/dns enable **MagicDNS** and
   **HTTPS Certificates** (two toggles). Run `go_online.bat` again.

It prints an address like `https://scanner-pc.tail1234.ts.net`. That is the
live view.

### Who can open it

- **Anyone with the link** (default, `tailscale funnel`): Dr. Zhang and
  reviewers need nothing installed; the https address just opens. Funnel
  was enabled for this network on 2026-09-15. The view is read-only.
- **Tailnet only**: change `funnel --bg 8501` to `serve --bg 8501` in
  `go_online.bat`; then only devices signed into our Tailscale account can
  open it.

### Turning it off

Close the `go_online.bat` window and the minimized GUI window, or run
`"C:\Program Files\Tailscale\tailscale.exe" serve reset`.

### What is exposed

Only the Streamlit GUI on port 8501 in read-only mode. The database (port
27017) and the capture API (port 8600) stay local to this PC.
