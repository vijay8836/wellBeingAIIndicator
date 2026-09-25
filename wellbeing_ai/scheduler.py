"""
Periodic running.

Two ways: a foreground loop (fine for a machine that's always on), or install
a real system job -- cron on Linux, launchd on macOS, Task Scheduler on
Windows -- which is what you actually want, because it survives reboots.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .config import Config, PROJECT_ROOT

LABEL = "com.local.wellbeing-ai"


def run_once(cfg: Optional[Config] = None) -> dict:
    from .pipeline import WellbeingSystem
    ws = WellbeingSystem(cfg or Config.load())
    return ws.run_daily(deliver=True, html=True)


def run_forever(at: str = "08:00", cfg: Optional[Config] = None) -> None:
    hh, mm = (int(x) for x in at.split(":"))
    while True:
        now = datetime.now()
        nxt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(days=1)
        sleep_s = (nxt - now).total_seconds()
        print(f"[{now:%Y-%m-%d %H:%M}] next run at {nxt:%Y-%m-%d %H:%M} "
              f"({sleep_s / 3600:.1f}h)")
        try:
            time.sleep(sleep_s)
        except KeyboardInterrupt:
            print("\nstopped")
            return
        try:
            res = run_once(cfg)
            print(f"[{datetime.now():%Y-%m-%d %H:%M}] "
                  f"{res.get('date')} alerts={res.get('alerts')}")
        except Exception as exc:          # a bad day shouldn't kill the loop
            print(f"[{datetime.now():%Y-%m-%d %H:%M}] run failed: {exc}")


# --------------------------------------------------------------- installers
def install(at: str = "08:00") -> str:
    hh, mm = (int(x) for x in at.split(":"))
    python = sys.executable
    root = PROJECT_ROOT
    system = platform.system()

    if system == "Darwin":
        plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key>
  <array><string>{python}</string><string>-m</string>
    <string>wellbeing_ai</string><string>run</string></array>
  <key>WorkingDirectory</key><string>{root}</string>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>{hh}</integer>
        <key>Minute</key><integer>{mm}</integer></dict>
  <key>StandardOutPath</key><string>{root}/out/scheduler.log</string>
  <key>StandardErrorPath</key><string>{root}/out/scheduler.err</string>
</dict></plist>"""
        path = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(plist)
        subprocess.run(["launchctl", "unload", str(path)],
                       capture_output=True, check=False)
        subprocess.run(["launchctl", "load", str(path)],
                       capture_output=True, check=False)
        return f"launchd job installed: {path}"

    if system == "Windows":
        cmd = (f'schtasks /Create /F /SC DAILY /TN "WellbeingAI" '
               f'/TR "\\"{python}\\" -m wellbeing_ai run" '
               f'/ST {hh:02d}:{mm:02d}')
        subprocess.run(cmd, shell=True, check=False)
        return ("Task Scheduler entry 'WellbeingAI' created. "
                f"Set its 'Start in' folder to {root} if it fails to find "
                "the package.")

    # Linux / BSD: cron.
    line = (f"{mm} {hh} * * * cd {root} && {python} -m wellbeing_ai run "
            f">> {root}/out/scheduler.log 2>&1")
    try:
        existing = subprocess.run(["crontab", "-l"], capture_output=True,
                                  text=True, check=False).stdout
    except FileNotFoundError:
        return ("cron isn't available. Add this to your scheduler manually:\n"
                + line)
    lines = [l for l in existing.splitlines() if "wellbeing_ai run" not in l]
    lines.append(line)
    proc = subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n",
                          text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return f"Could not install cron entry. Add manually:\n{line}"
    return f"cron entry installed:\n  {line}"


def uninstall() -> str:
    system = platform.system()
    if system == "Darwin":
        path = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
        subprocess.run(["launchctl", "unload", str(path)],
                       capture_output=True, check=False)
        if path.exists():
            path.unlink()
        return "launchd job removed"
    if system == "Windows":
        subprocess.run('schtasks /Delete /F /TN "WellbeingAI"', shell=True,
                       check=False)
        return "Task Scheduler entry removed"
    existing = subprocess.run(["crontab", "-l"], capture_output=True,
                              text=True, check=False).stdout
    lines = [l for l in existing.splitlines() if "wellbeing_ai run" not in l]
    subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n", text=True,
                   capture_output=True, check=False)
    return "cron entry removed"
