"""Open a folder from the dashboard in the desktop file manager.

The dashboard runs on the same machine as the files it describes, so a path
written into a note can be opened for real. This launches a desktop program, so
the web layer only exposes it on a loopback bind and never passes a shell string
— always an argv list ending in a path that already exists.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


class OpenError(Exception):
    """The path could not be shown in a file manager."""


def _launch(argv: list[str]) -> None:
    subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def _commands(target: Path, is_file: bool) -> list[list[str]]:
    """Candidate argv lists, best first, for showing ``target``."""
    folder = str(target.parent if is_file else target)

    if sys.platform == "darwin":
        return [["open", "-R", str(target)] if is_file else ["open", folder]]
    if sys.platform.startswith("win"):
        return [["explorer", f"/select,{target}"] if is_file else ["explorer", folder]]

    candidates: list[list[str]] = []
    if is_file:
        # Highlight the file itself where the file manager supports it.
        for browser, flag in (("nautilus", "--select"), ("nemo", None), ("caja", "--select")):
            if shutil.which(browser):
                candidates.append([browser, flag, str(target)] if flag else [browser, str(target)])
        if shutil.which("dolphin"):
            candidates.append(["dolphin", "--select", str(target)])
    candidates.append(["xdg-open", folder])
    if shutil.which("gio"):
        candidates.append(["gio", "open", folder])
    return candidates


def open_in_file_manager(path: str | Path) -> dict[str, str]:
    """Show ``path`` in the file manager; files are revealed in their folder."""
    if not str(path).strip():
        raise OpenError("no path given")
    target = Path(str(path)).expanduser()
    try:
        target = target.resolve(strict=True)
    except (OSError, RuntimeError):
        raise OpenError(f"nothing exists at {path}") from None

    is_file = target.is_file()
    if not is_file and not target.is_dir():
        raise OpenError(f"{target} is neither a file nor a folder")

    errors: list[str] = []
    for argv in _commands(target, is_file):
        if not shutil.which(argv[0]):
            errors.append(f"{argv[0]} not installed")
            continue
        try:
            _launch(argv)
        except OSError as error:
            errors.append(f"{argv[0]}: {error}")
            continue
        return {"opened": str(target.parent if is_file else target), "with": argv[0]}

    raise OpenError("; ".join(errors) or "no file manager available")
