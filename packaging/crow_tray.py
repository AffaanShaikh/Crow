"""
System tray launcher for Crow-Elda, ENTRY POINT for the packaged .exe app.
PyInstaller compiles this file into crow.exe.

Does the following:
  1. Start the FastAPI backend in a background thread
  2. Serve the built React frontend from the backend (StaticFiles)
  3. Show a system tray icon with a menu
  4. Open the browser to the UI on demand
  5. Optionally start wake-word detection (headless mode)
  6. Handle clean shutdown
The goal is a system-level agent that runs in the background - like Ollama.
A tray icon is the Windows/macOS/Linux standard for "background service with
UI available on demand". The user can:
  - Left-click the tray icon -> opens the browser UI
  - Right-click -> menu with "Open Crow", "Settings", "Quit"

PROCESS MODEL
──────────────
  Main process
    ├── System tray (pystray - runs on main thread, required by macOS/Windows)
    └── Backend thread (uvicorn, daemon=True)
          ├── FastAPI app
          │     └── All existing API routes + static file serving
          └── Wake-word thread (daemon=True, optional)

  uvicorn in a thread because pystray requires the main thread for the tray icon on all platforms.
  uvicorn (the ASGI server) runs fine in a background thread.
  The backend thread is a daemon thread, so it dies when the main process exits.

PACKAGING goes like:-
  Run from the repo root:
    python packaging/build.py

  Or manually:
    npm run build       (in frontend/)
    pyinstaller packaging/crow.spec
"""

from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path


# Path resoltion for bundled resources (icons, frontend build, etc.)
# PyInstaller sets sys.frozen = True and sys._MEIPASS to the temp-extracted dir.
# In dev mode, these are absent.

def _resource_path(relative: str) -> Path:
    """
    Resolve a path to a bundled resource.
    Works both in dev (relative to this file) and in PyInstaller .exe.
    """
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)           # type: ignore[attr-defined]
    else:
        base = Path(__file__).parent

    return base / relative


def _data_dir() -> Path:
    """User data directory (tokens, chromadb, etc.) - never inside _MEIPASS."""
    from utils.paths import get_data_dir
    return get_data_dir()


# Config.
BACKEND_PORT = int(os.environ.get("PORT", 8000))
FRONTEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"
STARTUP_TIMEOUT = 60 # seconds to wait for the backend to become ready

_ICON_PATH = _resource_path("assets/crow_icon.ico")


# Backend startup

def _add_backend_to_sys_path() -> None:
    """
    In dev mode, ensure the backend directory is on sys.path.
    In packaged mode, PyInstaller handles this via the .spec pathex.
    """
    if not getattr(sys, "frozen", False):
        backend_dir = str(Path(__file__).parent / "backend")
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)


def _start_backend() -> None:
    """
    Start the FastAPI backend with uvicorn in the current thread.
    Call this in a daemon thread.
    """
    _add_backend_to_sys_path()

    # Set the working directory to where data/ lives, so relative config paths work
    os.chdir(str(_data_dir().parent))

    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",   # bind only loopback - not exposed to network
        port=BACKEND_PORT,
        log_config=None,
        access_log=False,
    )


def _wait_for_backend(timeout: float = STARTUP_TIMEOUT) -> bool:
    """
    Poll the health endpoint until it responds or the timeout expires.
    Returns True when the backend is ready.
    """
    import urllib.request
    url = f"{FRONTEND_URL}/api/v1/health"
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.3)

    return False


# System tray

def _make_icon():
    """
    Load the tray icon. Falls back to a generated icon if the file is missing.
    pystray requires a PIL Image object.
    """
    try:
        from PIL import Image
        if _ICON_PATH.exists():
            return Image.open(_ICON_PATH)
    except ImportError:
        pass

    # Fallback: generate a simple coloured square
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (64, 64), (26, 0, 8, 255))
        draw = ImageDraw.Draw(img)
        draw.ellipse([16, 16, 48, 48], fill=(212, 168, 56, 255))
        return img
    except ImportError:
        return None


def _open_ui(_icon=None, _item=None) -> None:
    """Open Crow in the default browser."""
    webbrowser.open(FRONTEND_URL)


def _quit_app(icon, _item=None) -> None:
    """Stop the tray icon and exit the process cleanly."""
    icon.stop()
    # Brief delay so the tray icon can clean up before we exit
    threading.Timer(0.5, lambda: os.kill(os.getpid(), 9)).start()


def _run_tray(ready: bool) -> None:
    """
    Build and run the system tray icon.
    Must be called from the MAIN THREAD (required by pystray on all platforms).
    """
    try:
        import pystray
        from pystray import MenuItem as Item, Menu
    except ImportError:
        # pystray not available - just open the browser and wait forever
        print("[Crow] pystray not installed. Open http://127.0.0.1:8000 in your browser.")
        webbrowser.open(FRONTEND_URL)
        threading.Event().wait()
        return

    icon_image = _make_icon()

    status_label = "● Ready" if ready else "⚠ Backend failed to start"

    menu = Menu(
        Item("Open Crow",    _open_ui, default=True),
        Menu.SEPARATOR,
        Item(status_label,   None, enabled=False),
        Menu.SEPARATOR,
        Item("Quit",         _quit_app),
    )

    icon = pystray.Icon(
        "Crow",
        icon=icon_image,
        title="Crow - Local AI",
        menu=menu,
    )

    # Auto-open browser on first launch
    if ready:
        threading.Timer(0.5, webbrowser.open, args=[FRONTEND_URL]).start()

    icon.run()


# main.py patch: serve the React frontend
# In production (packaged), the React app is built to frontend/dist/.
# FastAPI mounts it as a StaticFiles app.
# This patching is done here so main.py stays clean for dev use.

# def _patch_frontend_serving() -> None:
#     """
#     Mount the built React frontend onto FastAPI when running in packaged mode.
#     In dev mode, Vite serves the frontend - this function does nothing.
#     """
#     if not getattr(sys, "frozen", False):
#         return  # dev: Vite handles frontend serving

#     frontend_dist = _resource_path("face/dist")
#     if not frontend_dist.exists():
#         print(f"[Crow] Warning: frontend/dist not found at {frontend_dist}")
#         return

#     # We do this import lazily, after the backend module is on sys.path
#     _add_backend_to_sys_path()
#     from main import app
#     from fastapi.staticfiles import StaticFiles
#     from fastapi.responses import FileResponse

#     # Serve static assets
#     app.mount(
#         "/assets",
#         StaticFiles(directory=str(frontend_dist / "assets")),
#         name="assets",
#     )

#     # SPA catch-all: any unmatched path serves index.html
#     # This must be registered AFTER all API routes
#     @app.get("/{full_path:path}", include_in_schema=False)
#     async def _spa_fallback(full_path: str):
#         return FileResponse(str(frontend_dist / "index.html"))
def _patch_frontend_serving() -> None:
    if not getattr(sys, "frozen", False):
        return

    frontend_dist = _resource_path("face/dist")
    if not frontend_dist.exists():
        print(f"[Crow] Warning: face/dist not found at {frontend_dist}")
        return

    _add_backend_to_sys_path()
    from main import app
    from fastapi.staticfiles import StaticFiles
    from starlette.routing import Route

    # Remove the existing GET "/" route from main.py.
    # It returns a JSON info object which intercepts the React app root.
    # All /api/v1/* routes are unaffected - they registered before this runs
    # and sit earlier in app.routes, so they always match first.
    app.routes[:] = [
        r for r in app.routes
        if not (
            isinstance(r, Route)
            and r.path == "/"
            and r.methods
            and "GET" in r.methods
        )
    ]

    # Mount the entire face/dist/ as an HTML-mode static files app.
    # html=True does two things:
    #   1. Serves index.html for directory requests (handles "/")
    #   2. Serves index.html for 404s (handles React Router paths like /settings)
    # Because this is appended AFTER all existing routes, /api/v1/* routes
    # still match first - this only catches paths nothing else claimed.
    app.mount(
        "/",
        StaticFiles(directory=str(frontend_dist), html=True),
        name="frontend",
    )

    print(f"[Crow] Frontend served from {frontend_dist}")


# entry point

def main() -> None:
    """
    Application entry point.

    Boot sequence:
      1. Start the backend in a daemon thread
      2. Poll the health endpoint until ready
      3. Patch frontend serving if packaged
      4. Run the system tray on the main thread
    """
    print(f"[Crow] Starting backend on port {BACKEND_PORT}...")

    backend_thread = threading.Thread(
        target=_start_backend,
        name="crow-backend",
        daemon=True,
    )
    backend_thread.start()

    ready = _wait_for_backend()
    if ready:
        print(f"[Crow] Backend ready at {FRONTEND_URL}")
        _patch_frontend_serving()
    else:
        print("[Crow] Warning: backend did not become ready in time. Starting tray anyway.")

    _run_tray(ready)


if __name__ == "__main__":
    main()