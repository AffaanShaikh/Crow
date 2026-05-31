"""
Build automation script

Usage:
    python packaging/build.py               # full build
    python packaging/build.py --skip-npm    # skip frontend build (for backend-only changes)
    python packaging/build.py --skip-pyi    # skip PyInstaller (for frontend-only changes)
    python packaging/build.py --installer   # also build the .exe installer (Windows + Inno Setup)

BUILD PIPELINE:-
  1. Verify prerequisites (npm, pyinstaller, ollama)
  2. Build the React frontend:  npm ci && npm run build  ->  frontend/dist/
  3. Copy frontend/dist/ into the PyInstaller bundle via crow.spec datas
  4. Run PyInstaller:  pyinstaller packaging/crow.spec  ->  dist/crow/
  5. (Optional) Run Inno Setup ISCC  ->  dist/installer/CrowSetup-x.x.x.exe

Automated script:-
  Manual builds have four steps across three tools. Miss one and the build
  is wrong in ways that may not be obvious (stale frontend, wrong version).
  The build script is the single source of truth for how the app is built.
  It's also CI/CD-ready - a GitHub Action can run this script on every tag.

Why not a Makefile?
  Python scripts are cross-platform. Makefiles require make (absent on Windows
  by default). This script runs on Windows, macOS, and Linux without changes.

Version management:
  Version is read from backend/config.py (app_version field) and written
  into the installer script automatically. Change it once, propagates everywhere.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT     = Path(__file__).parent.parent # repo root
BACKEND  = ROOT #/ "backend"
FRONTEND = ROOT / "face"
DIST     = ROOT / "dist"
SPEC     = ROOT / "packaging" / "crow.spec"
ISS      = ROOT / "packaging" / "crow_installer.iss"


# utilities

# def _run(cmd: list[str], cwd: Path | None = None, desc: str = "") -> None:
#     """Run a command, streaming output. Exits on failure."""
#     print(f"\n{'─'*60}")
#     if desc:
#         print(f"  {desc}")
#     print(f"  $ {' '.join(cmd)}")
#     print(f"{'─'*60}")

#     result = subprocess.run(cmd, cwd=cwd)
#     if result.returncode != 0:
#         print(f"\n[ERROR] Command failed with code {result.returncode}")
#         sys.exit(result.returncode)
def _run(cmd: list[str], cwd: Path | None = None, desc: str = "") -> None:
    print(f"\n{'─'*60}")
    if desc:
        print(f"  {desc}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─'*60}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=False,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] Failed to start command: {cmd[0]}")
        print(f"[ERROR] cwd={cwd}")
        print(f"[ERROR] {e}")
        sys.exit(1)

    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with code {result.returncode}")
        sys.exit(result.returncode)


def _require(binary: str, install_hint: str) -> None:
    """Check that a binary is in PATH."""
    if not shutil.which(binary):
        print(f"[ERROR] '{binary}' not found in PATH.")
        print(f"  Install: {install_hint}")
        sys.exit(1)


def _read_version() -> str:
    """Read app_version from backend/config.py."""
    config_src = (BACKEND / "config.py").read_text()
    m = re.search(r'app_version:\s*str\s*=\s*["\']([^"\']+)["\']', config_src)
    return m.group(1) if m else "0.0.0"


def _patch_iss_version(version: str) -> None:
    """Update #define AppVersion in the Inno Setup script."""
    iss = ISS.read_text()
    iss = re.sub(
        r'(#define AppVersion\s+)["\'][^"\']*["\']',
        f'\\1"{version}"',
        iss,
    )
    ISS.write_text(iss)
    print(f"  [info] Installer version set to {version}")


# build steps

def step_verify() -> None:
    print("\n[1/4] Verifying prerequisites...")
    _require("node",          "https://nodejs.org")
    _require("npm",           "https://nodejs.org")
    _require("pyinstaller",   "pip install pyinstaller")
    print("  ✓ All prerequisites found")


# def step_frontend() -> None:
#     print("\n[2/4] Building React frontend...")
#     _require("npm", "https://nodejs.org")
#     _run(["npm", "ci"],          cwd=FRONTEND, desc="Install npm dependencies")
#     _run(["npm", "run", "build"], cwd=FRONTEND, desc="Vite production build")

#     dist_index = FRONTEND / "dist" / "index.html"
#     if not dist_index.exists():
#         print(f"[ERROR] Expected {dist_index} - build may have failed")
#         sys.exit(1)

#     print(f"  ✓ Frontend built to {FRONTEND / 'dist'}")
def step_frontend() -> None:
    print("\n[2/4] Building React frontend...")

    if not FRONTEND.exists():
        print(f"[ERROR] Frontend folder not found: {FRONTEND}")
        sys.exit(1)

    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm:
        print("[ERROR] npm not found")
        sys.exit(1)

    _run([npm, "ci"], cwd=FRONTEND, desc="Install npm dependencies")
    _run([npm, "run", "build"], cwd=FRONTEND, desc="Vite production build")


def step_pyinstaller() -> None:
    print("\n[3/4] Running PyInstaller...")
    _require("pyinstaller", "pip install pyinstaller")

    # clean previous build to avoid stale files
    stale = ROOT / "dist" / "crow"
    if stale.exists():
        print(f"  Removing stale build at {stale}")
        shutil.rmtree(stale)

    _run(
        ["pyinstaller", str(SPEC), "--noconfirm"],
        cwd=ROOT,
        desc="PyInstaller onedir build",
    )

    exe = ROOT / "dist" / "crow" / ("crow.exe" if sys.platform == "win32" else "crow")
    if not exe.exists():
        print(f"[ERROR] Expected {exe} - PyInstaller may have failed")
        sys.exit(1)

    print(f"  ✓ App built to {ROOT / 'dist' / 'crow'}")


def step_installer(version: str) -> None:
    print("\n[4/4] Building installer...")

    if sys.platform != "win32":
        print("  [skip] Inno Setup installer is Windows-only")
        print("  For macOS: use create-dmg or Platypus")
        print("  For Linux: use AppImage or Flatpak")
        return

    iscc = shutil.which("ISCC") or shutil.which("iscc")
    if not iscc:
        # Try default install path
        default = Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe")
        if default.exists():
            iscc = str(default)
        else:
            print("  [skip] ISCC (Inno Setup compiler) not found in PATH")
            print("  Download from https://jrsoftware.org/isinfo.php")
            return

    _patch_iss_version(version)
    (ROOT / "dist" / "installer").mkdir(parents=True, exist_ok=True)
    _run([iscc, str(ISS)], desc="Inno Setup compilation")

    installer = ROOT / "dist" / "installer" / f"CrowSetup-{version}.exe"
    if installer.exists():
        size_mb = installer.stat().st_size / 1024 / 1024
        print(f"  ✓ Installer: {installer} ({size_mb:.1f} MB)")
    else:
        print(f"  [warn] Installer not found at expected path {installer}")


# main

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Crow for distribution")
    parser.add_argument("--skip-npm",    action="store_true", help="Skip frontend build")
    parser.add_argument("--skip-pyi",    action="store_true", help="Skip PyInstaller")
    parser.add_argument("--installer",   action="store_true", help="Also build installer")
    args = parser.parse_args()

    version = _read_version()
    print(f"\n{'='*60}")
    print(f"  Building Crow v{version}")
    print(f"  Platform: {sys.platform}")
    print(f"  Root:     {ROOT}")
    print(f"{'='*60}")

    step_verify()

    if not args.skip_npm:
        step_frontend()
    else:
        print("\n[2/4] Frontend build skipped (--skip-npm)")

    if not args.skip_pyi:
        step_pyinstaller()
    else:
        print("\n[3/4] PyInstaller skipped (--skip-pyi)")

    if args.installer:
        step_installer(version)
    else:
        print("\n[4/4] Installer skipped (use --installer to build)")

    print(f"\n{'='*60}")
    print(f"  ✓ Build complete - dist/crow/")
    if args.installer and sys.platform == "win32":
        print(f"  ✓ Installer  - dist/installer/CrowSetup-{version}.exe")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()