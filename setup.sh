#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WITH_QE=0

usage() {
  cat <<'EOF'
Usage: ./setup_install_ubuntu.sh [--with-qe]

Installs Debian/Ubuntu/Mint apt packages needed to launch the PyQt6 GUI,
then installs requirements.txt into the current user's Python site-packages.

Options:
  --with-qe   Also install the distro Quantum ESPRESSO package and OpenMPI.
  -h, --help  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-qe)
      WITH_QE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v apt-get >/dev/null 2>&1; then
  echo "ERROR: this installer expects a Debian-based system with apt-get." >&2
  exit 1
fi

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=()
elif command -v sudo >/dev/null 2>&1; then
  SUDO=(sudo)
else
  echo "ERROR: run as root or install sudo first." >&2
  exit 1
fi

APT_PACKAGES=(
  ca-certificates
  curl
  git
  build-essential
  pkg-config
  python3
  python3-dev
  python3-pip
  libdbus-1-3
  libegl1
  libfontconfig1
  libglib2.0-0
  libgl1
  libice6
  libsm6
  libx11-6
  libxi6
  libxkbcommon-x11-0
  libxrender1
  libxcb1
  libxcb-cursor0
  libxcb-glx0
  libxcb-icccm4
  libxcb-image0
  libxcb-keysyms1
  libxcb-randr0
  libxcb-render0
  libxcb-render-util0
  libxcb-shape0
  libxcb-shm0
  libxcb-sync1
  libxcb-util1
  libxcb-xfixes0
  libxcb-xinerama0
  libxcb-xkb1
)

if [[ "${WITH_QE}" -eq 1 ]]; then
  APT_PACKAGES+=(
    openmpi-bin
    quantum-espresso
  )
fi

echo "Installing apt packages..."
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y "${APT_PACKAGES[@]}"

echo "Installing Python packages into the user site..."
python3 -m pip install --user --break-system-packages --upgrade pip setuptools wheel
python3 -m pip install --user --break-system-packages -r "${APP_DIR}/requirements.txt"

cat > "${APP_DIR}/run_qe_gui.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${APP_DIR}"
exec python3 qe.py "\$@"
EOF
chmod +x "${APP_DIR}/run_qe_gui.sh"

echo
echo "Install complete."
echo "Launch with: ${APP_DIR}/run_qe_gui.sh"
if [[ "${WITH_QE}" -eq 0 ]]; then
  echo "Quantum ESPRESSO was not installed. Re-run with --with-qe if you want distro pw.x/ph.x binaries."
fi
