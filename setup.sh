sudo apt install python3-venv python3-full -y
python3 -m venv .venv
source .venv/bin/activate
which python
pip install ipykernel
python -m ipykernel install --user --name=venv
sudo apt install -y \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-xkb1 \
    libxrender1 \
    libx11-xcb1

sudo apt install -y \
  libxcb-shape0 \
  libxcb-xfixes0 \
  libxcb-render0 \
  libxcb-shm0

export DISPLAY=:0
export QT_QPA_PLATFORM=xcb