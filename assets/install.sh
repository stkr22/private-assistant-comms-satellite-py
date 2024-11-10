sudo apt-get install -y libasound2-dev libportaudio2 libportaudiocpp0 portaudio19-dev python3-dev python3-venv
mkdir ~/assistant-bridge
cd ~/assistant-bridge
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
