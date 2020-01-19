command -v python3 >/dev/null 2>&1 || { echo >&2 "I require python3.  Aborting."; exit 1; }
command -v pip3 >/dev/null 2>&1 || { echo >&2 "I require python3.  Aborting."; exit 1; }
pip3 install keras --user
pip3 install numpy --user
pip3 install scipy --user
pip3 install sklearn --user
pip3 install h5py --user

