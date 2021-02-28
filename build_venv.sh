# You should be inside NMR-EsPy dir when you run this 
# Assuming that python3.9 is being used

# create venv
python3 -m venv nmrespy-venv
# activate venv
source nmrespy-venv/bin/activate
# install required packages
pip install -r venv_requirements.txt
# copy nmrespy to venv as well
cp -r nmrespy nmrespy-venv/lib/python3.9/site-packages/
