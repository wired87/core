
clone="""
git clone --recurse-submodules https://github.com/wired87/betse_drf.git
"""
init=f"""

source workenv/bin/activate && cd betse_drf && export PYTHONPATH=$PYTHONPATH:$(pwd) && git pull
"""


gunicorn_restart=rf"""

sudo systemctl stop gunicorn 
ps aux | grep gunicorn 
sudo kill -9 <PID> 
sudo systemctl start gunicorn

"""