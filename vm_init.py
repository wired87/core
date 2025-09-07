
clone="""
git clone --recurse-submodules https://github.com/wired87/betse_drf.git
"""
init=f"""

source workenv/bin/activate && cd BestBrain && export PYTHONPATH=$PYTHONPATH:$(pwd) && git pull
"""

gunicorn_restart=rf"""
sudo nginx -t
sudo systemctl reload nginx
sudo systemctl restart gunicorn
sudo systemctl restart fail2ban
sudo journalctl -u gunicorn -f
"""