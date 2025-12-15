
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

raw_init=r"""
sudo apt update && sudo apt install python3.11-venv git tmux -y && \
python3 -m venv workenv && git clone https://github.com/wired87/BestBrain.git && \
export PYTHONPATH=$PYTHONPATH:$(pwd) && source workenv/bin/activate && \
cd BestBrain && pip install -r r.txt  && \
python manage.py migrate && python manage.py collectstatic && \
git clone https://github.com/wired87/_google.git && \
git clone https://github.com/wired87/file.git && \
git clone https://github.com/wired87/embedder.git && \
git clone https://github.com/wired87/qf_sim.git && \
git clone https://github.com/wired87/_betse.git && \
git clone https://github.com/wired87/utils.git && \
"""