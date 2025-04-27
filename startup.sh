#!/bin/bash

# Exit if any command fails
set -e

# Check for --skip-install flag
INSTALL=false
for arg in "$@"; do
  if [ "$arg" == "--init" ]; then
    INSTALL=true
  fi
done


if [ "$INSTALL" = true ]; then
  # Install system packages
  sudo apt update && sudo apt install python3.11-venv git tmux nginx -y

  # Set up Python virtual environment and project
  python3 -m venv workenv
  git clone https://github.com/wired87/bm.git
fi

# Always set up environment
source workenv/bin/activate
cd bm
export PYTHONPATH=$PYTHONPATH:$(pwd)
if [ "$INSTALL" = true ]; then
  pip install --upgrade pip
  pip install -r r.txt
fi

# Django setup
python manage.py migrate
python manage.py collectstatic --noinput

# Gunicorn systemd service setup
if [ ! -f "/etc/systemd/system/gunicorn.service" ]; then
  sudo bash -c "cat > /etc/systemd/system/gunicorn.service" <<EOF
[Unit]
Description=gunicorn daemon
After=network.target

[Service]
User=$USER
Group=www-data
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/../workenv/bin/gunicorn --workers 2 --bind unix:$(pwd)/gunicorn.sock bm.wsgi:application

[Install]
WantedBy=multi-user.target
EOF

  sudo systemctl daemon-reload
  sudo systemctl enable gunicorn
fi

# Nginx setup
if [ ! -f "/etc/nginx/sites-available/bm" ]; then
  sudo bash -c "cat > /etc/nginx/sites-available/bm" <<EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://unix:$(pwd)/gunicorn.sock;
    }
}
EOF

  sudo ln -sf /etc/nginx/sites-available/bm /etc/nginx/sites-enabled/bm
  sudo nginx -t
  sudo systemctl restart nginx
fi

# Start Gunicorn manually (immediate)
gunicorn --workers 2 --bind 0.0.0.0:8080 --timeout 300 bm.wsgi:application
