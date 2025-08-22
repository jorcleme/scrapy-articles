#!/bin/bash

# DigitalOcean Droplet setup script
# Run this on a fresh Ubuntu 22.04 droplet

set -e

echo "🌊 Setting up DigitalOcean droplet for Scrapy crawler..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3 python3-pip python3-venv git nginx supervisor cron

# Create application user
sudo useradd -m -s /bin/bash scrapy-user
sudo usermod -aG sudo scrapy-user

# Create application directory
sudo mkdir -p /opt/scrapy-articles
sudo chown scrapy-user:scrapy-user /opt/scrapy-articles

# Clone repository (you'll need to do this step manually)
echo "📂 Clone your repository to /opt/scrapy-articles"
echo "   sudo -u scrapy-user git clone <your-repo-url> /opt/scrapy-articles"

# Create virtual environment
sudo -u scrapy-user python3 -m venv /opt/scrapy-articles/venv

# Create systemd service
cat > /tmp/scrapy-scheduler.service << 'EOF'
[Unit]
Description=Scrapy Articles Scheduler
After=network.target

[Service]
Type=simple
User=scrapy-user
WorkingDirectory=/opt/scrapy-articles
Environment=PATH=/opt/scrapy-articles/venv/bin
Environment=PYTHONPATH=/opt/scrapy-articles
ExecStart=/opt/scrapy-articles/venv/bin/python scheduler.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo mv /tmp/scrapy-scheduler.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable scrapy-scheduler

# Create log rotation
cat > /tmp/scrapy-logrotate << 'EOF'
/opt/scrapy-articles/scheduler.log {
    weekly
    rotate 4
    compress
    delaycompress
    missingok
    notifempty
    create 644 scrapy-user scrapy-user
}
EOF

sudo mv /tmp/scrapy-logrotate /etc/logrotate.d/scrapy-articles

# Create environment file for secrets
sudo -u scrapy-user touch /opt/scrapy-articles/.env
echo "OPENAI_API_KEY=your_openai_api_key_here" | sudo -u scrapy-user tee /opt/scrapy-articles/.env

echo "✅ DigitalOcean setup complete!"
echo ""
echo "Manual steps remaining:"
echo "1. Clone your repository to /opt/scrapy-articles"
echo "2. Install dependencies: sudo -u scrapy-user /opt/scrapy-articles/venv/bin/pip install -r requirements.txt"
echo "3. Add your OpenAI API key to /opt/scrapy-articles/.env"
echo "4. Start the service: sudo systemctl start scrapy-scheduler"
echo "5. Check status: sudo systemctl status scrapy-scheduler"

EOF
