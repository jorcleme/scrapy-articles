# ScrapyArticles-Cisco Small Business Dev Team

ScrapyArticles-Cisco is a Scrapy spider that collects and categorizes Cisco Small Business article links by family name. The spider is scheduled to run on a weekly basis, providing an efficient way to stay updated with the latest Cisco Small Business resources.

## Requirements

- Python 3.11.2
- Scrapy
- Scrapyd
- Scrapyd-client

## Installation

1. Clone this repository.
2. Create and activate a virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Linux/Mac
# or
env\Scripts\activate     # On Windows
```

3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Scrapyd Configuration

This project is configured to work with Scrapyd for deployment and management of scrapy spiders.

### Configuration Files

#### `scrapy.cfg`

```ini
[settings]
default = articles.settings

[deploy:production]
url = http://165.22.25.128/
project = articles

[deploy:local]
url = http://localhost:6800/
project = articles
```

#### `scrapyd.conf`

```ini
[scrapyd]
application       = scrapyd.app.application
bind_address      = 127.0.0.1
http_port         = 6800
logs_dir          = logs
eggs_dir          = eggs
dbs_dir           = dbs
max_proc_per_cpu  = 4
jobs_to_keep      = 5
```

#### `setup.py`

```python
from setuptools import setup, find_packages

setup(
    name         = 'articles',
    version      = '1.0',
    packages     = find_packages(),
    entry_points = {'scrapy': ['settings = articles.settings']},
)
```

### Quick Start with Scrapyd

1. **Start Scrapyd daemon:**

```bash
# Using the management script
./scrapyd_manager.sh start

# Or directly
./env/bin/scrapyd
```

2. **Deploy your project:**

```bash
# Using the management script
./scrapyd_manager.sh deploy

# Or directly
./env/bin/scrapyd-deploy local -p articles
```

3. **Access web interface:**
   - Open http://localhost:6800 in your browser

### Management Script

A convenient management script (`scrapyd_manager.sh`) is provided for common operations:

```bash
# Start/stop scrapyd daemon
./scrapyd_manager.sh start
./scrapyd_manager.sh stop

# Check daemon status
./scrapyd_manager.sh status

# Deploy project
./scrapyd_manager.sh deploy

# List projects and spiders
./scrapyd_manager.sh list

# Check current jobs
./scrapyd_manager.sh jobs

# Schedule a spider
./scrapyd_manager.sh schedule article_links

# Show web interface URL
./scrapyd_manager.sh web
```

### API Usage

Interact with Scrapyd via HTTP API:

#### Check daemon status:

```bash
curl http://localhost:6800/daemonstatus.json
```

#### List projects:

```bash
curl http://localhost:6800/listprojects.json
```

#### List spiders in articles project:

```bash
curl http://localhost:6800/listspiders.json?project=articles
```

#### Schedule a spider:

```bash
curl http://localhost:6800/schedule.json -d project=articles -d spider=article_links
```

#### Check job status:

```bash
curl http://localhost:6800/listjobs.json?project=articles
```

### Deployment Workflow

1. Make changes to your spiders in the `articles/spiders/` directory
2. Test locally with: `scrapy crawl article_links`
3. Deploy to scrapyd: `./scrapyd_manager.sh deploy`
4. Schedule spider: `./scrapyd_manager.sh schedule article_links`
5. Monitor via web interface at http://localhost:6800

### Available Spiders

- **article_links**: Collects and categorizes Cisco Small Business article links

### Logs and Data

- **Logs**: Available in `logs/` directory and via web interface
- **Data**: Scraped data stored in `articles/data/`
- **Database**: SQLite databases in `articles/dbs/`

### Troubleshooting

#### Scrapyd won't start

- Check if port 6800 is already in use: `lsof -i :6800`
- Verify configuration files are correct
- Check logs for error messages

#### Deployment fails

- Ensure `setup.py` is properly configured
- Verify virtual environment is activated
- Check that scrapyd daemon is running

#### Spider not found

- Redeploy the project: `./scrapyd_manager.sh deploy`
- Verify spider exists: `./scrapyd_manager.sh list`

### Environment Configuration

This project uses a Python virtual environment located at `env/`. The environment includes:

- Python 3.11.2
- Scrapy 2.11.2
- Scrapyd 1.4.3
- Scrapyd-client 1.2.3
- Additional dependencies as listed in `requirements.txt`

### Project Structure

```
scrapy-articles/
├── articles/                 # Main scrapy project
│   ├── spiders/             # Spider modules
│   ├── data/                # Scraped data output
│   ├── dbs/                 # SQLite databases
│   └── logs/                # Scrapy logs
├── env/                     # Python virtual environment
├── scrapy.cfg               # Scrapy deployment configuration
├── scrapyd.conf            # Scrapyd daemon configuration
├── setup.py                # Python package setup
├── scrapyd_manager.sh      # Management script
└── requirements.txt        # Python dependencies
```

### Development

For local development without scrapyd:

```bash
# Activate virtual environment
source env/bin/activate

# Run spider directly
scrapy crawl article_links

# Run with custom settings
scrapy crawl article_links -s LOG_LEVEL=DEBUG
```
