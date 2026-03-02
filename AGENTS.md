# AGENTS.md - ScrapyArticles-Cisco Development Guide

## Project Overview

This is a Scrapy-based web scraping project for collecting and parsing Cisco Small Business articles. The project uses Scrapyd for deployment, includes AI-powered content extraction with OpenAI, and stores data in ChromaDB vector databases.

**Tech Stack**: Python 3.11, Scrapy 2.11.2, BeautifulSoup4, Langchain, OpenAI, ChromaDB, Pydantic, aiohttp

---

## Build, Test & Run Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Spiders
```bash
# Run spider directly (development)
scrapy crawl article_links

# Run with custom log level
scrapy crawl article_links -s LOG_LEVEL=DEBUG

# Run article parser after spider completion
python -m articles.scrapers.articles
```

### Scrapyd Deployment (Production)
```bash
# Start Scrapyd daemon
./scrapyd_manager.sh start

# Deploy project
./scrapyd_manager.sh deploy

# Schedule spider
./scrapyd_manager.sh schedule article_links

# Check status
./scrapyd_manager.sh status
./scrapyd_manager.sh jobs

# Access web interface at http://localhost:6800
```

### Testing
**Note**: This project currently has NO test suite. Tests should be added for:
- Spider link extraction
- Article parsing logic
- Data validation (Pydantic models)
- Database operations

To add tests, create `tests/` directory and use pytest:
```bash
pip install pytest pytest-cov
pytest tests/ -v
pytest tests/test_parser.py::test_specific_function  # Single test
```

### Linting & Formatting
**Note**: No linter/formatter configuration found. Recommended setup:
```bash
# Install tools
pip install black ruff mypy

# Format code
black articles/ src/

# Lint code
ruff check articles/ src/

# Type check
mypy articles/ src/
```

---

## Code Style Guidelines

### Import Organization
Follow **PEP 8** import ordering:

```python
# 1. Future imports (if needed)
from __future__ import annotations

# 2. Standard library imports (alphabetical)
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# 3. Third-party imports (alphabetical)
import aiohttp
import requests
from bs4 import BeautifulSoup, Tag
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

# 4. Local application imports
from config import ARTICLES_DATA_DIR, LOGS_DIR
from src.constants import CISCO_CATALYST_1200_SERIES
```

**Key patterns**:
- Use `from __future__ import annotations` for forward references
- Group imports by category with blank lines
- Use absolute imports (`from config import X`) not relative
- Alphabetize within groups

### Type Hints

**REQUIRED** for all function signatures and class attributes:

```python
# Function signatures with type hints
def parse_table(soup: BeautifulSoup, obj: Dict[str, Any]) -> Dict[str, Any]:
    """Parse table data."""
    pass

# Use typing module for complex types
from typing import Optional, Union, List, Dict, Any, Sequence

# Pydantic models for data validation
class Article(BaseModel):
    series: str
    title: str
    document_id: str
    objective: Optional[str] = None
    applicable_devices: List[Dict[str, Any]] = Field(default_factory=list)

# Type aliases for readability
CollectionType = Literal["admin_guide", "cli_guide", "article"]

# Optional and Union types
def get_objective(self, soup: BeautifulSoup) -> Optional[str]:
    pass
```

**Patterns**:
- Use `Optional[T]` instead of `T | None` (Python 3.9 compatibility)
- Use `List`, `Dict` from typing (not `list`, `dict`)
- Use `Union[A, B]` for multiple types
- Pydantic models for structured data validation

### Naming Conventions

**Classes**: `PascalCase`
```python
class ArticleParser:
class LinksDict(BaseModel):
class ArticlesLogFormatter(logformatter.LogFormatter):
```

**Functions/Methods**: `snake_case`
```python
def get_article_links_after_spidering():
def convert_series_to_product_family(abbreviation: str):
def _get_title(self, soup: BeautifulSoup):  # Private method
```

**Variables**: `snake_case`
```python
article_links = []
document_id = "smb1234"
MAX_RETRIES = 3  # Constants in UPPER_CASE
```

**Private methods**: Prefix with single underscore
```python
def _get_title(self, soup: BeautifulSoup) -> str:
def _process_step(self, element: Tag):
def _clean_soup(soup: BeautifulSoup):
```

**Constants**: `UPPER_CASE_WITH_UNDERSCORES`
```python
DEFAULT_REQUEST_TIMEOUT = 30
MAX_LOG_FILE_SIZE = 1_000_000
USER_AGENT = "Mozilla/5.0 ..."
```

### Docstrings

Use **Google-style docstrings** with type information:

```python
def transform_catalyst_1000_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform Catalyst 1000 data based on specific business rules.
    
    Transformation rules:
    1. For the "fan" property, converts 'Y' to 'No' and otherwise to 'Yes'
    2. Appends "kg" to "unit_weight"
    3. Updates the "rj-45_ports" to "x Gigabit Ethernet" format
    4. Transforms the "uplink_ports" based on provided logic
    5. Adds calculated forwarding_rate, switching_capacity, and mtbf values
    
    Args:
        data: Raw scraped data dictionary
        
    Returns:
        Transformed data dictionary
    """
```

**Module docstrings**:
```python
"""
Scrapes Cisco's Support Page Datasheets for product information.

This module provides functionality to scrape product datasheets from Cisco's
support pages and extract structured data about various networking devices
including switches, access points, and routers.

Example:
    Basic usage:
        from datasheets import main, DEFAULT_URLS
        main(DEFAULT_URLS)
"""
```

### Error Handling

**Logging over print statements**:
```python
# Setup logger
logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.info("Processing concept: %s", concept)
logger.warning("Invalid URL config: %s", url_config)
logger.error("Request failed for %s: %s", url, e)
logger.debug("Row cells: %s", cells)
```

**Try-except patterns**:
```python
# Specific exception handling
try:
    request = make_request(url=url)
    request.raise_for_status()
except requests.RequestException as e:
    logger.error("Request failed for %s: %s", url, e)
    raise

# Continue on non-critical errors
try:
    article = self.parser.parse(soup, url, series)
except Exception as e:
    logger.error(f"Error parsing article {i}: {e}")
    if not self.continue_on_failure:
        raise

# Graceful degradation
try:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf8")
    logger.addHandler(file_handler)
except Exception as e:
    logger.warning(f"Failed to set up file logging: {e}")
```

**Don't suppress errors silently**:
```python
# BAD
try:
    risky_operation()
except:
    pass

# GOOD
try:
    risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    # Handle or re-raise
```

### Path Handling

**Always use `pathlib.Path`**:
```python
from pathlib import Path

# Configure paths in config.py
CWD = Path.cwd()
DATA_DIR = CWD / "data"
LOGS_DIR = CWD / "logs"

# Create directories safely
DATA_DIR.mkdir(parents=True, exist_ok=True)

# File operations
output_file = DATA_DIR / "output.json"
with output_file.open("w", encoding="utf-8") as f:
    json.dump(data, f)
```

### Data Validation

**Use Pydantic models for structured data**:
```python
from pydantic import BaseModel, Field, field_validator

class LinksDict(BaseModel):
    """Type-safe representation of scraped links."""
    url: str
    family: str
    
    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        if not v or not isinstance(v, str) or not v.startswith("http"):
            raise ValueError("Invalid URL")
        return v
```

### Async/Await Patterns

```python
async def scrape_all(self) -> List[Optional[Article]]:
    """Scrape all articles with improved error handling."""
    html_contents = await self._fetch_all_urls()
    # Process...

async def _fetch_url_with_retries(self, url: str, max_retries: int = 3) -> Optional[str]:
    """Fetch single URL with retries."""
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2**attempt)  # Exponential backoff
    return None
```

---

## Project Structure

```
scrapy-articles/
├── articles/                 # Scrapy project
│   ├── spiders/             # Spider modules
│   │   └── links.py         # Main article links spider
│   ├── scrapers/            # Custom scrapers
│   │   └── articles.py      # Article content scraper
│   ├── utils/               # Utilities
│   │   └── logger.py        # Custom logging formatter
│   ├── items.py             # Scrapy item definitions
│   ├── pipelines.py         # Item processing pipelines
│   ├── middlewares.py       # Custom middlewares
│   └── settings.py          # Scrapy settings
├── src/                     # Application source
│   ├── services/            # Business logic services
│   ├── db/                  # Database operations
│   ├── utils/               # Shared utilities
│   └── constants.py         # Project-wide constants
├── data/                    # Data output directory
│   ├── datasheets/
│   ├── articles/
│   ├── admin_guides/
│   └── vector_db/           # ChromaDB storage
├── logs/                    # Log files
├── config.py               # Path and environment config
├── requirements.txt        # Python dependencies
├── scrapy.cfg             # Scrapy deployment config
├── scrapyd.conf           # Scrapyd daemon config
└── setup.py               # Package setup
```

---

## Environment Variables

Create `.env` file with:
```bash
# Required
OPENAI_API_KEY=sk-...
SCRAPEOPS_API_KEY=...

# Optional
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

Access in code:
```python
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

---

## Important Patterns & Conventions

### Scrapy Spiders
```python
class ArticleLinks(Spider):
    name = "article_links"
    start_urls = [...]
    
    def __init__(self, *args, **kwargs):
        dispatcher.connect(self.spider_closed, scrapy.signals.spider_closed)
        super().__init__(*args, **kwargs)
    
    def parse(self, response: scrapy.http.Response):
        # Yield items or follow links
        yield {"url": url, "family": family}
```

### BeautifulSoup Parsing
```python
soup = BeautifulSoup(html_content, "lxml")  # Use lxml parser

# Find elements
element = soup.find("div", id="content")
elements = soup.find_all("a", href=True)

# CSS selectors
breadcrumb = soup.select_one('#fw-breadcrumb > ul > li:last-child')

# Extract text
text = element.get_text(strip=True, separator=" ")
```

### Data Persistence
```python
# JSON output
with open(output_path, "w", encoding="utf8") as file:
    json.dump(data_list, file, indent=4, ensure_ascii=True)

# Pydantic serialization
articles_data = [article.model_dump() for article in articles]
```

---

## Common Tasks

### Adding a New Spider
1. Create spider file in `articles/spiders/`
2. Define `name`, `start_urls`, and `parse()` method
3. Update `setup.py` if needed
4. Deploy: `./scrapyd_manager.sh deploy`

### Adding a New Scraper Service
1. Create module in `src/services/`
2. Follow existing patterns (see `datasheets.py`)
3. Use `config.py` for paths
4. Add logging with module logger
5. Use type hints throughout

### Modifying Article Parsing
- Main logic in `articles/scrapers/articles.py`
- `ArticleParser` class handles HTML parsing
- Update `ArticleStep` Pydantic model for new fields
- Test with: `python -m articles.scrapers.articles`

---

## CI/CD

GitHub Actions workflow (`.github/workflows/weekly-crawl.yml`):
- Runs weekly on Sunday at 2 AM UTC
- Manual trigger available via `workflow_dispatch`
- Installs dependencies, runs spider, commits results
- Requires secrets: `SCRAPEOPS_API_KEY`, `OPENAI_API_KEY`

---

## Best Practices

1. **Always use type hints** - Required for function signatures
2. **Use Pydantic for data validation** - Catch errors early
3. **Log, don't print** - Use appropriate log levels
4. **Path objects over strings** - Use `pathlib.Path`
5. **Handle errors explicitly** - No bare `except:` clauses
6. **Document public APIs** - Google-style docstrings
7. **Rate limit external requests** - Respect robots.txt
8. **Validate scraped data** - Check for None/empty values
9. **Use constants** - Define once in `src/constants.py`
10. **Test before deploying** - Run locally before Scrapyd deploy

---

## Troubleshooting

**Scrapyd won't start**: Check if port 6800 is in use: `lsof -i :6800`

**Spider not found**: Redeploy with `./scrapyd_manager.sh deploy`

**Import errors**: Activate venv: `source env/bin/activate`

**Parsing failures**: Enable debug logging: `scrapy crawl article_links -s LOG_LEVEL=DEBUG`

**Rate limiting**: Adjust `DOWNLOAD_DELAY` in `articles/settings.py`
