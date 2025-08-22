import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Common paths
CWD = Path.cwd()

CURRENT_FILE = Path(__file__).resolve()

DATA_DIR = CWD / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = CWD / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SRC_DIR = CWD / "src"
SRC_DIR.mkdir(parents=True, exist_ok=True)

ARTICLES_DIR = CWD / "articles"
ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

ARTICLES_DATA_DIR = ARTICLES_DIR / "data"

SPIDER_DIR = ARTICLES_DIR / "spiders"

DATASHEETS_DATA_DIR = DATA_DIR / "datasheets"
DATASHEETS_DATA_DIR.mkdir(parents=True, exist_ok=True)

ADMIN_GUIDE_DATA_DIR = DATA_DIR / "admin_guides"
ADMIN_GUIDE_DATA_DIR.mkdir(parents=True, exist_ok=True)

CLI_GUIDE_DATA_DIR = DATA_DIR / "cli_guides"
CLI_GUIDE_DATA_DIR.mkdir(parents=True, exist_ok=True)

QUICK_RESOURCES_DATA_DIR = DATA_DIR / "quick_resources"
QUICK_RESOURCES_DATA_DIR.mkdir(parents=True, exist_ok=True)

VIDEOS_DATA_DIR = DATA_DIR / "videos"
VIDEOS_DATA_DIR.mkdir(parents=True, exist_ok=True)

ETC_DATA_DIR = DATA_DIR / "etc"
ETC_DATA_DIR.mkdir(parents=True, exist_ok=True)

CHROMA_DATA_DIR = DATA_DIR / "vector_db"
CHROMA_DATA_DIR.mkdir(parents=True, exist_ok=True)


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
