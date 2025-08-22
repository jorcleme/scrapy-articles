#!/bin/bash

# Post-crawl Article Parser
# Runs after the spider completes to parse articles with LLM

set -e

# Configuration
PROJECT_DIR="/home/jclemens/projects/scrapy-articles"
ARTICLES_DATA="$PROJECT_DIR/articles/data/articles.json"
LOG_FILE="/var/log/cisco-article-parser.log"
VENV_PATH="$PROJECT_DIR/env"

# Function to log with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to activate virtual environment
activate_venv() {
    if [ -f "$VENV_PATH/bin/activate" ]; then
        source "$VENV_PATH/bin/activate"
        log "✅ Virtual environment activated"
    else
        log "❌ Virtual environment not found at $VENV_PATH"
        exit 1
    fi
}

# Function to check if articles need parsing
check_articles() {
    if [ ! -f "$ARTICLES_DATA" ]; then
        log "⚠️  No articles data found at $ARTICLES_DATA"
        return 1
    fi
    
    local article_count
    article_count=$(python3 -c "
import json
try:
    with open('$ARTICLES_DATA', 'r') as f:
        data = json.load(f)
    print(len(data) if isinstance(data, list) else 1)
except:
    print(0)
")
    
    log "📊 Found $article_count articles to process"
    
    if [ "$article_count" -eq 0 ]; then
        return 1
    fi
    
    return 0
}

# Function to run article parsing
parse_articles() {
    log "🔄 Starting article parsing with LLM..."
    
    cd "$PROJECT_DIR"
    
    # Run the article parsing service
    python3 -c "
import sys
sys.path.append('.')
from articles.services.articles import ArticleService
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize service
service = ArticleService()

# Load articles
with open('$ARTICLES_DATA', 'r') as f:
    articles_data = json.load(f)

if not isinstance(articles_data, list):
    articles_data = [articles_data]

print(f'Processing {len(articles_data)} articles...')

success_count = 0
for i, article_data in enumerate(articles_data, 1):
    try:
        print(f'\\n=== Processing Article {i}/{len(articles_data)} ===')
        
        # Extract steps using LLM
        steps = service.extract_steps(
            html_content=article_data.get('html', ''),
            url=article_data.get('url', ''),
            title=article_data.get('title', '')
        )
        
        print(f'✅ Extracted {len(steps)} steps from article')
        success_count += 1
        
    except Exception as e:
        print(f'❌ Failed to process article {i}: {e}')
        continue

print(f'\\n🎉 Processing complete! Successfully parsed {success_count}/{len(articles_data)} articles')
"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log "✅ Article parsing completed successfully"
    else
        log "❌ Article parsing failed with exit code $exit_code"
        return 1
    fi
}

# Function to cleanup old data
cleanup_old_data() {
    log "🧹 Cleaning up old data..."
    
    # Archive current data with timestamp
    if [ -f "$ARTICLES_DATA" ]; then
        local timestamp
        timestamp=$(date +"%Y%m%d_%H%M%S")
        local archive_file="$PROJECT_DIR/articles/data/archive/articles_$timestamp.json"
        
        mkdir -p "$PROJECT_DIR/articles/data/archive"
        cp "$ARTICLES_DATA" "$archive_file"
        log "📁 Archived data to $archive_file"
        
        # Keep only last 4 weeks of archives
        find "$PROJECT_DIR/articles/data/archive" -name "articles_*.json" -mtime +28 -delete
        log "🗑️  Cleaned up old archives"
    fi
}

# Main execution
main() {
    log "🏁 Starting post-crawl article parsing"
    
    # Activate virtual environment
    activate_venv
    
    # Check if we have articles to parse
    if ! check_articles; then
        log "ℹ️  No articles to parse. Exiting."
        exit 0
    fi
    
    # Parse articles with LLM
    if parse_articles; then
        log "🎉 Article parsing completed successfully"
        
        # Cleanup old data
        cleanup_old_data
    else
        log "💥 Article parsing failed"
        exit 1
    fi
    
    log "✨ Post-crawl processing complete"
}

# Run main function
main "$@"
