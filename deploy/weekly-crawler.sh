#!/bin/bash

# Weekly Cisco SMB Articles Crawler
# This script triggers the spider on your existing Scrapyd server

set -e  # Exit on any error

# Configuration
SCRAPYD_URL="http://165.22.25.128:6800"
PROJECT_NAME="articles"
SPIDER_NAME="article_links"
LOG_FILE="/var/log/cisco-crawler.log"

# Function to log with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to trigger spider
trigger_spider() {
    local job_id
    
    log "🚀 Triggering spider: $SPIDER_NAME"
    
    # Schedule the spider
    response=$(curl -s -X POST \
        -d "project=$PROJECT_NAME" \
        -d "spider=$SPIDER_NAME" \
        "$SCRAPYD_URL/schedule.json")
    
    # Extract job ID
    job_id=$(echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('jobid', 'unknown'))")
    
    if [ "$job_id" != "unknown" ]; then
        log "✅ Spider scheduled successfully. Job ID: $job_id"
        return 0
    else
        log "❌ Failed to schedule spider. Response: $response"
        return 1
    fi
}

# Function to check spider status
check_spider_status() {
    local project="$1"
    
    log "📊 Checking spider status..."
    
    response=$(curl -s "$SCRAPYD_URL/listjobs.json?project=$project")
    echo "$response" | python3 -c "
import sys, json
data = json.load(sys.stdin)
running = len(data.get('running', []))
finished = len(data.get('finished', []))
pending = len(data.get('pending', []))
print(f'Running: {running}, Pending: {pending}, Finished: {finished}')
"
}

# Function to deploy project to Scrapyd (if needed)
deploy_project() {
    log "📦 Deploying project to Scrapyd..."
    
    # Assuming scrapyd-client is installed and configured
    if command -v scrapyd-deploy &> /dev/null; then
        scrapyd-deploy production
        log "✅ Project deployed successfully"
    else
        log "⚠️  scrapyd-deploy not found. Please install scrapyd-client"
    fi
}

# Main execution
main() {
    log "🏁 Starting weekly Cisco SMB articles crawl"
    
    # Check if Scrapyd is accessible
    if ! curl -s "$SCRAPYD_URL/daemonstatus.json" > /dev/null; then
        log "❌ Cannot connect to Scrapyd server at $SCRAPYD_URL"
        exit 1
    fi
    
    # Check current status
    check_spider_status "$PROJECT_NAME"
    
    # Trigger the spider
    if trigger_spider; then
        log "🎉 Weekly crawl initiated successfully"
    else
        log "💥 Failed to initiate crawl"
        exit 1
    fi
    
    log "📝 Crawl process started. Check Scrapyd dashboard for progress."
    log "🔗 Dashboard: $SCRAPYD_URL"
}

# Run main function
main "$@"
