#!/bin/bash

# Scrapyd Management Script for Articles Project
# Usage: ./scrapyd_manager.sh [start|stop|status|deploy|list|schedule <spider>]

SCRAPYD_HOST="http://localhost:6800"
PROJECT_NAME="articles"
VENV_PATH="/home/jclemens/projects/scrapy-articles/env"

case "$1" in
    start)
        echo "Starting Scrapyd daemon..."
        cd /home/jclemens/projects/scrapy-articles
        $VENV_PATH/bin/scrapyd &
        echo "Scrapyd started. Web interface available at $SCRAPYD_HOST"
        ;;
    
    stop)
        echo "Stopping Scrapyd daemon..."
        pkill -f scrapyd
        echo "Scrapyd stopped."
        ;;
    
    status)
        echo "Checking Scrapyd daemon status..."
        curl -s $SCRAPYD_HOST/daemonstatus.json | python3 -m json.tool
        ;;
    
    deploy)
        echo "Deploying project to local Scrapyd..."
        cd /home/jclemens/projects/scrapy-articles
        $VENV_PATH/bin/scrapyd-deploy local -p $PROJECT_NAME
        ;;
    
    list)
        echo "Available projects:"
        curl -s $SCRAPYD_HOST/listprojects.json | python3 -m json.tool
        echo -e "\nSpiders in $PROJECT_NAME project:"
        curl -s "$SCRAPYD_HOST/listspiders.json?project=$PROJECT_NAME" | python3 -m json.tool
        ;;
    
    jobs)
        echo "Current jobs for $PROJECT_NAME project:"
        curl -s "$SCRAPYD_HOST/listjobs.json?project=$PROJECT_NAME" | python3 -m json.tool
        ;;
    
    schedule)
        if [ -z "$2" ]; then
            echo "Usage: $0 schedule <spider_name>"
            echo "Available spiders:"
            curl -s "$SCRAPYD_HOST/listspiders.json?project=$PROJECT_NAME" | python3 -c "import sys, json; data=json.load(sys.stdin); print('\n'.join(data['spiders']))"
            exit 1
        fi
        echo "Scheduling spider: $2"
        curl -s $SCRAPYD_HOST/schedule.json -d project=$PROJECT_NAME -d spider=$2 | python3 -m json.tool
        ;;
    
    web)
        echo "Opening Scrapyd web interface..."
        echo "Visit: $SCRAPYD_HOST"
        ;;
    
    *)
        echo "Scrapyd Management Script"
        echo "Usage: $0 {start|stop|status|deploy|list|jobs|schedule <spider>|web}"
        echo ""
        echo "Commands:"
        echo "  start    - Start the Scrapyd daemon"
        echo "  stop     - Stop the Scrapyd daemon"
        echo "  status   - Check daemon status"
        echo "  deploy   - Deploy project to Scrapyd"
        echo "  list     - List projects and spiders"
        echo "  jobs     - Show current jobs"
        echo "  schedule - Schedule a spider to run"
        echo "  web      - Show web interface URL"
        exit 1
        ;;
esac
