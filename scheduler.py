import pytz
import requests
import subprocess
from apscheduler.schedulers.twisted import TwistedScheduler
from twisted.internet import reactor


def send_request():
    requests.post(
        "https://scrapy-articles-965244f91aa6.herokuapp.com/schedule.json",
        data={"project": "articles", "spider": "article_links"},
    )


if __name__ == "__main__":
    subprocess.run("scrapyd-deploy", shell=True, universal_newlines=True)
    scheduler = TwistedScheduler(timezone=pytz.timezone("US/Eastern"))
    scheduler.add_job(send_request, "cron", day_of_week="mon", hour=3, minute=0)
    scheduler.start()
    reactor.run()
