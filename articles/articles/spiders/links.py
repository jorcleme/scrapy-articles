import scrapy
import json
import scrapy.signals
import subprocess
import scrapy.http
from scrapy import Spider
from scrapy.signalmanager import dispatcher
from scrapy.http.response.html import HtmlResponse
from scrapy.http.response.text import TextResponse
from urllib3.util import parse_url


class ArticleLinks(Spider):
    name = "article_links"

    start_urls = [
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/CBS220/jcr:content/Title/full/Full/widenarrow_5d4b_copy/WN-Wide-1/drawertabscontainer_/responsive-drawertab-parsys-container/drawertab_6b89/responsive-drawertab-parsys-forTabContent/list_dynamic_3fba.feed.json",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/CBS250.html",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/CBS350/jcr:content/Grid/widenarrow_5d4b/WN-Wide-1/drawertabscontainer_/responsive-drawertab-parsys-container/drawertab_6b89/responsive-drawertab-parsys-forTabContent/list_dynamic_3fba.feed.json",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/Catalyst-1200.html",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/Catalyst-1300.html",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/switches-350-family.html",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/switches-350x-family/jcr:content/Grid/widenarrow_5d4b/WN-Wide-1/drawertabscontainer_/responsive-drawertab-parsys-container/drawertab_6b89/responsive-drawertab-parsys-forTabContent/list_dynamic_1675.feed.json",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/switches-550x-family/jcr:content/Grid/widenarrow_5d4b/WN-Wide-1/drawertabscontainer_/responsive-drawertab-parsys-container/drawertab_6b89/responsive-drawertab-parsys-forTabContent/list_dynamic_76b1.feed.json",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/routers-100-family.html",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/routers-100-family/jcr:content/Grid/widenarrow_5d4b/WN-Wide-1/drawertabscontainer_/responsive-drawertab-parsys-container/drawertab_6b89/responsive-drawertab-parsys-forTabContent/list_dynamic_8602.feed.json",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/routers-320-family.html",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/routers-320-family/jcr:content/Grid/widenarrow_5d4b/WN-Wide-1/drawertabscontainer_/responsive-drawertab-parsys-container/drawertab_6b89/responsive-drawertab-parsys-forTabContent/list_dynamic_9dbf.feed.json",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/routers-340-family.html",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/routers-340-family/jcr:content/Grid/widenarrow_5d4b/WN-Wide-1/drawertabscontainer_/responsive-drawertab-parsys-container/drawertab_6b89/responsive-drawertab-parsys-forTabContent/list_dynamic_3fba.feed.json",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/routers-160-family.html",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/routers-160-family/jcr:content/Grid/widenarrow_5d4b/WN-Wide-1/drawertabscontainer_/responsive-drawertab-parsys-container/drawertab_6b89/responsive-drawertab-parsys-forTabContent/list_dynamic_9953.feed.json",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/routers-260-family.html",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/routers-260-family/jcr:content/Grid/widenarrow_5d4b/WN-Wide-1/drawertabscontainer_/responsive-drawertab-parsys-container/drawertab_6b89/responsive-drawertab-parsys-forTabContent/list_dynamic_579e.feed.json",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/wireless-mesh-100-200-series.html?cachemode=refresh",
        "https://www.cisco.com/c/en/us/support/smb/product-support/small-business/wireless-mesh-100-AX-series.html?cachemode=refresh",
    ]

    def __init__(self, *args, **kwargs):
        dispatcher.connect(self.spider_closed, scrapy.signals.spider_closed)
        super().__init__(*args, **kwargs)

    def parse(self, response: scrapy.http.Response):

        if isinstance(response, HtmlResponse):
            family_name = parse_url(response.url).path.split("/")[-1].split(".")[0]
            print(f"Family: {family_name}")
            yield response.follow(
                response.url,
                callback=self.parse_html,
                cb_kwargs={"family": family_name},
            )
        elif isinstance(response, TextResponse):
            data = json.loads(response.text)
            family_name = parse_url(data["link"]).path.split("/")[-1].split(".")[0]
            for item in data["items"]:
                yield {
                    "url": item["link"],
                    "family": family_name,
                }
            if "link" in data:
                family_name = parse_url(data["link"]).path.split("/")[-1].split(".")[0]
                yield response.follow(
                    data["link"],
                    callback=self.parse_html,
                    cb_kwargs={"family": family_name},
                )

    def parse_html(self, response: HtmlResponse, family: str):
        hrefs = response.xpath(
            "//div[@class='dm0 dmc-list-dynamic']/ul/li/p/a/@href"
        ).getall()
        links = [f"http://www.cisco.com{link}" for link in hrefs]
        for link in links:
            yield {
                "url": link,
                "family": family,
            }

    def spider_closed(self, spider: scrapy.Spider):
        subprocess.call(["python", "-m", "services.articles"])
        subprocess.call(
            [
                "C:\\Users\\jorcleme\\Projects\\scrapy-articles\\env\\Scripts\\python.exe",
                "-m",
                "services.articles",
            ]
        )
