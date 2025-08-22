# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import scrapy
import dotenv
import logging
import scrapy.crawler
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem

dotenv.load_dotenv(dotenv.find_dotenv())


class ArticleLinkFilter:
    def __init__(self, feed_options) -> None:
        self.feed_options = feed_options
        print(f"FEED OPTIONS: {self.feed_options}")
        self.seen = {}

    def accepts(self, item: scrapy.Item) -> bool:
        item = ItemAdapter(item)
        family_name: str = item["family"]
        self.seen.setdefault(family_name, [])
        if item["url"] in self.seen[family_name]:
            print(f"Duplicate item found: {item!r}")
            return False
        self.seen[family_name].append(item["url"])
        return True


class DuplicatesPipeline:
    def __init__(self, feed_options) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.info("Initializing DuplicatesPipeline")
        self.logger.info(f"Feed options: {feed_options}")
        self.ids_seen = {}
        self.feed_options = feed_options

    @classmethod
    def from_crawler(cls, crawler: scrapy.crawler.Crawler):
        feed_options = crawler.settings.getdict("FEEDS")
        return cls(feed_options)

    def process_item(self, item: scrapy.Item, spider: scrapy.Spider):
        adapter = ItemAdapter(item)
        family_name = adapter["family"]
        self.ids_seen.setdefault(f"{family_name}", [])
        if adapter["url"] in self.ids_seen[f"{family_name}"]:
            raise DropItem(f"Duplicate item found: {item!r}")
        else:
            self.ids_seen[f"{family_name}"].append(adapter["url"])
            return item
