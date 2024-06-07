# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import pymongo.errors
import scrapy.crawler
from scrapy.exceptions import DropItem
import scrapy
import pymongo
import dotenv
import os

dotenv.load_dotenv(dotenv.find_dotenv())


class ArticleLinkFilter:
    def __init__(self, feed_options) -> None:
        self.feed_options = feed_options
        print(f"FEED OPTIONS: {self.feed_options}")
        self.seen = {}

    def accepts(self, item):
        item = ItemAdapter(item)
        family_name = item["family"]
        self.seen.setdefault(family_name, set())
        if item["url"] in self.seen[family_name]:
            return False
        return True


class DuplicatesPipeline:
    def __init__(self, feed_options) -> None:
        self.ids_seen = {}
        self.feed_options = feed_options
        print(f"FEED OPTIONS: {self.feed_options}")

    @classmethod
    def from_crawler(cls, crawler: scrapy.crawler.Crawler):
        feed_options = crawler.settings.getdict("FEED_OPTIONS")
        return cls(feed_options)

    def process_item(self, item, spider: scrapy.Spider):
        adapter = ItemAdapter(item)
        family_name = adapter["family"]
        self.ids_seen.setdefault(f"{family_name}", [])
        if adapter["url"] in self.ids_seen[f"{family_name}"]:
            raise DropItem(f"Duplicate item found: {item!r}")
        else:
            self.ids_seen[f"{family_name}"].append(adapter["url"])
            return item


class MongoPipeline:
    collection_name = "article_links"

    def __init__(self, mongo_uri: str, mongo_db: str) -> None:
        self.mongo_uri = mongo_uri.replace(
            "<username>", os.environ.get("MONGODB_APP_USER")
        ).replace("<password>", os.environ.get("MONGODB_APP_PASS"))
        self.mongo_db = mongo_db

    def create_index(self):
        self.db[self.collection_name].create_index(
            [("url", pymongo.ASCENDING), ("family", pymongo.ASCENDING)],
            unique=True,
            name="url_and_family_unique_index",
        )

    @classmethod
    def from_crawler(cls, crawler: scrapy.crawler.Crawler):
        return cls(
            mongo_uri=crawler.settings.get("MONGO_URI"),
            mongo_db=crawler.settings.get("MONGO_DATABASE", "smb_documents"),
        )

    def open_spider(self, spider: scrapy.Spider):
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client[self.mongo_db]
        self.create_index()

    def close_spider(self, spider: scrapy.Spider):
        self.client.close()

    def process_item(self, item, spider: scrapy.Spider):

        self.db[self.collection_name].insert_one(ItemAdapter(item).asdict())

        # raise DropItem(f"Duplicate item found: {item!r}")
        return item
