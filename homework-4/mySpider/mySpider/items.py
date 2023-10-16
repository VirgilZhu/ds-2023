# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy

class MyspiderItem(scrapy.Item):
    idx = scrapy.Field()
    name = scrapy.Field()
    price = scrapy.Field()
    author = scrapy.Field()
    publisher = scrapy.Field()
    date = scrapy.Field()
    comments = scrapy.Field()
    intro = scrapy.Field()