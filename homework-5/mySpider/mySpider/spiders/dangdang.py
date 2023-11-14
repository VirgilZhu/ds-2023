import scrapy

class DangdangSpider(scrapy.Spider):
    name = "dangdang"
    allowed_domains = ["dangdang.com"]
    start_urls = ["https://category.dangdang.com/cp01.54.00.00.00.00.html"]
    book_idx = 1

    def start_requests(self):
        for page in range(1, 101):
            url = f'https://category.dangdang.com/pg{str(page)}-cp01.54.00.00.00.00.html'
            yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
        book_list = response.xpath("/html/body/div[2]/div/div[3]/div[1]/div[1]/div[2]/div/ul/li")
        for book in book_list:
            temp = {}
            temp['index'] = self.book_idx
            self.book_idx += 1
            temp['name'] = book.xpath(".//a/@title").get()
            temp['price'] = book.xpath(".//p[3]/span[1]/text()")[0].get()
            try:
                temp['author'] = book.xpath(".//p[5]/span[1]/a[1]/text()")[0].get()
            except:
                temp['author'] = ""
            try:
                temp['publisher'] = book.xpath(".//p[5]/span[3]/a/text()")[0].get()
            except:
                temp['publisher'] = ""
            try:
                temp['date'] = book.xpath(".//p[5]/span[2]/text()")[0].get()[2:]
            except:
                temp['date'] = ""
            try:
                temp['comments'] = book.xpath(".//p[4]/a/text()")[0].get()
            except:
                temp['comments'] = ""
            try:
                temp['intro'] = book.xpath(".//p[2]/text()")[0].get()
            except:
                temp['intro'] = ""
            yield temp
