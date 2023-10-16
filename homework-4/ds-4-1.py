import mysql.connector
from lxml import etree
import requests
import csv

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="10244ro",
    database="ds-homework-4"
)
mycursor = mydb.cursor()

rank = 1

with open('films.csv', 'w', newline='', encoding='utf-8-sig') as fp:
    fp.seek(0)
    fp.truncate()
    writer = csv.writer(fp)
    writer.writerow(('电影名', '主演', '上映日期', '评分', '影评数', '热门影评'))
    urls = ['https://movie.douban.com/top250?start={}&filter='.format(str(i)) for i in range (0, 251, 25)]
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.67'}

    for url in urls:
        html = requests.get(url, headers=headers)
        selector = etree.HTML(html.text)
        infos = selector.xpath("//ol[@class='grid_view']/li")
        for info in infos:
            name = info.xpath(".//div[@class='hd']//a//span[1]/text()")[0]
            actor = info.xpath(".//div[@class='bd']//p/text()")[0]
            actor = actor.split(': ')
            if(len(actor) == 2):
                actor = "未知"
            elif(' ' not in actor[2]):
                actor = actor[2]
            else:
                actor = actor[2].split(' ')[0]
            date = info.xpath(".//div[@class='bd']//p/br/following-sibling::text()")[0]
            date = date.split('/')[0]
            star = info.xpath(".//div[@class='bd']//div[@class='star']//span[@class='rating_num' and @property='v:average']/text()")[0]
            recommendation = info.xpath(".//div[@class='bd']//div[@class='star']//span[4]/text()")[0]
            recommendation = recommendation.split('评')[0]
            introduction = ''
            try:
                introduction = info.xpath(".//div[@class='bd']//p[@class='quote']//span[@class='inq']/text()")[0]
            except:
                pass
            writer.writerow((name, actor, date, star, recommendation, introduction))

            sql = "INSERT INTO movieinfo (movie_rank, name, actor, date, star, comments, hot) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            val = (rank, name, actor, date, star, recommendation, introduction)
            mycursor.execute(sql, val)
            rank += 1

mydb.commit()
mycursor.close()
mydb.close()