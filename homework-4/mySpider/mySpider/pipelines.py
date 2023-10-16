import json
import mysql.connector

class MyspiderPipeline(object):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="10244ro",
        database="ds-homework-4"
    )
    mycursor = mydb.cursor()

    def __init__(self):
        self.file = open('booksInfo.json', 'w', encoding='utf-8-sig')
        self.file.seek(0)
        self.file.truncate()

    def process_item(self, item, spider):
        json_data = json.dumps(item, default=str, ensure_ascii=False) + '\n'
        self.file.write(json_data)
        sql = "INSERT INTO bookinfo (idx, name, price, author, publisher, date, comments, intro) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        val = (item['index'], item['name'], item['price'], item['author'], item['publisher'], item['date'], item['comments'], item['intro'])
        self.mycursor.execute(sql, val)
        return item

    def __del__(self):
        self.file.close()
        self.mydb.commit()
        self.mycursor.close()
        self.mydb.close()