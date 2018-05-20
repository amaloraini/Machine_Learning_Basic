import urllib3
import sys
import re
import os
import sqlite3


#to create a DB 
def connect_DB(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except:
        print('Error!')
        return
#to create a table in the DB
def create_table(conn, create_table_sql):
        try:
            c = conn.cursor()
            c.execute(create_table_sql)
        except Exception as e:
            print(e)

#insert tuple in the database
def insert_tuple(conn, player, country):
    sql = "INSERT INTO KB(player, country) VALUES('"+ player + "', '"+country +"')"
    conn.execute(sql)
    conn.commit()

conn = connect_DB('results/KnowlwdgeBase.sqlite3')
#create_table(conn, 'CREATE TABLE KB (player TEXT, country TEXT)')
#insert_tuple(conn, 'Abdul', 'Bayern')
#conn.commit()
cur = conn.cursor()
cur.execute("select sqlite_version();")
data = cur.fetchone()
print(data)
conn.close()


