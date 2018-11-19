from datetime import datetime
import time


day="2018-02-12 17:40:51"
week = time.strptime(day,"%Y-%m-%d %H:%M:%S")
print(week)
print(week[6])