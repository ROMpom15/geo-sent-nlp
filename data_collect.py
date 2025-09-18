# https://mediastack.com/documentation

# // Live News Data

# http://api.mediastack.com/v1/news
#     ? access_key = YOUR_ACCESS_KEY
    
# // optional parameters: 

#     & sources = cnn,bbc
#     & categories = business,sports
#     & countries = us,au
#     & languages = en,-de
#     & keywords = virus,-corona
#     & sort = published_desc
#     & offset = 0
#     & limit = 100


import http.client, urllib.parse

conn = http.client.HTTPConnection('api.mediastack.com')

params = urllib.parse.urlencode({
    'access_key': 'ACCESS_KEY',
    'categories': '-general,-sports',
    'sort': 'published_desc',
    'limit': 10,
    })

conn.request('GET', '/v1/news?{}'.format(params))

res = conn.getresponse()
data = res.read()

print(data.decode('utf-8'))