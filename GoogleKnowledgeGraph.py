import json
import urllib3
import requests

#Retrieving api service key and service url
config = json.load(open('config/GoogleKnowledgeGraphSettings.json'))

#Setting up the connection
http = urllib3.PoolManager()
api_key = config['api_key']
service_url = config['service_url']

#The query and its parameters
#query = 'Springfield+Armory'
query = 'browning+shooting'

params = {
    'query': query,
    'limit': 15,
    'indent': True,
    'key': api_key,
}

url = service_url + '?' + urllib3.request.urlencode(params)
print(url)
#response = json.loads(str(http.request('GET', url).read())
response = (requests.get(url).text)
#result = (response).decode('utf-8')
print(response)
jresponse = json.loads(response)
for element in jresponse['itemListElement']:
    print (element['result']['name'] + ' (' + str(element['resultScore']) + ')')


