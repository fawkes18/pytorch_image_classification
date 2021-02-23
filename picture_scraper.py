import requests
import json
import os

os.chdir('C://Users/Marco/Desktop/')


def get_urls(query):
    image_urls = []
    for i in range(65, 204):
        url = f'https://unsplash.com/napi/search/photos?query={query}&per_page=20&page={i}&xp=feedback-loop-v2%3Aexperiment'
        r = requests.get(url)
        json_data = json.loads(r.content)
        page_urls = [img['links']['download'] for img in json_data['results']]
        image_urls.append(page_urls)
    return image_urls


def download_images(dir, query):
    image_urls = get_urls(query)
    final_urls = [item for sublist in image_urls for item in sublist]
    for i, url in enumerate(final_urls):
        img_name = f'{query}_{i}.jpg'
        with open(dir + img_name, 'wb') as f:
            f.write(requests.request('GET', url).content)



#download_images('Women2/', 'woman')
#download_images('Men2/', 'man')