'''
Usage:
python .\script.py api_key api_secret "keyowrds to searcg" "path\\to\\output\\folder" "category name for subfolder"
'''

import argparse
import os
import flickrapi
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("api_key", help="Enter your API key here", type=str)
parser.add_argument("api_secret", help="Enter your API secret here", type=str)
parser.add_argument("keywords", help="Enter the keywords to search here, comma delimited", type=str)
parser.add_argument("path", help="Enter the output folder path", type=str)
parser.add_argument("category", help="Enter the category of images", type=str)
parser.add_argument("--num_images", help="Number of images")
args = parser.parse_args()

num_images = 200
if (args.num_images):
    num_images = args.num_images

flickr = flickrapi.FlickrAPI(args.api_key, args.api_secret, cache=True)

photos = flickr.walk(
            text=args.keywords,
            tag_mode='any',
            tag=args.keywords,
            per_page=num_images,
            sort='relevance',
            content_type=0
        )

path = args.path + "\\" + args.category + "\\"
if not os.path.isdir(path):
        os.makedirs(path)

for i, photo in enumerate(photos):

    server = photo.get('server')
    id = photo.get('id')
    secret = photo.get('secret')
    
    url = "http://static.flickr.com/{}/{}_{}_b.jpg".format(server, id, secret)
    urllib.request.urlretrieve(url, path + f'{i+1:04d}' + ".jpg")

    if (i >= num_images-1):
        print("Images downloaded successfully to", path)
        break
