# Import libraries
from urllib.request import urljoin
from bs4 import BeautifulSoup
import requests
import os
from urllib.request import urlparse,urlopen


# Set for storing urls with same domain
links_intern = set()
input_url = "https://www.acquisition.gov/far/"
depth = 2
encoding = "utf-8"
lvl=0
current_url_domain = urlparse(input_url).netloc
if(os.path.exists(current_url_domain)==False):
	os.mkdir(current_url_domain)

# Set for storing urls with different domain
links_extern = set()

def remove_newlines(serie):
    serie = serie.replace('\n', ' ')
    serie = serie.replace('\\n', ' ')
    serie = serie.replace('  ', ' ')
    return serie

# Method for crawling a url at next level
def level_crawler(input_url):
	global lvl
	lvl+=1
	temp_urls = set()
	current_url_domain = urlparse(input_url).netloc


	# Creates beautiful soup object to extract html tags
	beautiful_soup_object = BeautifulSoup(
		requests.get(input_url).content, "lxml")

	# Access all anchor tags from input
	# url page and divide them into internal
	# and external categories
	count=0
	for anchor in beautiful_soup_object.findAll("a"):
		href = anchor.attrs.get("href")
		if(href != "" or href != None):
			href = urljoin(input_url, href)
			href_parsed = urlparse(href)
			href = href_parsed.scheme
			href += "://"
			href += href_parsed.netloc
			href += href_parsed.path
			final_parsed_href = urlparse(href)
			is_valid = bool(final_parsed_href.scheme) and bool(
				final_parsed_href.netloc)
			if is_valid:
				# if current_url_domain not in href and href not in links_extern:
				# 	print("Extern - {}".format(href))
				# 	links_extern.add(href)
				if href.startswith(input_url) and href not in links_intern:
					count+=1
					print("Link %d: "%(count) + href)
					links_intern.add(href)
					# Extract text from href and store it in a text_scraped folder.
					page = urlopen(href)
					encoding = "utf-8"
					htmlcontent = (page.read()).decode(encoding)
					soup = BeautifulSoup(htmlcontent,"html.parser")
					with open(current_url_domain+"/depth%d_%d.txt"%(lvl,count),'w',encoding=encoding,errors="ignore") as f:
						f.write(href+"\n"+remove_newlines(soup.get_text()))
					temp_urls.add(href)
	return temp_urls


if(depth == 0):
	print(input_url)

elif(depth == 1):
	lvl=0
	level_crawler(input_url)

else:
	# We have used a BFS approach
	# considering the structure as
	# a tree. It uses a queue based
	# approach to traverse
	# links upto a particular depth.
	
	queue = []
	queue.append(input_url)
	for j in range(depth):
		for count in range(len(queue)):
			url = queue.pop(0)
			urls = level_crawler(url)
			for i in urls:
				queue.append(i)


# Improve scraping

# url = "https://www.acquisition.gov/far/part-12"
# page = urlopen(url)
# encoding = "utf-8"
# htmlcontent = (page.read()).decode(encoding)
# soup = BeautifulSoup(htmlcontent,"html.parser")
# with open("output.txt",'w',encoding=encoding,errors="ignore") as f:
# 	f.write(url+"\n"+remove_newlines(soup.get_text()))