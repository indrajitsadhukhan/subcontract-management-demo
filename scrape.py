
URL = ""
def depth0(url):
    try:
        url_text=[]
        page = urlopen(url)
        htmlcontent = (page.read()).decode("latin1")
        soup = BeautifulSoup(htmlcontent,"html.parser")
        # Loop through all the hyperlinks present in the HTML and if we get http at the begining we add them to a list
        for link in soup.find_all('a'):
            h=link.get('href')
            if h and h.startswith('http'):
                url_text.append(link.get('href'))
        return url_text,soup.get_text()
    except:
        print("Failed to do depth0 scraping.")
        return [],""

def check_domain(url):
    domain = urlparse(url).netloc
    if(domain ==""):
        return True
    if(domain != "acquisition.gov"):
        return False
    return True
BASE_URL="https://www.acquisition.gov/"
def depth1(url):
    urls,mainpage_content = depth0(url)
    print("Number of links for Depth=1: ",len(urls))
    depth1_urls=[]
    for c,link in enumerate(urls):
        if(check_domain(link)==False):
            continue
        link = BASE_URL+link
        if(c>upper_limit):
            break
        text_hyperlink_list,hyperlink_content = depth0(link)
        for link_text in text_hyperlink_list:
            depth1_urls.append(link_text)
        print("Link %d: "%(c+1),link)
        with open("text/depth1_%d.txt"%(c+1),'w',encoding="latin1",errors="ignore") as f:
            f.write(hyperlink_content)
    print("Depth1 scraping done!")
    return depth1_urls

def depth2(url):
    depth1_urls= depth1(url)
    print("Number of links for Depth=2:",len(depth1_urls))
    depth2_urls=[]
    for c,link in enumerate(depth1_urls):
        link = BASE_URL+link
        if(check_domain(link)==False):
            continue
        text_hyperlink_list,hyperlink_content = depth0(link)
        for text_link in text_hyperlink_list:
            depth2_urls.append(text_link)
        print("Link %d: "%(c+1),link)
        with open("text/depth2_%d.txt"%(c+1),'w',encoding="latin1",errors="ignore") as f:
            f.write(hyperlink_content)
    print("Depth2 scraping done!")
    return depth2_urls

def crawl(url):
    depth=1
    # Create a directory to store the text files
    if not os.path.exists("text/"):
            os.mkdir("text/")
    if depth==0:
        depth1_urls,mainpage_content = depth0(url)
        with open("text/depth_0.txt",'w',encoding="latin1",errors='ignore') as f:
            f.write(mainpage_content)
        print("Depth0 scraping done!")
    if depth==1:
       depth1(url)
    if depth==2:
        depth2(url)

    text_csv()
