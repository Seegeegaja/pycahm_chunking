import requests
from bs4 import BeautifulSoup

def get_urls():
    url = "https://zzzmini.github.io/sample.html"

    response = requests.get(url)
    # BeautifulSoup4로 Html 파싱후 img 태그만 필터링
    soup = BeautifulSoup(response.text,"html.parser")
    img_list = soup.find_all('img')

    url_list = [tag.get('src') for tag in img_list if tag.get('src')]
    return url_list
if __name__ == '__main__':
    print(get_urls())