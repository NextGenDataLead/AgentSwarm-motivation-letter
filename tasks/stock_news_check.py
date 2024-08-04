# filename: stock_news_check.py

import requests
from bs4 import BeautifulSoup

def find_news(stock_symbol):
    url = f'https://finance.yahoo.com/quote/{stock_symbol}?p={stock_symbol}&.tsrc=fin-srch'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    news_table = soup.find('ul', {'class': 'My(0) Ov(h) P(0) Wow(bw)'})
    if news_table is not None:
        news_items = news_table.findAll('li')

        print(f"\n===== {stock_symbol} Recent News =====")
        for news in news_items[:5]:
            headline = news.find('a').text
            link = 'https://finance.yahoo.com'+ news.find('a')['href']
            print(f"Headline: {headline}")
            print(f"Link: {link}")
    else:
        print(f"No news found for {stock_symbol}.")
        
find_news('NVDA')
find_news('TSLA')