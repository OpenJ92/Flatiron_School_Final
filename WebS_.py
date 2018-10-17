import sc2reader
import bs4
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import requests
import pandas as pd
import os
from os import listdir
import time
from os.path import isfile, join
import webbrowser
from functools import reduce
import numpy as np

# Scrape replay files from:
#           A) https://gggreplays.com/matches/160000
#               Bronze       https://gggreplays.com/matches#?average_league=0&game_type=1v1&page=166 ~1660 total
#               Silver       https://gggreplays.com/matches#?average_league=1&game_type=1v1&page=1361 ~13610 total
#               Gold         https://gggreplays.com/matches#?average_league=2&game_type=1v1&page=2413 ~24130 total
#               Platinum     https://gggreplays.com/matches#?average_league=3&game_type=1v1&page=3427 ~34270 total
#               Diamond      https://gggreplays.com/matches#?average_league=4&game_type=1v1&page=4083 ~40830 total
#               Master       https://gggreplays.com/matches#?average_league=5&game_type=1v1&page=822 ~8220 total
#               Grand-Master https://gggreplays.com/matches#?average_league=6&game_type=1v1&page=10 ~100 total

p_B = 166
S_p = 1361
G_p = 2413
P_p = 3427
D_p = 4083
M_p = 822
GM_p = 10

link = 'https://gggreplays.com'
_link = 'https://gggreplays.com/matches#?average_league='
_link_ = '&game_type=1v1&page='
link_ = [_link + str(i) + _link_ for i in range(0, 7)]
page_num = [166, 1361, 2416, 3427, 4083, 822, 10]
page_name = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond', 'Master', 'Grand_Master']

#### Referance: https://stackoverflow.com/questions/46113924/beautifulsoup-does-not-return-all-elements-on-page

def scrape_Category(category, start = 1, mode = 'w'):
    Q = []
    driver = webdriver.Chrome()
    for pagenum in range(start, page_num[category-1] + 1):
        driver.get(link_[category-1] + str(pagenum))
        time.sleep(2.5)
        print(link_[category-1] + str(pagenum))
        r = driver.page_source
        soup = BeautifulSoup(r, "html.parser")
        W = [(link + td.a['href'], td.a.text) for td in soup.find_all('td', class_ = 'map nowrap')]
        Q += W

    driver.quit()
    with open('./sc2_Replays/' + page_name[category - 1] + '/' + page_name[category - 1] + '_links.txt', mode) as f:
        for item in Q:
            f.write(item[0] + ', ' + item[1] + '\n')
    return Q

def scrape_Replay(league, numSamples, start = 0, stop = None, getReplay = False):
    Q = pd.read_csv('Sc2_Replays/'+ league + '/' + league + '_links.txt')
    if numSamples > 0:
        Q = np.array(Q['link'].sample(numSamples))
    else:
        Q = np.array(Q['link'])

    if stop == None:
        stop = len(Q)

    chromeOptions = webdriver.ChromeOptions()
    prefs = {'download.default_directory' : '/Users/flatironschool/Desktop/PySC2/sc2reader/sc2_Replays/' + league + '/' + league +'_replays'}
    chromeOptions.add_experimental_option("prefs",prefs)
    driver = webdriver.Chrome(chrome_options=chromeOptions)
    counter = start
    for link in Q[start:stop]:
        #print(link)
        driver.get(link)
        source = driver.page_source
        soup = BeautifulSoup(source, "html.parser")
        link_ = soup.find_all('a', class_ = 'dlbutton button2 button2-lime')
        try:
            download_link = 'https://gggreplays.com/' + link_[0]['href']
            if getReplay:
                time.sleep(1.5)
                driver.get(download_link)
        except:
            pass
        print(counter)
        counter += 1

    driver.quit()

    return Q

#scrape_Replay('Bronze', numSamples = 0, start = 0, getReplay = True)
#scrape_Replay('Silver', numSamples = 0, start = 0, getReplay = True)
#scrape_Replay('Gold', numSamples = 0, start = 0, getReplay = True)
#scrape_Replay('Platinum', numSamples = 0, start = 0, getReplay = True)
scrape_Replay('Diamond', numSamples = 0, start = 23096, getReplay = True)
scrape_Replay('Master', numSamples = 0, start = 0, getReplay = True)
scrape_Replay('Grand_Master', numSamples = 0, start = 0, getReplay = True)
