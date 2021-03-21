import json
import os
import re
import requests

from bs4 import BeautifulSoup


SESSIONS_FOLDER = 'sessions/'


def scrape_list_page(page_content):
    '''
    Scrapes the links to individual session pages from a page containing a list of sessions
    '''

    soup = BeautifulSoup(page_content, 'html.parser')

    all_links = soup.find_all('a')

    session_link_pattern = r'https://otvoreniparlament.rs/transkript/[0-9]{4}'
    session_page_links = list(filter(
        lambda l: re.fullmatch(session_link_pattern, l.get('href')) != None,
        all_links
    ))

    return [link.get('href') for link in session_page_links]


def gather_session_urls():
    '''
    Loops through all of the pages containing links to all of the session and aggregates the links
    '''

    print('Gathering urls for each session page')

    list_page_base_url = 'https://otvoreniparlament.rs/transkript?page={}'
    session_urls = []

    for list_page_id in range(1, 199):

        url = list_page_base_url.format(list_page_id)
        print('Scraping the list page url: {}\r'.format(url), end='')

        try:
            list_page = requests.get(url)
            page_session_urls = scrape_list_page(list_page.content)
            session_urls += page_session_urls

        except requests.exceptions.ConnectionError:
            print('\nCould not load the list page url')
            exit(1)
    print()

    return session_urls


def scrape_session_date(page_soup):
    '''
    Scrapes the date of the session from the first page of a session
    '''

    div_lvl_1 = page_soup.body('div', recursive=False)
    div_lvl_2 = div_lvl_1[1]('div', recursive=False)
    div_lvl_3 = div_lvl_2[0]('div', recursive=False)

    # Get date
    meta_data_div = div_lvl_3[0]
    return meta_data_div('p', string=re.compile(r'\d\d\.\d\d\.\d\d\d\d'))[0].string


def text_to_ascii(text):
    '''
    Replaces non ASCII characters in the parsed text with their ASCII equivalents
    '''

    return text.replace('đ', 'dj').replace('Đ', 'Dj').\
        replace('ž', 'z').replace('Ž', 'Z').\
        replace('ć', 'c').replace('Ć', 'C').\
        replace('č', 'c').replace('Č', 'C').\
        replace('š', 's').replace('Š', 'S').\
        replace('–', '-').\
        replace('…', '.').\
        replace('„', '\'').replace('“', '\'').replace('"', '\'')


def scrape_speeches(page_soup):
    '''
    Scrapes all of the speeches on a single page of a session
    '''

    speeches_data = []

    # Get to the page section with speeches
    div_lvl_1 = page_soup.body('div', recursive=False)
    div_lvl_2 = div_lvl_1[1]('div', recursive=False)
    div_lvl_3 = div_lvl_2[0]('div', recursive=False)

    # Get to the list of secions for each speech
    speeches_data_div = div_lvl_3[1]
    speeches_div_lvl_1 = speeches_data_div('div', recursive=False)
    speeches_div_lvl_2 = speeches_div_lvl_1[0]('div', recursive=False)
    speeches_div_lvl_3 = speeches_div_lvl_2[0]('div', class_=re.compile(r'tag'), recursive=False)

    for speech_div in speeches_div_lvl_3:

        # Get the div whit the speech data
        speech_div_lvl_1 = speech_div('div', class_='media-body')[0]

        # Remove all unnecessary blank characters
        speech = speech_div_lvl_1('div', id=re.compile(r'transkript.*content'))[0].text
        speech = re.sub(r'\s', ' ', speech)
        speech = re.sub(r' +', ' ', speech)
        speech = speech.strip()

        # Record the speaker and the speech text
        speech_data = {
            'speaker': text_to_ascii(speech_div_lvl_1('a')[0].string),
            'speech': text_to_ascii(speech)
        }

        speeches_data.append(speech_data)

    return speeches_data


def scrape_session_pages(base_session_url):
    '''
    Loops through all pages containing data on a singe session and aggregates gathered data
    '''

    session_data = {
        'url': base_session_url,
        'date': '',
        'speeches': []
    }

    session_page_id = 1
    all_pages_covered = False
    while not all_pages_covered:

        url = '{}?page={}'.format(base_session_url, session_page_id)
        print('Scraping the session page {}: {}\r'.format(session_page_id, url), end='')

        try:
            # Retrive the page
            session_page = requests.get(url)

            page_soup = BeautifulSoup(session_page.content, 'html.parser')

            # For the first page of a session, get the date
            if session_page_id == 1:
                session_data['date'] = scrape_session_date(page_soup)

            # Scrape speeches from each page of a session
            speeches = scrape_speeches(page_soup)
            session_data['speeches'] += speeches

            session_page_id += 1

            if len(speeches) == 0:
                all_pages_covered = True

        except requests.exceptions.ConnectionError:
            print('\nCould not load the session page url')
            exit(1)
    print()

    print('Session speeches: {}'.format(len(session_data['speeches'])))

    return session_data


def gether_sessions_data(session_urls):
    '''
    Loops through all gathered session urls and aggregates gathered data for each session
    '''

    print('Gathering data for each session')

    sessions_data = []
    for i, url in enumerate(session_urls):

        print('Scraping the session page {}/{}: {}\r'.format(i, len(session_urls), url), end='')

        session_data = scrape_session_pages(url)
        sessions_data.append(session_data)
    print()

    return sessions_data


def main():
    '''
    Data scraping steps:
    - Gether URLs for each session
    - Gether speech data for each session
    - Save all gathered speech data to json file
    '''

    session_urls = gather_session_urls()
    print('Total num of session page links {}'.format(len(session_urls)))

    sessions_data = gether_sessions_data(session_urls)
    print('Number of sessions {}'.format(len(sessions_data)))

    if not os.path.exists(SESSIONS_FOLDER):
        os.makedirs(SESSIONS_FOLDER)

    with open(SESSIONS_FOLDER + 'sessions_data.json', 'w') as out_file:
        json.dump(sessions_data, out_file)


if __name__ == '__main__':
    main()
