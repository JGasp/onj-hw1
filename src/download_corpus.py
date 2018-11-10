from selenium import webdriver
from unidecode import unidecode


"""
You need to download packages 
- selenium (used to render and query Javascript pages), 
- unidecode (filters unicode characters) 

This script uses Chrome as web driver
Chrome WebDriver [https://sites.google.com/a/chromium.org/chromedriver/downloads]

"""


def download_corpus(pages=1):

    browser = webdriver.Chrome()
    browser.maximize_window()

    for current_page in range(0, pages):

        url = 'https://www.extremetech.com/page/' + str(current_page + 1)
        browser.get(url)

        news = browser.find_element_by_css_selector('.story-river')\
            .find_elements_by_tag_name('li')

        links = []

        for n in news:

            if 'ad' not in n.get_attribute('class'):  # Remove ads
                link = n.find_element_by_tag_name('a').get_attribute('href')
                if 'et-deals' not in link:  # Remove ET-Deals ads
                    links.append(link)

        for l in links:
            browser.get(l)

            post = browser.find_element_by_css_selector('.post-container')

            post_title = post.find_element_by_css_selector('.post-title')

            post_title_text = post_title.find_element_by_tag_name('h1').text

            post_author_and_date = post_title.find_element_by_css_selector('.by').text

            post_content = post.find_element_by_css_selector('.post-content')

            post_content_paragraphs = post_content.find_element_by_id('intelliTXT')\
                .find_elements_by_tag_name('p')

            paragraphs = []
            for p in post_content_paragraphs[:-1]:
                text = p.text.strip()
                if len(text) > 0:
                    if 'Image credit' not in text and 'Now Read:' not in text:
                        paragraphs.append(text)

            content = unidecode(post_title_text + '\n' + '\n'.join(paragraphs))

            file_name = unidecode(post_author_and_date)
            file_path = 'res/' + file_name.replace(' ', '_').replace(':', '-') + '.txt'

            with open(file_path, 'w') as f:
                f.write(content)


if __name__ == "__main__":
    download_corpus(1000)

