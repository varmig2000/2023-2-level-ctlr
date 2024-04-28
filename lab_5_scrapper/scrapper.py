"""
Crawler implementation.
"""
import datetime
import json
# pylint: disable=too-many-arguments, too-many-instance-attributes, unused-import, undefined-variable
import pathlib
import re
from typing import Pattern, Union

import requests
from bs4 import BeautifulSoup

from core_utils.article.article import Article
from core_utils.article.io import to_meta, to_raw
from core_utils.config_dto import ConfigDTO
from core_utils.constants import ASSETS_PATH, CRAWLER_CONFIG_PATH


class IncorrectSeedURLError(Exception):
    """
    If seed URL does not match standard pattern "https?://(www.)?"
    """


class NumberOfArticlesOutOfRangeError(Exception):
    """
    If total number of articles is out of range from 1 to 150.
    """


class IncorrectNumberOfArticlesError(Exception):
    """
    If total number of articles to parse is not integer.
    """


class IncorrectHeadersError(Exception):
    """
    If headers are not in a form of dictionary.
    """


class IncorrectEncodingError(Exception):
    """
    If encoding must be specified as a string.
    """


class IncorrectTimeoutError(Exception):
    """
    If timeout value must be a positive integer less than 60.
    """


class IncorrectVerifyError(Exception):
    """
    If verify certificate value and headless mode must either be True or False.
    """


class Config:
    """
    Class for unpacking and validating configurations.
    """

    def __init__(self, path_to_config: pathlib.Path) -> None:
        """
        Initialize an instance of the Config class.

        Args:
            path_to_config (pathlib.Path): Path to configuration.
        """
        self.path_to_config = path_to_config
        self._validate_config_content()
        self._config = self._extract_config_content()
        self._seed_urls = self._config.seed_urls
        self._num_articles = self._config.total_articles
        self._headers = self._config.headers
        self._encoding = self._config.encoding
        self._timeout = self._config.timeout
        self._should_verify_certificate = self._config.should_verify_certificate

    def _extract_config_content(self) -> ConfigDTO:
        """
        Get config values.

        Returns:
            ConfigDTO: Config values
        """
        with open(file=self.path_to_config) as file:
            my_config = json.load(fp=file)

        result = ConfigDTO(**my_config)

        return result

    def _validate_config_content(self) -> None:
        """
        Ensure configuration parameters are not corrupt.
        """
        with open(file=self.path_to_config) as file:
            my_config = json.load(fp=file)

        if not isinstance(my_config['seed_urls'], list) or \
                not all(re.match("https?://(www.)?", x) for x in my_config['seed_urls']):
            raise IncorrectSeedURLError

        total = my_config['total_articles_to_find_and_parse']

        if not isinstance(total, int) or total <= 0:
            raise IncorrectNumberOfArticlesError

        if not (1 <= total <= 150):
            raise NumberOfArticlesOutOfRangeError

        if not isinstance(my_config['headers'], dict):
            raise IncorrectHeadersError

        if not isinstance(my_config['encoding'], str):
            raise IncorrectEncodingError

        if not isinstance(my_config['timeout'], int) or not (0 < my_config['timeout'] < 60):
            raise IncorrectTimeoutError

        if not isinstance(my_config['should_verify_certificate'], bool):
            raise IncorrectVerifyError

        if not isinstance(my_config['headless_mode'], bool):
            raise IncorrectVerifyError

    def get_seed_urls(self) -> list[str]:
        """
        Retrieve seed urls.

        Returns:
            list[str]: Seed urls
        """
        return self._config.seed_urls

    def get_num_articles(self) -> int:
        """
        Retrieve total number of articles to scrape.

        Returns:
            int: Total number of articles to scrape
        """
        return self._config.total_articles

    def get_headers(self) -> dict[str, str]:
        """
        Retrieve headers to use during requesting.

        Returns:
            dict[str, str]: Headers
        """
        return self._config.headers

    def get_encoding(self) -> str:
        """
        Retrieve encoding to use during parsing.

        Returns:
            str: Encoding
        """
        return self._config.encoding

    def get_timeout(self) -> int:
        """
        Retrieve number of seconds to wait for response.

        Returns:
            int: Number of seconds to wait for response
        """
        return self._config.timeout

    def get_verify_certificate(self) -> bool:
        """
        Retrieve whether to verify certificate.

        Returns:
            bool: Whether to verify certificate or not
        """
        return self._config.should_verify_certificate

    def get_headless_mode(self) -> bool:
        """
        Retrieve whether to use headless mode.

        Returns:
            bool: Whether to use headless mode or not
        """
        return self._config.headless_mode


def make_request(url: str, config: Config) -> requests.models.Response:
    """
    Deliver a response from a request with given configuration.

    Args:
        url (str): Site url
        config (Config): Configuration

    Returns:
        requests.models.Response: A response from a request
    """
    return requests.get(
        url=url,
        timeout=config.get_timeout(),
        headers=config.get_headers(),
        verify=config.get_verify_certificate()
    )


class Crawler:
    """
    Crawler implementation.
    """

    url_pattern: Union[Pattern, str]

    def __init__(self, config: Config) -> None:
        """
        Initialize an instance of the Crawler class.

        Args:
            config (Config): Configuration
        """
        self.config = config
        self.urls = []
        self.url_pattern = 'https://gorvesti.ru/feed'

    def _extract_url(self, article_bs: BeautifulSoup) -> str:
        """
        Find and retrieve url from HTML.

        Args:
            article_bs (bs4.BeautifulSoup): BeautifulSoup instance

        Returns:
            str: Url from HTML
        """
        links = article_bs.find(name='div', class_='feed feed-items')

        for link in links.find_all('a'):
            if link.get('href').endswith('html'):
                url = self.url_pattern + link.get('href')
                if url not in self.urls:
                    self.urls.append(url)
                    return url

    def find_articles(self) -> None:
        """
        Find articles.
        """
        seed_urls = self.get_search_urls()
        nec_len = self.config.get_num_articles()

        while len(self.urls) < nec_len:

            for seed_url in seed_urls:
                response = make_request(seed_url, self.config)
                if not response.ok:
                    continue

                soup = BeautifulSoup(response.text, features='html.parser')

                new_url = self._extract_url(soup)
                if len(self.urls) >= nec_len:
                    break
                self.urls.append(new_url)

                if len(self.urls) >= nec_len:
                    break

    def get_search_urls(self) -> list:
        """
        Get seed_urls param.

        Returns:
            list: seed_urls param
        """
        return self.config.get_seed_urls()


# 10
# 4, 6, 8, 10


class HTMLParser:
    """
    HTMLParser implementation.
    """

    def __init__(self, full_url: str, article_id: int, config: Config) -> None:
        """
        Initialize an instance of the HTMLParser class.

        Args:
            full_url (str): Site url
            article_id (int): Article id
            config (Config): Configuration
        """
        self.full_url = full_url
        self.article_id = article_id
        self.config = config
        self.article = Article(self.full_url, self.article_id)

    def _fill_article_with_text(self, article_soup: BeautifulSoup) -> None:
        """
        Find text of article.

        Args:
            article_soup (bs4.BeautifulSoup): BeautifulSoup instance
        """
        my_div = article_soup.find(name='article', class_='item block')

        all_ps = my_div.find_all('p')

        texts = []
        for p in all_ps:
            texts.append(p.text)

        self.article.text = ''.join(texts)

    def _fill_article_with_meta_information(self, article_soup: BeautifulSoup) -> None:
        """
        Find meta information of article.

        Args:
            article_soup (bs4.BeautifulSoup): BeautifulSoup instance
        """
        title = article_soup.find(name='h1', itemprop="headline")
        if title:
            self.article.title = title.text

        authors = article_soup.find(name='div', class_='item-field item-ath')
        for author in authors:
            article_soup.find("#text")
            if author:
                self.article.author.append(author.text)
            else:
                self.article.author.append('NOT FOUND')

    def unify_date_format(self, date_str: str) -> datetime.datetime:
        """
        Unify date format.

        Args:
            date_str (str): Date in text format

        Returns:
            datetime.datetime: Datetime object
        """

    def parse(self) -> Union[Article, bool, list]:
        """
        Parse each article.

        Returns:
            Union[Article, bool, list]: Article instance
        """
        response = make_request(self.full_url, self.config)

        if response.ok:
            article_bs = BeautifulSoup(response.text, features='html.parser')
            self._fill_article_with_text(article_bs)
            self._fill_article_with_meta_information(article_bs)

        return self.article


def prepare_environment(base_path: Union[pathlib.Path, str]) -> None:
    """
    Create ASSETS_PATH folder if no created and remove existing folder.

    Args:
        base_path (Union[pathlib.Path, str]): Path where articles stores
    """
    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True)
    else:
        for file in base_path.iterdir():
            file.unlink()


def main() -> None:
    """
    Entrypoint for scrapper module.
    """
    config = Config(path_to_config=CRAWLER_CONFIG_PATH)
    prepare_environment(base_path=ASSETS_PATH)

    crawler = Crawler(config=config)
    crawler.find_articles()
    urls = crawler.urls
    for index, url in enumerate(urls):
        html_parser = HTMLParser(full_url=url, article_id=index+1, config=config)
        article = html_parser.parse()
        to_raw(article)
        to_meta(article)


if __name__ == "__main__":
    main()
