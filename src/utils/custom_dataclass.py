"""aaa"""
import dataclasses

import pandas as pd
from llama_index.core.bridge.pydantic import BaseModel, Field


def company_validator(company: str) -> str:
    """
    Validates the company parameter.

    Parameters
    ----------
    company : str
        The name of the company to search for.

    Returns
    -------
    str
        The validated company name.
    """
    if not isinstance(company, str) or not company.strip():
        raise ValueError("company must be a non-empty string.")
    return company.strip()


def from_date_validator(from_date: str) -> str:
    """
    Validates the from_date parameter.

    Parameters
    ----------
    from_date : str
        The start date for the search in 'YYYY-MM-DD' format.

    Returns
    -------
    str
        The validated from_date value.
    """
    if not isinstance(from_date, str) or not from_date.strip():
        raise ValueError("from_date must be a non-empty string")

    # Check if the date format is correct
    try:
        pd.to_datetime(from_date, format='%Y-%m-%d', errors='raise')
    except ValueError as e:
        raise ValueError("from_date must be in 'YYYY-MM-DD' format") from e
    return from_date.strip()


def size_page_validator(size_page: int) -> int:
    """
    Validates the size_page parameter.

    Parameters
    ----------
    size_page : int
        The number of results to return per page.

    Returns
    -------
    int
        The validated size_page value.
    """
    if not isinstance(size_page, int) or size_page <= 0:
        raise ValueError("size_page must be a positive integer.")
    return size_page


def n_pages_validator(n_pages: int) -> int:
    """
    Validates the n_pages parameter.

    Parameters
    ----------
    n_pages : int
        The number of pages to retrieve.

    Returns
    -------
    int
        The validated n_pages value.
    """
    if not isinstance(n_pages, int) or n_pages <= 0:
        raise ValueError("n_pages must be a positive integer.")
    return n_pages


def search_url_validator(search_url: str) -> str:
    """
    Validates the search_url parameter.

    Parameters
    ----------
    search_url : str
        The URL of the Bing Search API endpoint.

    Returns
    -------
    str
        The validated search_url value.
    """
    if not isinstance(search_url, str) or not search_url.startswith("http"):
        raise ValueError("search_url must be a valid URL string.")
    return search_url


def headers_validator(headers: dict[str, str]) -> dict[str, str]:
    """
    Validates the headers parameter.

    Parameters
    ----------
    headers : dict[str, str]
        Headers to be used in the request, including the subscription key.

    Returns
    -------
    dict[str, str]
        The validated headers dictionary.
    """
    exp_key = 'Ocp-Apim-Subscription-Key'
    if not isinstance(headers, dict) or exp_key not in headers:
        raise ValueError(f"headers must be a dictionary containing '{exp_key}'.")
    if headers[exp_key] == "":
        raise ValueError("Ocp-Apim-Subscription-Key cannot be an empty string.")
    return headers


@dataclasses.dataclass
class WebSearchQueryData:
    """
    Base class for web search events.
    """
    company: str
    from_date: str = "2022-01-01"
    to_date: str = "2025-06-20"

    def __post_init__(self):
        self.company = company_validator(self.company)
        self.from_date = from_date_validator(self.from_date)
        self.to_date = from_date_validator(self.to_date)


@dataclasses.dataclass
class WebSearchConfigData:
    """
    Base class for web search events.
    """
    size_page: int
    n_pages: int
    search_url: str
    headers: dict[str, str]

    def __post_init__(self):
        self.size_page = size_page_validator(self.size_page)
        self.n_pages = n_pages_validator(self.n_pages)
        self.search_url = search_url_validator(self.search_url)
        self.headers = headers_validator(self.headers)


# Pydantic data structures
class CompanySearchQueryDefinitionFormat(BaseModel):
    """Format used to get llm validation output

        Parameters
        ----------
        obj : type

        Returns
        -------
        obj : type
            description
        """

    company_name: str = Field(description="The company name")
    company_core_business: str = Field(description="The company core business")


class WebSearchQueryDefinitionFormat(BaseModel):
    """description

    Parameters
    ----------
    obj : type

    Returns
    -------
    obj : type
    """

    question: str = Field(description="The question")
    query: str = Field(description="The search engine query")


class MGSearchQueryDefinitionFormat(BaseModel):
    """Format used to get llm validation output"""

    output: list[list[str]] = Field(
        description="List of most similar MGs to the company description.",
        example=[
            """
            [
                ["Red products", "Red products are a key focus for this company."],
                ["Red services", "The company has a strong presence in the red market."]
            ]
            """,
        ])
