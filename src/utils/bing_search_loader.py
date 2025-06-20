"""
The module provides a class `BingSearchLoader` that allows users
to search for company information using the Bing Search API.
It includes methods for initializing the loader with specific
parameters and performing searches based on company names and date ranges.
"""
import json

import pandas as pd
import requests

from .custom_dataclass import WebSearchConfigData, WebSearchQueryData


def bing_search(
        query: WebSearchQueryData,
        config: WebSearchConfigData) -> list[dict]:
    """ The search method retrieves information about a company
    from Bing Search API. This method constructs a query to search
    for the company's activities and returns a list of dictionaries
    containing the search results.

    Parameters
    ----------
    company : str
        The name of the company to search for.
    from_date : str, optional
        Start date ("YYYY-MM-DD") for the search, default "2022-01-01"
    to_date : str, optional
        _description_, by default "2025-06-20"

    Returns
    -------
    list[dict]
        _description_
    """

    df_container = []
    params = {
        "q": f"What does {query.company} do?", "textDecorations": True,
        "textFormat": "Raw",
        "setLang": "it",
        "BingAPIs-Market": "it-IT",
        "X-Search-Location": "disp:Rome,Italy",
        # "cc": "it-IT",
        "count": config.size_page,
        "freshness": f"{query.from_date}..{query.to_date}",
        "mkt": "it-IT"
    }
    print(params)

    for i in range(config.n_pages):
        try:
            params["offset"] = i

            print(f'\t {i} Search term: {params["q"]} - Offset: {params["offset"]}')
            response = requests.get(config.search_url,
                                    headers=config.headers,
                                    params=params,  # type: ignore
                                    timeout=30)

            #
            response.raise_for_status()
            search_results = response.json()
            search_results_json = json.dumps(search_results, indent=2)

            response_dict = json.loads(search_results_json)
            response_results = response_dict['webPages']['value']

            #
            df_stg = pd.DataFrame(response_results)
            df_stg['offset'] = params["offset"]

            #
            df_container.append(df_stg)

        except KeyError as e:
            print(e)
            break

    if len(df_container) > 0:

        df = pd.concat(df_container)
        df_final = df.drop_duplicates(subset='url', keep="first")

        return df_final.to_dict(orient='records')

    return [{}]


if __name__ == "__main__":

    try:
        from env_var_loader import (
            BING_FROM_DATE,
            BING_HEADERS,
            BING_N_PAGES,
            BING_SEARCH_URL,
            BING_SIZE_PAGE,
            BING_TO_DATE,
        )
    except ImportError:
        import os
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from env_var_loader import (
            BING_FROM_DATE,
            BING_HEADERS,
            BING_N_PAGES,
            BING_SEARCH_URL,
            BING_SIZE_PAGE,
            BING_TO_DATE,
        )

    #
    SEARCH_COMPANY = "Suzuki"

    web_search_query = WebSearchQueryData(
        SEARCH_COMPANY,
        BING_FROM_DATE,
        BING_TO_DATE
    )

    web_search_query_config = WebSearchConfigData(
        BING_SIZE_PAGE,
        BING_N_PAGES,
        BING_SEARCH_URL,
        BING_HEADERS
    )

    output = bing_search(web_search_query, web_search_query_config)

    # Iterate over the DataFrame and print each row
    for row in output:
        print(f"Title: {row['name']}")
        print(f"URL: {row['url']}")
        print(f"Snippet: {row['snippet']}")
        print(f"Offset: {row['offset']}")
        print("-" * 40)

    # output.to_csv("output_bing_search.csv", index=False, encoding='utf-8-sig')
    # print("File saved as output_bing_search.csv")
