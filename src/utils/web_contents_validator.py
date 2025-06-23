"""aaa"""


def web_contents_validator(web_contents: list[dict]) -> tuple[bool, str]:
    """
    Validates the web_contents parameter.

    Parameters
    ----------
    web_contents : str
        The contents of the web page to validate.

    Returns
    -------
    str
        The validated web_contents value.
    """

    verified = True
    message = "Web contents are valid"

    # Check if web contents already exists
    if len(web_contents) == 0:
        message = "No contents found. Not able to answer the question"
        verified = False
        return verified, message

    # Validate retrieved content structure
    if not isinstance(web_contents, list):
        raise ValueError("Retrieved content must be a list of dictionaries")

    # Check if each element in the list is a dictionary with 'url' key
    if not all(isinstance(el, dict) and 'url' in el for el in web_contents):
        raise ValueError("Retrieved content must be a list of dicts with 'url' key")

    return verified, message
