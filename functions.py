import re


def extract_class_from_result(text: str, categories: list) -> str:
    """
    From the text answer of a LLM, the function extracts the category selected by the LLM.
    If the LLM answer contains more than one category, the result is considered INEXPLOITABLE.
    :param categories:
    :return:
    """
    tag_category = ''
    for category in categories:
        pattern = fr'\b{category}\b'

        if re.search(pattern, text):
            if tag_category == '':
                tag_category = category
            else:
                return 'INEXPLOITABLE'

    return tag_category