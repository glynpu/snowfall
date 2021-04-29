
from typing import Optional
from typing import Tuple
from typing import Union

def remove_parenthesis(value: str):
    value = value.strip()
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1]
    elif value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return value

def remove_quotes(value: str):
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    elif value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    return value

def str2triple_str(value: str) -> Tuple[str, str, str]:
    """str2triple_str.

    Examples:
        >>> str2triple_str('abc,def ,ghi')
        ('abc', 'def', 'ghi')
    """
    value = remove_parenthesis(value)
    a, b, c = value.split(",")

    # Workaround for configargparse issues:
    # If the list values are given from yaml file,
    # the value givent to type() is shaped as python-list,
    # e.g. ['a', 'b', 'c'],
    # so we need to remove quotes from it.
    return remove_quotes(a), remove_quotes(b), remove_quotes(c)
