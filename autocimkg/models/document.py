import datetime
from typing import Union

class Document:
    """
    Designed to represent a document (chunk) up for processing by AutoCimKG.
    """

    name: str = ""
    doc_type: str = ""
    content: Union[str, list[str]]
    authors: list[str] = []
    language: str = ""

    def __init__(self, name: str, doc_type: str, content: Union[str, list[str]], authors: list[str], language: str):
        self.name = name
        self.doc_type = doc_type
        self.content = content
        self.authors = authors
        self.language = language