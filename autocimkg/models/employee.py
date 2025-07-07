import datetime
import numpy as np

class Employee:
    """
    Designed to represent an employee up for processing by AutoCimKG.
    """

    id: int = 0
    name: str = ""
    name_embedding: np.array
    company: str = ""
    department: str = ""
    status: bool = False

    def __init__(self, id_: int, name: str, name_embedding: np.array, company: str, department: str, status: bool):
        self.id = id_
        self.name = name
        self.name_embedding = name_embedding
        self.company = company
        self.department = department
        self.status = status