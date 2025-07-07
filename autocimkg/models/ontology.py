class Ontology:
    """
    Designed to represent a lightweight ontology up for processing by AutoCimKG.
    Topic is of form "Name: Description (e.g. Examples)".
    Relation is of form "Name: Description" (restriction in the form (Topic, Name, Topic) TBD!).
    Strict determines if ontology should be enforced or is open for enrichment (w.r.t. the topics).
    Relations always get enforced.
    """

    topics: list[dict] = []
    relations: list[dict] = [] # Union[list[dict], list[Tuple[str, str, str]]] TBD!
    strict: bool = True

    def __init__(self, topics: list[dict], relations: list[dict], strict: bool):
        self.topics = topics
        self.relations = relations
        self.strict = strict

class AlignedEntity:
    name: str = ""
    label: str = ""
    topic: str = ""

    def __init__(self, name: str, label: str, topic: str):
        self.name = name
        self.label = label
        self.topic = topic

class NovelTopic:
    topic: str = ""
    description: str = ""
    examples: list[str] = ""

    def __init__(self, topic: str, description: str, examples: list[str]):
        self.topic = topic
        self.description = description
        self.examples = examples

class AlignedRelationship:
    list_number: int = 0
    start_entity: str = ""
    name: str = ""
    end_entity: str = ""

    def __init__(self, list_number: int, start_entity: str, name: str, end_entity: str):
        self.list_number = list_number
        self.start_entity = start_entity
        self.name = name
        self.end_entity = end_entity