from datetime import datetime

class KnowledgeGraphVersion:
    """
    Designed to represent metadata about a single KG version produced by AutoCimKG.
    """

    kg_name: str = ""
    agent: str = ""
    start_proc_ts: datetime = None
    end_proc_ts: datetime = None

    def __init__(self, kg_name: str, agent: str, start_proc_ts: datetime, end_proc_ts: datetime):
        self.kg_name = kg_name
        self.agent = agent
        self.start_proc_ts = start_proc_ts
        self.end_proc_ts = end_proc_ts
