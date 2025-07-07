from datetime import datetime

class Log:
    """
    Designed to represent a single log entry produced by AutoCimKG.
    """

    ts: datetime = None
    logger_name: str = ""
    log_level: str = ""
    message: str = ""

    def __init__(self, ts:datetime, logger_name:str, log_level:str, message:str):
        self.ts = ts
        self.logger_name = logger_name
        self.log_level = log_level
        self.message = message