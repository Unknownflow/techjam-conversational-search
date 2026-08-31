"""Initial conversational intent routing."""


class IntentRouter:
    """Choose precision-first Buying or discovery-first Browsing retrieval."""

    BUYING_SIGNALS = (
        "key requirement",
        "what i need",
        "must have",
        "need ",
        "budget",
    )
    BROWSING_SIGNALS = (
        "exploring",
        "browse",
        "ideas",
        "inspiration",
        "not sure",
    )

    @classmethod
    def route(cls, message: str) -> str:
        lowered = message.lower()
        if any(signal in lowered for signal in cls.BROWSING_SIGNALS):
            return "browsing"
        if any(signal in lowered for signal in cls.BUYING_SIGNALS):
            return "buying"
        return "browsing"
