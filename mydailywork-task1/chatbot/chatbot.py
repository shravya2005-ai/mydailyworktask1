import re
from typing import List, Tuple, Callable, Pattern

class RuleBasedChatbot:
    """Simple rule-based chatbot.

    Usage:
        bot = RuleBasedChatbot()
        bot.respond("Hello")

    The bot has a list of (pattern, handler) rules. Patterns are regex strings
    (case-insensitive). Handlers are functions that take the match object and
    return a response string.
    """

    def __init__(self):
        self.rules: List[Tuple[Pattern, Callable]] = []
        self._register_default_rules()

    def add_rule(self, pattern: str, handler: Callable):
        """Add a regex rule. Pattern is compiled as case-insensitive."""
        self.rules.append((re.compile(pattern, re.IGNORECASE), handler))

    def _register_default_rules(self):
        # Greetings
        self.add_rule(r"^(hi|hello|hey)\b", lambda m: "Hello! How can I help you today?")
        # Goodbye
        self.add_rule(r"\b(bye|goodbye|see you)\b", lambda m: "Goodbye! Have a great day.")
        # Thanks
        self.add_rule(r"\b(thank(s| you)|thx)\b", lambda m: "You're welcome!")
        # Ask name
        self.add_rule(r"\b(what is your name|who are you)\b", lambda m: "I'm a simple rule-based chatbot.")
        # Ask time
        self.add_rule(r"\b(what time|current time|time is it)\b", self._handle_time)
        # Weather (very naive)
        self.add_rule(r"\b(weather|rain|sunny|cloudy)\b", lambda m: "I can't check the weather, but it sounds like you're curious about it.")
        # Affirmation
        self.add_rule(r"\b(yes|sure|okay|ok)\b", lambda m: "Alright.")
        # Numeric question example: add two numbers
        self.add_rule(r"add (\d+) and (\d+)", lambda m: f"The sum is {int(m.group(1)) + int(m.group(2))}.")

    def _handle_time(self, match):
        from datetime import datetime
        now = datetime.now()
        return f"Current local time is {now.strftime('%Y-%m-%d %H:%M:%S')}."

    def respond(self, text: str) -> str:
        """Return a response according to the first matching rule, or a fallback."""
        text = (text or "").strip()
        if not text:
            return "Please say something so I can respond."

        for pattern, handler in self.rules:
            m = pattern.search(text)
            if m:
                try:
                    return handler(m)
                except Exception:
                    return "Sorry, I couldn't process that."  # safe fallback for handler errors

        # No rule matched
        return "I don't understand. Can you rephrase?"


if __name__ == "__main__":
    bot = RuleBasedChatbot()
    print(bot.respond("hello"))
