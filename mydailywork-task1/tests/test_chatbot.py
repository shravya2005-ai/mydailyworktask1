import re
from chatbot.chatbot import RuleBasedChatbot


def test_greeting():
    bot = RuleBasedChatbot()
    assert "hello" in bot.respond("Hello").lower()


def test_fallback():
    bot = RuleBasedChatbot()
    res = bot.respond("asdkfjaskdf")
    assert "don't understand" in res or "rephrase" in res


def test_add_numbers():
    bot = RuleBasedChatbot()
    assert "The sum is 7" in bot.respond("add 3 and 4")


def test_empty():
    bot = RuleBasedChatbot()
    assert "please say something" in bot.respond("").lower()


def test_time_response_contains_date():
    bot = RuleBasedChatbot()
    res = bot.respond("what time is it")
    # very basic check that a year-like number exists
    assert re.search(r"\d{4}-\d{2}-\d{2}", res)
