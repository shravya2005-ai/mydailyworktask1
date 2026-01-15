# Rule-Based Chatbot

A tiny Python rule-based chatbot that responds to user input using predefined regex rules.

Files
- `chatbot/chatbot.py` - core RuleBasedChatbot class with rules and `respond` method.
- `run_chatbot.py` - simple interactive CLI to talk to the bot.
- `tests/test_chatbot.py` - basic pytest tests.

Run the chatbot (powershell):

```pwsh
python -m run_chatbot
```

Run tests:

```pwsh
python -m pytest -q
```

Run the web interface (development server):

```pwsh
C:/Users/LENOVO/OneDrive/Desktop/mydailywork-task1/.venv/Scripts/python.exe -m webapp.app
```

Then open http://127.0.0.1:5000/ in your browser.

Notes:
- This is intentionally simple for learning purposes. Extend by adding rules or a small NLU parser.
