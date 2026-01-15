from chatbot.chatbot import RuleBasedChatbot


def main():
    bot = RuleBasedChatbot()
    print("Simple rule-based chatbot. Type 'quit' or 'exit' to stop.")
    while True:
        try:
            text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not text:
            continue
        if text.strip().lower() in {"quit", "exit"}:
            print("Bot: Goodbye!")
            break
        resp = bot.respond(text)
        print("Bot:", resp)


if __name__ == "__main__":
    main()
