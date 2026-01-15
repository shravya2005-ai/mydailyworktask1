from flask import Flask, render_template, request, jsonify
from chatbot.chatbot import RuleBasedChatbot

app = Flask(__name__)
bot = RuleBasedChatbot()


@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the chat page. Supports a regular form POST for non-JS clients/tests
    and progressive enhancement: the page's JavaScript will talk to `/api/respond`.
    """
    user_input = ''
    response = None
    if request.method == 'POST':
        user_input = request.form.get('message', '')
        response = bot.respond(user_input)
    return render_template('index.html', user_input=user_input, response=response)


@app.route('/api/respond', methods=['POST'])
def respond_api():
    """JSON API used by the single-page UI. Accepts JSON: {"text": "..."}."""
    data = request.get_json(silent=True) or {}
    text = data.get('text') or data.get('message') or ''
    resp = bot.respond(text)
    return jsonify({'response': resp})


if __name__ == '__main__':
    app.run(debug=True)
