from webapp.app import app


def test_index_get():
    client = app.test_client()
    r = client.get('/')
    assert r.status_code == 200


def test_post_message():
    client = app.test_client()
    r = client.post('/', data={'message': 'Hello'})
    # bot's greeting includes "How can I help" per default rules
    assert b'How can I help' in r.data
