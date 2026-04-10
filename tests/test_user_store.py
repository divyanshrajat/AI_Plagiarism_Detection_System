import time
from modules import user_store


def test_reset_token_and_password():
    username = 'testuser_reset'
    # ensure user exists
    user_store.create_user(username, 'origpass', role='student')
    token = user_store.generate_reset_token(username)
    assert token
    verified = user_store.verify_reset_token(token)
    assert verified == username
    # reset password
    assert user_store.reset_password(username, 'newpass')
    auth = user_store.authenticate(username, 'newpass')
    assert auth and auth['username'] == username
    # cleanup
    user_store.delete_user(username)
