import os
import uuid
import secrets
import toml

def generate_secret(secret_length: int = 64) -> str:
    """
    Generates a secret value of length `secret_length` 
    and returns it as a string.

    :param secret_length: (int) The number of characters you want in your secret.
    :returns: (str) String representation of the generated secret key.
    """

    secret = secrets.token_urlsafe(secret_length)
    return str(secret)

def write_secret(secret: str, config_toml: str) -> bool:
    """
    Writes a provided secret to a `toml` file.

    :param secret: (str) Secret value to be written.
    :param config_toml: (str) path of file to write to.
    :returns: True if successfully written else False.
    """

    success = False
    try:
        config = toml.load(config_toml)
        config['server']['cookieSecret'] = secret
        with open(config_toml, 'w') as config_file:
            config_file.write(toml.dumps(config))
            config_file.close()
        success = True
    # TODO: Add better error handling
    except Exception as e:
        print(f'Unable to write secret to config file: {e}')
    return success

if __name__ == '__main__':
    secret = generate_secret()
    write_secret(secret, '~/.streamlit/config.toml')
    