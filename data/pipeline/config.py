import os
from dotenv import load_dotenv

load_dotenv()

def get_env_var(key):
    value = os.getenv(key)
    if value is None:
        raise EnvironmentError(f"{key} is not set.")
    return value

ALPHA_VANTAGE_API_KEY=get_env_var("ALPHA_VANTAGE_API_KEY")
POSTGRES_PASS=get_env_var("POSTGRES_PASS")
