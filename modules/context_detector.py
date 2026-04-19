
from datetime import datetime

def get_context():

    hour = datetime.now().hour

    if hour < 12:
        return "morning"
    elif hour < 18:
        return "afternoon"
    else:
        return "night"
