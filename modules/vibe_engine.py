
def map_vibe(emotion, sentiment, context):

    if emotion == "happy":
        return "energetic"

    if emotion == "sad":
        return "calm"

    if emotion == "angry":
        return "relax"

    if sentiment == "positive":
        return "party"

    if context == "night":
        return "chill"

    return "neutral"
