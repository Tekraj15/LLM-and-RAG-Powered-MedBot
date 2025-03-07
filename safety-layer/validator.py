def validate_response(response: str) -> str:
    unsafe_phrases = ["take more", "stop medication", "self-diagnose"]
    if any(phrase in response.lower() for phrase in unsafe_phrases):
        return "Please consult a healthcare professional."
    return response