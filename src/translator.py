"""Translator service module for LLM-based language classification and translation."""
import json
import os
from typing import Tuple

import ollama  # pylint: disable=import-error


# Get model name from environment variable or use default
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")


def get_language(post: str) -> str:
    """
    Classifies the language of the post as 'English' or 'Non-English'.

    Returns 'English', 'Non-English', or 'Classification Failed' on error.
    """
    context = (
        "You are a language detection system. Your only task is to classify the primary "
        "language of the following text into 'English' or 'Non-English'. "
        "You MUST respond with a valid JSON object following this exact schema: "
        '{"language_classification": "[CLASSIFICATION]"}'
    )

    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": f"Classify the language of this post: '{post}'"}
    ]

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            format='json'
        )

        raw = response.get('message', {}).get('content', '')
        data = json.loads(raw) if isinstance(raw, str) else {}
        classified_language = str(data.get("language_classification", "")).strip().lower()

        if classified_language == "english":
            return "English"
        if classified_language == "non-english":
            return "Non-English"
        return "Classification Failed"

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error querying Ollama for language classification: {e}")
        return "Classification Failed"


def get_translation(post: str) -> str:
    """
    Translates non-English text into English.

    Returns the translated text or an error message.
    """
    context = (
        "You are a highly skilled machine translator. "
        "Your task is to translate any non-English text provided by the user "
        "into fluent, professional English. "
        "If the text is already English, return it unchanged. "
        "Respond only with the translated text, without any additional "
        "commentary, notes, or introductions"
    )

    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": f"Translate the following post: {post}"}
    ]

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages
        )

        return response['message']['content'].strip()

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error querying Ollama for translation: {e}")
        return f"Translation failed due to an error: {e}"


def translate_content(content: str) -> Tuple[bool, str]:
    """
    Main function that classifies language and translates if needed.

    Returns (is_english: bool, translated_content: str)

    This function is robust and handles errors gracefully:
    - If language classification fails, assumes English
    - If translation fails, returns original content
    """
    # Handle empty strings
    if not content or not content.strip():
        return True, content

    # Classify language
    language_result = get_language(content)

    # Handle classification failures gracefully
    if language_result == "Classification Failed":
        # Default to English if classification fails
        return True, content

    # Clean and normalize the language result
    cleaned_lang_result = language_result.strip().rstrip('.,;!?')
    normalized_lang = cleaned_lang_result.title()

    # Check if it's a valid classification
    if normalized_lang not in ("English", "Non-English"):
        # Invalid format, default to English
        return True, content

    is_english = normalized_lang == "English"

    if is_english:
        # Already English, return as-is
        return True, content

    # Non-English, translate it
    translation_result = get_translation(content)

    # Check if translation failed
    if translation_result.startswith("Translation failed"):
        # Translation failed, return original content
        return False, content

    return False, translation_result
