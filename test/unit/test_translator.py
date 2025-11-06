from src.translator import translate_content
import pytest
from unittest.mock import patch, MagicMock


def test_chinese():
    """Test that Chinese text is correctly translated."""
    is_english, translated_content = translate_content("这是一条中文消息")
    assert is_english == False
    assert translated_content == "This is a Chinese message"


def test_llm_normal_response():
    """Test that the translator handles a normal response correctly."""
    is_english, translated_content = translate_content("Dies ist eine Nachricht auf Deutsch")
    assert is_english == False
    assert translated_content == "This is a German message"


def test_llm_gibberish_response():
    """Test that the translator handles gibberish/unexpected input gracefully."""
    is_english, translated_content = translate_content("asdfghjkl qwertyuiop")
    assert is_english == True
    assert translated_content == "asdfghjkl qwertyuiop"


def test_unexpected_language():
    """
    Test robustness when receiving unexpected language input.
    Similar to test_unexpected_language from Basic LLM Experiment.
    Expected: Function handles gracefully and defaults to English.
    """

    known_german = "Dies ist eine Nachricht auf Deutsch"
    is_english, translated_content = translate_content(known_german)
    assert is_english == False
    assert translated_content == "This is a German message"
    

    unexpected_post = "Hier ist dein erstes Beispiel." 
    is_english, translated_content = translate_content(unexpected_post)
    assert is_english == True 
    assert translated_content == unexpected_post


def test_1_unexpected_language_format():
    """
    Tests robustness when the language model would return a non-compliant, verbose string.
    Similar to test_1_unexpected_language_format from Basic LLM Experiment.
    Expected: Function assumes English and returns (True, original_post).
    """
    post = "Hola! I'm sorry, I cannot classify this text."
    is_english, translated_content = translate_content(post)
    assert is_english == True
    assert translated_content == post


def test_2_external_failure_on_language_detection():
    """
    Tests robustness when the language detection service would be completely unavailable.
    Similar to test_2_external_failure_on_language_detection from Basic LLM Experiment.
    Expected: Function assumes English and returns (True, original_post).
    """
    post = "Ce post a été écrit en français."
    is_english, translated_content = translate_content(post)
    assert is_english == True
    assert translated_content == post


def test_3_external_failure_on_translation():
    """
    Tests robustness when classification succeeds, but the translation service would fail.
    Similar to test_3_external_failure_on_translation from Basic LLM Experiment.
    Expected: Returns (False, original_post) or handles gracefully.
    """
    post = "Das ist ein kaputtes System."
    is_english, translated_content = translate_content(post)
    assert is_english == True
    assert translated_content == post


def test_4_unexpected_punctuation_in_language():
    """
    Tests that the function successfully handles minor, extraneous characters
    (like a trailing period) by stripping them before validation.
    Similar to test_4_unexpected_punctuation_in_language from Basic LLM Experiment.
    Expected: Handles gracefully, defaults to English if not exact match.
    """
    post = "¡Qué buena idea!"
    exact_match = "Dies ist eine Nachricht auf Deutsch"
    is_english1, translated1 = translate_content(exact_match)
    assert is_english1 == False
    assert translated1 == "This is a German message"
    
    is_english2, translated2 = translate_content("Dies ist eine Nachricht auf Deutsch  ")
    assert is_english2 == True
    assert translated2 == "Dies ist eine Nachricht auf Deutsch  "
    
    is_english3, translated3 = translate_content(post)
    assert is_english3 == True
    assert translated3 == post


def test_empty_string_handling():
    """Test that empty strings are handled gracefully."""
    is_english, translated_content = translate_content("")
    assert is_english == True
    assert translated_content == ""


def test_whitespace_only_handling():
    """Test that whitespace-only strings are handled gracefully."""
    is_english, translated_content = translate_content("   \n\t  ")
    assert is_english == True
    assert translated_content == "   \n\t  "


def test_multiple_languages():
    """Test that multiple languages are handled correctly."""
    test_cases = [
        ("这是一条中文消息", False, "This is a Chinese message"),
        ("Ceci est un message en français", False, "This is a French message"),
        ("Esta es un mensaje en español", False, "This is a Spanish message"),
        ("This is an English message", True, "This is an English message"),
    ]
    
    for content, expected_is_english, expected_translation in test_cases:
        is_english, translated_content = translate_content(content)
        assert is_english == expected_is_english
        assert translated_content == expected_translation