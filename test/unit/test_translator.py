"""Unit tests for translator service module."""
from unittest.mock import patch

from src.translator import translate_content


# Unit tests with mocked LLM calls (following Basic LLM Experiment pattern)
@patch('src.translator.get_translation')
@patch('src.translator.get_language')
def test_chinese(mock_language, mock_translate):
    """Test that Chinese text is correctly translated."""
    mock_language.return_value = "Non-English"
    mock_translate.return_value = "This is a Chinese message"

    is_english, translated_content = translate_content("这是一条中文消息")
    assert not is_english
    assert translated_content == "This is a Chinese message"


@patch('src.translator.get_translation')
@patch('src.translator.get_language')
def test_llm_normal_response(mock_language, mock_translate):
    """Test that the translator handles a normal response correctly."""
    mock_language.return_value = "Non-English"
    mock_translate.return_value = "This is a German message"

    is_english, translated_content = translate_content("Dies ist eine Nachricht auf Deutsch")
    assert not is_english
    assert translated_content == "This is a German message"


@patch('src.translator.get_translation')
@patch('src.translator.get_language')
def test_llm_gibberish_response(mock_language, _mock_translate):
    """Test that the translator handles gibberish/unexpected input gracefully."""
    mock_language.return_value = "English"

    is_english, translated_content = translate_content("asdfghjkl qwertyuiop")
    assert is_english
    assert translated_content == "asdfghjkl qwertyuiop"


@patch('src.translator.get_translation')
@patch('src.translator.get_language')
def test_unexpected_language(mock_language, mock_translate):
    """
    Test robustness when receiving unexpected language input.

    Similar to test_unexpected_language from Basic LLM Experiment.
    Expected: Function handles gracefully and defaults to English.
    """
    # Test with known German text
    mock_language.return_value = "Non-English"
    mock_translate.return_value = "This is a German message"

    known_german = "Dies ist eine Nachricht auf Deutsch"
    is_english, translated_content = translate_content(known_german)
    assert not is_english
    assert translated_content == "This is a German message"

    # Test with unexpected input that returns unexpected format
    mock_language.return_value = "I don't understand your request"

    unexpected_post = "Hier ist dein erstes Beispiel."
    is_english, translated_content = translate_content(unexpected_post)
    assert is_english
    assert translated_content == unexpected_post


@patch('src.translator.get_translation')
@patch('src.translator.get_language')
def test_1_unexpected_language_format(mock_language, _mock_translate):
    """
    Tests robustness when the language model would return a non-compliant, verbose string.

    Similar to test_1_unexpected_language_format from Basic LLM Experiment.
    Expected: Function assumes English and returns (True, original_post).
    """
    post = "Hola! I'm sorry, I cannot classify this text."
    mock_language.return_value = "I am a helpful assistant, and I classify the text as Spanish."

    is_english, translated_content = translate_content(post)
    assert is_english
    assert translated_content == post


@patch('src.translator.get_translation')
@patch('src.translator.get_language')
def test_2_external_failure_on_language_detection(mock_language, _mock_translate):
    """
    Tests robustness when the language detection service would be completely unavailable.

    Similar to test_2_external_failure_on_language_detection from Basic LLM Experiment.
    Expected: Function assumes English and returns (True, original_post).
    """
    post = "Ce post a été écrit en français."
    mock_language.return_value = "Classification Failed"

    is_english, translated_content = translate_content(post)
    assert is_english
    assert translated_content == post


@patch('src.translator.get_translation')
@patch('src.translator.get_language')
def test_3_external_failure_on_translation(mock_language, mock_translate):
    """
    Tests robustness when classification succeeds, but the translation service would fail.

    Similar to test_3_external_failure_on_translation from Basic LLM Experiment.
    Expected: Returns (False, original_post) or handles gracefully.
    """
    post = "Das ist ein kaputtes System."
    mock_language.return_value = "Non-English"
    mock_translate.return_value = "Translation failed due to an error: Connection error"

    is_english, translated_content = translate_content(post)
    assert not is_english
    assert translated_content == post


@patch('src.translator.get_translation')
@patch('src.translator.get_language')
def test_4_unexpected_punctuation_in_language(mock_language, mock_translate):
    """
    Tests that the function successfully handles minor, extraneous characters.

    (like a trailing period) by stripping them before validation.
    Similar to test_4_unexpected_punctuation_in_language from Basic LLM Experiment.
    Expected: Passes validation, proceeds to successful translation.
    """
    # Test exact match works
    exact_match = "Dies ist eine Nachricht auf Deutsch"
    mock_language.return_value = "Non-English"
    mock_translate.return_value = "This is a German message"

    is_english1, translated1 = translate_content(exact_match)
    assert not is_english1
    assert translated1 == "This is a German message"

    # Test with extra whitespace (should default to English since not exact match)
    mock_language.return_value = "English"
    is_english2, translated2 = translate_content("Dies ist eine Nachricht auf Deutsch  ")
    assert is_english2
    assert translated2 == "Dies ist eine Nachricht auf Deutsch  "

    # Test unexpected punctuation (with trailing period/newline)
    post = "¡Qué buena idea!"
    mock_language.return_value = "Non-English.\n"
    mock_translate.return_value = "What a good idea!"

    is_english3, translated3 = translate_content(post)
    assert not is_english3
    assert translated3 == "What a good idea!"


def test_empty_string_handling():
    """Test that empty strings are handled gracefully."""
    is_english, translated_content = translate_content("")
    assert is_english
    assert translated_content == ""


def test_whitespace_only_handling():
    """Test that whitespace-only strings are handled gracefully."""
    is_english, translated_content = translate_content("   \n\t  ")
    assert is_english
    assert translated_content == "   \n\t  "


@patch('src.translator.get_translation')
@patch('src.translator.get_language')
def test_multiple_languages(mock_language, mock_translate):
    """Test that multiple languages are handled correctly."""
    test_cases = [
        ("这是一条中文消息", False, "This is a Chinese message"),
        ("Ceci est un message en français", False, "This is a French message"),
        ("Esta es un mensaje en español", False, "This is a Spanish message"),
        ("This is an English message", True, "This is an English message"),
    ]

    for content, expected_is_english, expected_translation in test_cases:
        if expected_is_english:
            mock_language.return_value = "English"
        else:
            mock_language.return_value = "Non-English"
            mock_translate.return_value = expected_translation

        is_english, translated_content = translate_content(content)
        assert is_english == expected_is_english
        assert translated_content == expected_translation
