"""The model, and the accounting.

    client       metered Gemini client - every call records input, output AND
                 thinking tokens, because 2.5+ models spend output budget on
                 hidden reasoning and omitting it understates agent cost most
    mock_policy  a deterministic rule-based stand-in, so the whole agent loop
                 can be exercised offline with --backend mock, no API key

Token accounting is first-class here rather than bolted on, because the
headline metric of this project is accuracy per 1,000 tokens.
"""
