"""Measurement: the oracle, the runner, the judge, the metrics, the dashboard.

    oracle     graph correctness gate - drives the tools with regexes, no LLM,
               so a failure here is a data bug and never a prompting bug
    run        every question x every pipeline -> results/answers/summary
    judge      LLM-as-judge with an exact-match fast path
    metrics    accuracy, completeness, citation precision, tokens, acc/1k
    dashboard  one self-contained HTML report

Judge tokens are metered separately from pipeline tokens so that no pipeline
is charged for the cost of being graded.
"""
