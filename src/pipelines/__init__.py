"""The three pipelines under comparison, plus the router ablation.

    rag       one similarity search, one generation. No graph, no second look.
    graphrag  one extraction call, a FIXED graph traversal, one generation.
              Structure-aware but not adaptive - the same sequence every time.
    agentic   an orchestrator choosing each step from what it has learned.
    base      the shared plumbing, so all three return the same Answer shape.

`agentic.py` also exports the no-router ablation. Shipping the experiment that
can disprove your own feature is the point, not a formality.
"""
