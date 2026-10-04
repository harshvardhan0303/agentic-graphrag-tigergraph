"""Agentic GraphRAG on TigerGraph.

Three question-answering pipelines over one corpus, one graph and one tool
layer, built so that the only thing differing between them is whether
retrieval reacts to what it finds:

    pipelines.rag       vector top-k, one shot
    pipelines.graphrag  a fixed graph traversal, same shape every question
    pipelines.agentic   an orchestrator that chooses each step

Everything else here exists to serve that comparison: `ingest` builds the
graph, `graph` serves it, `vectorstore` serves the text, `agents` holds the
agent machinery, `llm` meters every token, and `bench` scores the lot.
"""
