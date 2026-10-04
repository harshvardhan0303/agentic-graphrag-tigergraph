"""Graph access, behind one interface with two implementations.

    store       the interface + LocalGraphStore (in-memory, used by tests)
    tigergraph  TigerGraphStore, the real backend, over installed GSQL queries
    deploy      schema -> data -> queries -> vectors, re-runnably

Both stores must return the same answers on all 100 public questions. That is
not redundancy for its own sake: it is what makes the GSQL testable, and it is
how the `previous_games` bug was caught.
"""
