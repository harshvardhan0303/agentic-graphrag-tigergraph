"""Corpus -> property graph. The accuracy-critical half of the project.

    parse_corpus  Wikipedia infoboxes -> typed vertices and edges + text chunks
    normalize     the three corpus traps, each one silently costly if skipped

The design principle lives here: put the facts where they can be computed on,
not where they have to be read. Every infobox field that a question might
aggregate over becomes a typed attribute rather than a sentence.
"""
