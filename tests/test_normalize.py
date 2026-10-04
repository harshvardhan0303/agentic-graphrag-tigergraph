"""The three corpus traps, pinned as tests.

Each of these silently cost accuracy before it was fixed, so each gets a named
regression test rather than a comment.
"""
from src.ingest.normalize import norm, split_people, to_int


def test_number_unit_runon_normalises_like_spaced():
    # corpus has both "Men's 68kg" (2012) and "Men's 68 kg" (2008)
    assert norm("Men's 68kg") == norm("Men's 68 kg")
    assert norm("Men's 4x100 metre relay") == norm("Men's 4 x 100 metre relay")


def test_plus_weight_classes_stay_distinct():
    # "+80 kg" is a different event from "80 kg"; naive stripping merges them
    assert norm("Men's +80 kg") != norm("Men's 80 kg")
    assert "plus" in norm("Men's +80 kg")


def test_accents_and_dashes_fold():
    assert norm("Servet Tazegül") == norm("Servet Tazegul")
    assert norm("Athletics – Men's marathon") == norm("Athletics - Men's marathon")


def test_to_int_handles_messy_values():
    assert to_int("24") == 24
    assert to_int("1,024") == 1024
    assert to_int("24 (12 boats)") == 24
    assert to_int("") is None
    assert to_int(None) is None


def test_split_people_recovers_team_members():
    assert split_people("Dani KingLaura TrottJoanna Rowsell") == [
        "Dani King", "Laura Trott", "Joanna Rowsell"]
    assert split_people("Naim Süleymanoğlu") == ["Naim Süleymanoğlu"]
    assert split_people("") == []


# ---------------------------------------------------------------------------
# Trap 4: date spelling. Questions and infoboxes order dates differently, and
# infobox dates are frequently ranges. Comparing normalised strings therefore
# misses real events; date_keys compares (month, day) instead.
# ---------------------------------------------------------------------------
from src.ingest.normalize import date_keys


def test_date_word_order_does_not_matter():
    assert date_keys("August 12, 2008") == date_keys("12 August 2008") == {"08-12"}


def test_iso_and_written_dates_agree():
    assert date_keys("2004-08-24") == date_keys("24 August 2004") == {"08-24"}


def test_same_month_range_covers_every_day():
    assert date_keys("11–17 August") == {f"08-{d:02d}" for d in range(11, 18)}


def test_cross_month_range_keeps_endpoints_only():
    # Expanding across a month boundary needs month lengths we do not model;
    # the endpoints are the anchors questions actually name.
    assert date_keys("27 July to 3 August") == {"07-27", "08-03"}


def test_parenthetical_stages_are_ignored_but_both_days_kept():
    assert date_keys("25 September 2000 (qualification)27 September 2000 (final)") \
        == {"09-25", "09-27"}


def test_year_is_never_mistaken_for_a_day():
    assert date_keys("7 August 2016") == {"08-07"}


def test_empty_and_undated():
    assert date_keys("") == set()
    assert date_keys("TBD") == set()
