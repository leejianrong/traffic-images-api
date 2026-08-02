"""Camera-id lookup: mapped names, the documented fallbacks for unknown ids,
and a structural sanity check on the table itself."""

from __future__ import annotations

from sgtraffic.locations import CAMERA_NAMES, UNKNOWN, expressway, friendly_name


# --------------------------------------------------------------------------- #
# known ids
# --------------------------------------------------------------------------- #
def test_friendly_name_returns_the_mapped_location():
    assert friendly_name("2702") == "Woodlands Checkpoint (towards BKE)"
    assert friendly_name("6710") == "PIE (Clementi, towards Changi)"
    assert friendly_name("4798") == "Tuas Checkpoint"


def test_expressway_returns_the_mapped_zone_code():
    assert expressway("2702") == "BKE"
    assert expressway("6710") == "PIE"
    assert expressway("1701") == "CTE"
    assert expressway("4798") == "AYE"  # checkpoints are grouped by expressway


def test_every_mapped_id_resolves_to_its_table_entry():
    for camera_id, (zone, location) in CAMERA_NAMES.items():
        assert expressway(camera_id) == zone
        assert friendly_name(camera_id) == location


# --------------------------------------------------------------------------- #
# unknown ids -- documented graceful fallback
# --------------------------------------------------------------------------- #
def test_friendly_name_falls_back_to_the_camera_id():
    assert "9999" not in CAMERA_NAMES
    assert friendly_name("9999") == "Camera 9999"


def test_expressway_falls_back_to_the_unknown_group():
    assert expressway("9999") == UNKNOWN == "OTHER"


def test_lookups_never_raise_on_odd_ids():
    for camera_id in ("", "not-an-id", "2702 "):
        assert friendly_name(camera_id) == f"Camera {camera_id}"
        assert expressway(camera_id) == UNKNOWN


# --------------------------------------------------------------------------- #
# table structure
# --------------------------------------------------------------------------- #
def test_table_entries_are_well_formed():
    for camera_id, entry in CAMERA_NAMES.items():
        assert isinstance(camera_id, str), camera_id
        assert camera_id.isdigit(), camera_id  # ids are numeric strings
        assert isinstance(entry, tuple) and len(entry) == 2, camera_id
        zone, location = entry
        assert zone and zone == zone.strip(), camera_id
        assert location and location == location.strip(), camera_id


def test_table_zone_codes_are_short_uppercase_and_never_the_sentinel():
    for camera_id, (zone, _location) in CAMERA_NAMES.items():
        assert zone.isupper() and len(zone) <= 4, camera_id
        assert zone != UNKNOWN, camera_id  # UNKNOWN is the fallback, not a group
