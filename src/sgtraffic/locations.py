"""Static mapping of Data.gov.sg traffic camera IDs to human-friendly names.

The traffic-images API only returns numeric ``camera_id`` values and raw
lat/long coordinates -- it never tells you *where* a camera is. This module
holds the community-known mapping so the UI can show "Woodlands Checkpoint"
instead of "2702", and group cameras by expressway.

Any camera_id not in the table falls back to ``Camera <id>`` and the
``UNKNOWN`` expressway group, so the app never breaks when SG adds cameras.
"""

from __future__ import annotations

# camera_id -> (expressway/zone code, descriptive location)
CAMERA_NAMES: dict[str, tuple[str, str]] = {
    # --- Causeways & Checkpoints ---
    "2701": ("BKE", "Woodlands Causeway (towards Johor)"),
    "2702": ("BKE", "Woodlands Checkpoint (towards BKE)"),
    "2703": ("BKE", "BKE (towards Woodlands Checkpoint)"),
    "2704": ("BKE", "BKE (Woodlands Flyover, towards Checkpoint)"),
    "2705": ("BKE", "BKE (towards PIE)"),
    "2706": ("BKE", "BKE (Chantek F/O, towards PIE)"),
    "4798": ("AYE", "Tuas Checkpoint"),
    "4799": ("AYE", "Tuas Second Link"),
    # --- Central Expressway (CTE) ---
    "1701": ("CTE", "CTE (towards city, Braddell)"),
    "1702": ("CTE", "CTE (towards city, Ang Mo Kio Ave 1)"),
    "1703": ("CTE", "CTE (towards SLE, Blk 22 St George's)"),
    "1704": ("CTE", "CTE (towards AYE, Moulmein Flyover)"),
    "1705": ("CTE", "CTE (towards AYE, Chin Swee)"),
    "1706": ("CTE", "CTE (towards SLE, Ang Mo Kio Ave 5)"),
    "1707": ("CTE", "CTE (Yio Chu Kang, towards city)"),
    "1709": ("CTE", "CTE (towards SLE, Bukit Merah)"),
    "1711": ("CTE", "CTE (towards AYE, Ang Mo Kio)"),
    # --- East Coast Parkway (ECP) ---
    "1001": ("ECP", "ECP (Marina, exit to Marina Blvd)"),
    "1002": ("ECP", "ECP (Benjamin Sheares Bridge)"),
    "1003": ("ECP", "ECP (Fort Rd, towards city)"),
    "1004": ("ECP", "ECP (Tanjong Katong, towards city)"),
    "1005": ("ECP", "ECP (Tanjong Rhu)"),
    "1006": ("ECP", "ECP (Laguna Flyover, towards city)"),
    "3702": ("ECP", "ECP (towards city, Bedok Sth Ave 3)"),
    "3704": ("ECP", "ECP (towards Changi, Xilin Ave)"),
    "3705": ("ECP", "ECP (Entrance from PIE, Changi)"),
    "3793": ("ECP", "ECP (towards city, Tanah Merah)"),
    "3795": ("ECP", "ECP (towards Changi, Bedok Sth)"),
    "3796": ("ECP", "ECP (towards city, Marine Parade)"),
    "3797": ("ECP", "ECP (towards Changi, Tanah Merah)"),
    "3798": ("ECP", "ECP (towards city, Laguna F/O)"),
    # --- Kallang-Paya Lebar Expressway (KPE) ---
    "1111": ("KPE", "KPE (Sims Way, towards TPE)"),
    "1112": ("KPE", "KPE (Defu Flyover, towards ECP)"),
    "1113": ("KPE", "KPE (Tampines Rd, towards ECP)"),
    "8701": ("KPE", "KPE (towards ECP, Defu South)"),
    "8702": ("KPE", "KPE (towards TPE, Kallang)"),
    "8704": ("KPE", "KPE (towards ECP, Tampines)"),
    "8706": ("KPE", "KPE (towards TPE, Defu)"),
    # --- Pan Island Expressway (PIE) ---
    "5794": ("PIE", "PIE (towards Changi, Bedok Nth)"),
    "5795": ("PIE", "PIE (Eng. Corp, towards Jurong)"),
    "5797": ("PIE", "PIE (towards Changi, Paya Lebar)"),
    "5798": ("PIE", "PIE (towards Jurong, Kallang)"),
    "5799": ("PIE", "PIE (towards Changi, Kallang)"),
    "6701": ("PIE", "PIE (towards Jurong, Nanyang F/O)"),
    "6703": ("PIE", "PIE (Tuas, towards Jurong)"),
    "6704": ("PIE", "PIE (Jurong, towards Tuas)"),
    "6705": ("PIE", "PIE (Bukit Batok, towards city)"),
    "6706": ("PIE", "PIE (towards Jurong, Hong Kah)"),
    "6708": ("PIE", "PIE (Bukit Timah, towards Changi)"),
    "6710": ("PIE", "PIE (Clementi, towards Changi)"),
    "6711": ("PIE", "PIE (Mount Pleasant, towards Changi)"),
    "6712": ("PIE", "PIE (towards Jurong, Thomson)"),
    "6713": ("PIE", "PIE (Adam Rd, towards Changi)"),
    "6714": ("PIE", "PIE (Mount Pleasant, towards Jurong)"),
    "6715": ("PIE", "PIE (Kim Keat, towards Jurong)"),
    "6716": ("PIE", "PIE (Woodsville, towards Jurong)"),
    # --- Seletar / Tampines Expressways (SLE / TPE) ---
    "9701": ("SLE", "SLE (Woodlands Ave 2, towards SLE)"),
    "9702": ("SLE", "SLE (Ulu Sembawang, towards TPE)"),
    "9703": ("SLE", "SLE (Lentor F/O, towards TPE)"),
    "9704": ("SLE", "SLE (Mandai Rd, towards TPE)"),
    "9705": ("SLE", "SLE (Mandai Rd, towards Woodlands)"),
    "9706": ("SLE", "SLE (Marsiling, towards Woodlands)"),
    "7791": ("TPE", "TPE (towards SLE, Seletar F/O)"),
    "7793": ("TPE", "TPE (towards PIE, Upper Changi)"),
    "7794": ("TPE", "TPE (Punggol Flyover, towards PIE)"),
    "7795": ("TPE", "TPE (towards KPE, Sengkang)"),
    "7796": ("TPE", "TPE (towards SLE, TPE/KPE exit)"),
    "7797": ("TPE", "TPE (towards PIE, Tampines)"),
    "7798": ("TPE", "TPE (towards SLE, Tampines)"),
    # --- Ayer Rajah Expressway (AYE) ---
    "4701": ("AYE", "AYE (towards city, Alexandra)"),
    "4702": ("AYE", "AYE (Keppel Viaduct)"),
    "4703": ("AYE", "AYE (towards Jurong, Tuas West Rd)"),
    "4704": ("AYE", "AYE (towards MCE, Lower Delta)"),
    "4705": ("AYE", "AYE (near NUS, towards Clementi)"),
    "4706": ("AYE", "AYE (towards MCE, Telok Blangah)"),
    "4707": ("AYE", "AYE (Clementi Ave 6, towards Tuas)"),
    "4708": ("AYE", "AYE (towards city, Clementi)"),
    "4709": ("AYE", "AYE (towards Jurong, Jln Ahmad Ibrahim)"),
    "4710": ("AYE", "AYE (towards Tuas, Pandan Gardens)"),
    "4712": ("AYE", "AYE (towards Tuas, Jurong Town Hall)"),
    "4713": ("AYE", "AYE (towards city, Tuas Ave)"),
    "4714": ("AYE", "AYE (towards Tuas, Benoi)"),
    "4716": ("AYE", "AYE (towards Tuas, Boon Lay)"),
}

UNKNOWN = "OTHER"


def friendly_name(camera_id: str) -> str:
    """Return the descriptive location for a camera, or a safe fallback."""
    entry = CAMERA_NAMES.get(camera_id)
    return entry[1] if entry else f"Camera {camera_id}"


def expressway(camera_id: str) -> str:
    """Return the expressway/zone code for a camera, or ``OTHER``."""
    entry = CAMERA_NAMES.get(camera_id)
    return entry[0] if entry else UNKNOWN
