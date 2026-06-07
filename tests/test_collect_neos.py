from src.collect_neos import rows_from_page

PAGE = {
    "near_earth_objects": [
        {  # usable
            "designation": "1620",
            "name": "1620 Geographos",
            "absolute_magnitude_h": 15.27,
            "estimated_diameter": {"kilometers": {
                "estimated_diameter_min": 2.34, "estimated_diameter_max": 5.24}},
            "is_potentially_hazardous_asteroid": True,
            "close_approach_data": [{
                "relative_velocity": {"kilometers_per_second": "11.76"},
                "miss_distance": {"astronomical": "0.033"},
                "orbiting_body": "Earth",
            }],
        },
        {  # no close_approach_data -> skipped
            "designation": "X",
            "name": "X",
            "absolute_magnitude_h": 25.0,
            "estimated_diameter": {"kilometers": {
                "estimated_diameter_min": 0.01, "estimated_diameter_max": 0.02}},
            "is_potentially_hazardous_asteroid": False,
            "close_approach_data": [],
        },
    ]
}


def test_rows_from_page_skips_unusable_objects():
    rows = rows_from_page(PAGE)
    assert len(rows) == 1
    assert rows[0]["name"] == "1620 Geographos"
    assert rows[0]["is_potentially_hazardous_asteroid"] == 1
