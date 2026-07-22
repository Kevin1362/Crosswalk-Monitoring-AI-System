from __future__ import annotations

from pathlib import Path
import shutil
import sys

RUNNER = Path(__file__).resolve().parent / "winter_scenario_runner.py"


def fail(message: str) -> None:
    print(f"\nERROR: {message}")
    sys.exit(1)


if not RUNNER.exists():
    fail(
        "winter_scenario_runner.py was not found in this folder. "
        "Put this repair script in the same folder as the runner."
    )

text = RUNNER.read_text(encoding="utf-8")
backup = RUNNER.with_name("winter_scenario_runner_before_distance_patch.py")
shutil.copy2(RUNNER, backup)

# Add calibration constant.
if "PIXELS_PER_METER" not in text:
    marker = "CROSSWALK = (565, 365, 720, 650)"
    position = text.find(marker)
    if position == -1:
        fail("Could not find the CROSSWALK definition.")
    line_end = text.find("\n", position)
    if line_end == -1:
        line_end = len(text)
    insertion = (
        "\n# Prototype calibration: 155 pixels represents about 4 metres.\n"
        "PIXELS_PER_METER = 155 / 4.0\n"
    )
    text = text[:line_end] + insertion + text[line_end:]

# Add distance function.
if "def vehicle_distance_to_crosswalk_m(" not in text:
    marker = "\ndef calculate_risk("
    if marker not in text:
        fail("Could not find calculate_risk().")
    function_source = (
        "\n\ndef vehicle_distance_to_crosswalk_m(car_box) -> float:\n"
        "    \"\"\"Estimate vehicle distance from the crosswalk for this simulation.\"\"\"\n"
        "    if not car_box:\n"
        "        return 0.0\n"
        "\n"
        "    crosswalk_x1, _, crosswalk_x2, _ = CROSSWALK\n"
        "    if car_box[0] > crosswalk_x2:\n"
        "        distance_px = car_box[0] - crosswalk_x2\n"
        "    elif car_box[2] < crosswalk_x1:\n"
        "        distance_px = crosswalk_x1 - car_box[2]\n"
        "    else:\n"
        "        distance_px = 0.0\n"
        "\n"
        "    return max(0.0, distance_px / PIXELS_PER_METER)\n"
    )
    text = text.replace(marker, function_source + marker, 1)

# Calculate the value after detecting the car and pedestrian.
calculation = (
    "        distance_to_crosswalk_m = "
    "vehicle_distance_to_crosswalk_m(car_box)\n"
)
if calculation not in text:
    marker = "        car_box, person_box = detect_simulated_objects(frame)\n"
    if marker not in text:
        fail("Could not find the simulated object detection line.")
    text = text.replace(marker, marker + calculation, 1)

# Replace the hardcoded database value.
dynamic_value = "distance_to_crosswalk_m=round(distance_to_crosswalk_m, 1)"
if dynamic_value not in text:
    old = "                distance_to_crosswalk_m=0.0,\n"
    if old not in text:
        fail("Could not find distance_to_crosswalk_m=0.0 in the logger call.")
    text = text.replace(
        old,
        "                distance_to_crosswalk_m=round(distance_to_crosswalk_m, 1),\n",
        1,
    )

# Add the distance to notes, when the original notes line is present.
if "vehicle-to-crosswalk=" not in text:
    old_notes = '                notes="Controlled simulation: " + "; ".join(reasons),\n'
    if old_notes in text:
        new_notes = (
            "                notes=(\n"
            "                    \"Controlled simulation: \"\n"
            "                    + \"; \".join(reasons)\n"
            "                    + f\"; vehicle-to-crosswalk={distance_to_crosswalk_m:.1f}m\"\n"
            "                ),\n"
        )
        text = text.replace(old_notes, new_notes, 1)

RUNNER.write_text(text, encoding="utf-8")

# Verify.
updated = RUNNER.read_text(encoding="utf-8")
checks = {
    "Distance function": "def vehicle_distance_to_crosswalk_m(" in updated,
    "Per-frame calculation": calculation.strip() in updated,
    "Dynamic database value": dynamic_value in updated,
}

print("\nDistance patch result")
print("---------------------")
for label, result in checks.items():
    print(f"{label}: {result}")

print(f"\nBackup created: {backup.name}")
if not all(checks.values()):
    fail("The runner was not patched completely.")

print("\nSUCCESS: winter_scenario_runner.py was updated.")
print("\nRun this next:")
print('python winter_scenario_runner.py --source ".\\videos\\guidekaro_winter_crosswalk_scenario.mp4" --loop')