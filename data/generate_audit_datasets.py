"""Generate deterministic audit datasets for ReqCluster.

The files exercise UAV/FMS-style ingestion, quality edge cases, and a 600-row
stress upload path without depending on external data.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent

DOMAINS = [
    ("Command Authentication", "Command Security", [
        "authenticate uplink commands before execution",
        "reject commands with stale sequence counters",
        "log operator identity for every accepted command",
        "validate command freshness against the configured command timer",
        "isolate command buffers after authentication failure",
    ]),
    ("Mission Management", "Mission Upload", [
        "validate uploaded waypoints before storing the mission plan",
        "reject mission upload when waypoint count exceeds the configured limit",
        "preserve the active mission until the new mission passes validation",
        "set the mission_ready flag after all waypoints are verified",
        "clear the upload buffer after a failed mission transfer",
    ]),
    ("Navigation", "Sensor Fusion", [
        "compute navigation state from GPS and inertial reference data",
        "enter inertial fallback when GPS validity is lost",
        "raise navigation_degraded when position uncertainty exceeds the configured threshold",
        "publish the selected navigation source to guidance",
        "restore GPS navigation only after validity remains stable for the configured timer",
    ]),
    ("Guidance", "Waypoint Guidance", [
        "select the next waypoint after crossing the active waypoint acceptance radius",
        "compute lateral guidance commands from current track and desired path",
        "compute vertical guidance commands from target altitude and climb constraints",
        "hold the active waypoint when navigation validity is degraded",
        "output guidance mode status to telemetry",
    ]),
    ("Flight Control", "Automatic Mode", [
        "allow automatic mode entry only when navigation and guidance are valid",
        "exit automatic mode when pilot override is detected",
        "limit lateral control command rate to the configured bound",
        "limit vertical control command rate to the configured bound",
        "inhibit arming when any flight-control self-test fails",
    ]),
    ("Power Management", "Battery Readiness", [
        "estimate remaining battery capacity from voltage current and temperature",
        "inhibit mission launch when battery state of charge is below the configured threshold",
        "inhibit mission launch when battery temperature exceeds the configured limit",
        "publish battery readiness status to mission management",
        "record battery health trend data for maintenance",
    ]),
    ("Communications", "Lost Link", [
        "detect lost link when heartbeat messages are absent for the configured timeout",
        "enter lost-link behavior after the lost_link timer expires",
        "retry command acknowledgement within three seconds",
        "queue telemetry packets while the downlink is unavailable",
        "flush queued telemetry after link recovery",
    ]),
    ("Payload Control", "Payload Motion", [
        "reject payload slew commands that exceed configured motion limits",
        "stop payload motion when vehicle attitude is outside the safe envelope",
        "publish payload pointing status to telemetry",
        "hold payload position during automatic mode exit",
        "log payload constraint violations for maintenance review",
    ]),
    ("Health Monitoring", "Fault Response", [
        "run power-on self-test before enabling mission functions",
        "aggregate subsystem health flags into a vehicle readiness output",
        "raise emergency response when critical health flags are active",
        "store health-monitor events in the event log",
        "clear transient health flags only after the configured debounce timer",
    ]),
    ("Software Update", "Update Safety", [
        "verify software update package signature before installation",
        "reject update packages with an invalid rollback marker",
        "preserve the previous software image until post-update self-test passes",
        "block mission launch while update installation is in progress",
        "record software update result in the maintenance log",
    ]),
    ("Telemetry", "Reporting", [
        "generate telemetry frames containing navigation guidance power and health data",
        "tag telemetry frames with source timestamp and sequence number",
        "drop telemetry frames that exceed the configured payload size",
        "publish emergency telemetry immediately after emergency response entry",
        "report active control mode and mission phase to the ground station",
    ]),
    ("FMS", "LNAV VNAV Fuel Displays", [
        "compute LNAV path transitions from active flight-plan legs",
        "compute VNAV targets from altitude constraints and descent path geometry",
        "integrate fuel remaining with predicted time to destination",
        "display active flight-plan leg and lateral deviation to the cockpit display",
        "record flight-plan edits in the maintenance log",
    ]),
]

CONTRADICTION_ROWS = [
    ("CB-001", "The system shall accept commands with invalid sequence numbers.", "Command Authentication", "Contradiction"),
    ("CB-002", "The system shall reject commands with invalid sequence numbers.", "Command Authentication", "Contradiction"),
    ("CB-003", "The vehicle shall allow mission launch when battery temperature exceeds the configured limit.", "Power Management", "Contradiction"),
    ("CB-004", "The vehicle shall inhibit mission launch when battery temperature exceeds the configured limit.", "Power Management", "Contradiction"),
    ("CB-005", "The UAV shall maintain altitude within configured tolerance.", "Guidance", "Boundary"),
    ("CB-006", "The UAV shall maintain altitude.", "Guidance", "Boundary"),
    ("CB-007", "The system shall retry communication.", "Communications", "Boundary"),
    ("CB-008", "The system shall retry communication within 3 seconds.", "Communications", "Boundary"),
    ("CB-009", "The telemetry service shall archive messages for the configured retention period.", "Telemetry", "Boundary"),
    ("CB-010", "The telemetry service shall archive messages for 30 days.", "Telemetry", "Boundary"),
    ("CB-011", "The update manager shall verify update signatures before installation.", "Software Update", "Security"),
    ("CB-012", "The update manager shall install update packages before signature verification.", "Software Update", "Contradiction"),
]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "text", "module", "section", "page", "source_doc", "requirement_type", "criticality"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def medium_rows() -> list[dict[str, object]]:
    rows = []
    idx = 1
    for module, section, stems in DOMAINS:
        for stem in stems:
            for variant in (
                f"The {module.lower()} function shall {stem}.",
                f"The system shall maintain the {section.lower()} state when required to {stem}.",
            ):
                rows.append({
                    "id": f"UAV-FMS-{idx:03d}",
                    "text": variant,
                    "module": module,
                    "section": section,
                    "page": 1 + idx // 4,
                    "source_doc": "uav_fms_phase2_audit",
                    "requirement_type": "functional",
                    "criticality": "high" if module in {"Command Authentication", "Flight Control", "Power Management", "Software Update"} else "medium",
                })
                idx += 1
    return rows


def contradiction_rows() -> list[dict[str, object]]:
    rows = []
    for idx, (req_id, text, module, section) in enumerate(CONTRADICTION_ROWS, start=1):
        rows.append({
            "id": req_id,
            "text": text,
            "module": module,
            "section": section,
            "page": idx,
            "source_doc": "contradiction_boundary_audit",
            "requirement_type": "functional" if section != "Security" else "non_functional",
            "criticality": "high",
        })
    return rows


def stress_rows(total: int = 600) -> list[dict[str, object]]:
    base = medium_rows()
    rows = []
    for idx in range(total):
        template = base[idx % len(base)]
        page = 1 + idx // 4
        rows.append({
            "id": f"STRESS-{idx + 1:04d}",
            "text": f"{template['text']} Stress scenario page {page} shall preserve traceability for buffer {(idx % 17) + 1}.",
            "module": template["module"],
            "section": template["section"],
            "page": page,
            "source_doc": f"daily_uav_fms_batch_{1 + idx // 150}",
            "requirement_type": template["requirement_type"],
            "criticality": template["criticality"],
        })
    return rows


def main() -> None:
    write_rows(ROOT / "uav_fms_phase2_audit_requirements.csv", medium_rows())
    write_rows(ROOT / "contradiction_boundary_audit_requirements.csv", contradiction_rows())
    write_rows(ROOT / "uav_fms_stress_600_requirements.csv", stress_rows(600))


if __name__ == "__main__":
    main()
