"""Generate a realistic aerospace requirements dataset for ReqCluster.

Produces ~140 functional requirements across eight subsystems with intentional
dependency structure (explicit REQ cross-references, sequential preconditions,
and producer/consumer data links) so the clustering, dependency-tree, and MBSE
export features all have meaningful structure to discover.

Deterministic: running it again yields the same CSV.

Usage:
    python data/generate_aerospace_dataset.py
"""

from __future__ import annotations

import csv
import os

# (module, section, [requirement templates]). {ref} is filled with a prior REQ id.
DOMAINS = [
    ("Thermal", "Temperature Control", [
        "The thermal management unit shall measure baseplate temperature at no less than 5 Hz.",
        "The cooling fan shall activate once the baseplate temperature reported in {ref} exceeds 70 degrees Celsius.",
        "The system shall maintain the avionics bay temperature below 55 degrees Celsius during nominal operation.",
        "The thermal controller shall use the temperature measurement to compute a fan duty cycle.",
        "The heater element shall raise the battery temperature above -10 degrees Celsius prior to charging.",
        "The system shall log a thermal fault when the temperature gradient exceeds 8 degrees Celsius per minute.",
        "The radiator shall dissipate at least 120 watts of waste heat at the maximum design load.",
        "The thermal unit shall provide a temperature telemetry frame every 200 milliseconds.",
    ]),
    ("Power", "Electrical Distribution", [
        "The power distribution unit shall provide a regulated 28 volt bus within plus or minus 2 percent.",
        "The battery management system shall report state of charge with an accuracy of 3 percent.",
        "The avionics shall require the regulated 28 volt bus defined in {ref} to operate.",
        "The power controller shall disconnect a load drawing more than 15 amps within 50 milliseconds.",
        "The solar array shall generate at least 300 watts at beginning of life.",
        "The system shall log an undervoltage event when the main bus drops below 24 volts.",
        "The power unit shall provide an isolated 5 volt rail for the sensor suite.",
        "The battery shall supply 8 hours of runtime at the nominal load profile.",
    ]),
    ("Avionics", "Communication", [
        "The communication subsystem shall transmit telemetry to the ground station at 1 Mbps.",
        "The transceiver shall acquire carrier lock within 2 seconds after power-on.",
        "The data handler shall use the telemetry frames produced by {ref} for downlink.",
        "The system shall encrypt all command uplink traffic using AES-256.",
        "The communication unit shall retransmit a frame after a 100 millisecond acknowledgement timeout.",
        "The antenna controller shall point the high-gain antenna within 0.5 degrees of the ground station.",
        "The receiver shall reject out-of-band interference exceeding 40 dB.",
        "The system shall buffer up to 64 megabytes of telemetry during loss of signal.",
    ]),
    ("Structural", "Mechanical Reliability", [
        "The primary structure shall survive a 15 g mechanical shock without permanent deformation.",
        "The deployment mechanism shall release the solar array once the separation signal in {ref} is received.",
        "The chassis shall maintain alignment of the optical bench within 50 microns under thermal load.",
        "The fasteners shall be torqued to the values specified in the assembly procedure.",
        "The structure shall withstand a random vibration profile of 14.1 g RMS.",
        "The hinge assembly shall complete deployment within 4 seconds of actuation.",
        "The system shall provide a mechanical interface compliant with the launch vehicle adapter.",
        "The bracket shall constrain the reaction wheel to a first natural frequency above 120 Hz.",
    ]),
    ("Control", "Flight Software", [
        "The flight software shall execute the control loop at a fixed rate of 50 Hz.",
        "The autopilot shall use the attitude estimate provided by {ref} to compute actuator commands.",
        "The software shall enter safe mode after detecting three consecutive watchdog timeouts.",
        "The control unit shall limit the commanded torque to 0.2 newton meters.",
        "The system shall complete boot and self-test within 12 seconds of power application.",
        "The software shall record the last 256 fault events in non-volatile memory.",
        "The scheduler shall guarantee the control task meets its deadline 99.9 percent of the time.",
        "The software shall validate uplinked commands against the allowed command table before execution.",
    ]),
    ("Navigation", "Guidance", [
        "The navigation filter shall estimate attitude with an accuracy of 0.1 degrees, 1 sigma.",
        "The star tracker shall provide an attitude quaternion at 4 Hz.",
        "The guidance module shall use the position fix from {ref} to compute a maneuver plan.",
        "The GPS receiver shall provide a position fix with an accuracy of 10 meters.",
        "The system shall propagate the orbit state during GPS outages for up to 30 minutes.",
        "The navigation unit shall flag a degraded solution when the covariance exceeds the configured bound.",
        "The inertial measurement unit shall sample angular rate at 200 Hz.",
        "The guidance module shall recompute the maneuver after each new position fix.",
    ]),
    ("Safety", "Fault Management", [
        "The system shall isolate a failed reaction wheel within one control cycle.",
        "The fault manager shall raise an alarm when the constraint defined in {ref} is violated.",
        "The system shall provide a hardware inhibit on all pyrotechnic actuators until armed.",
        "The watchdog shall reset the processor if not serviced within 500 milliseconds.",
        "The safety monitor shall verify two independent signals before commanding a deployment.",
        "The system shall transition to safe mode upon detecting a loss of attitude knowledge.",
        "The fault manager shall log the time and cause of every safe-mode entry.",
        "The system shall require ground confirmation before exiting safe mode.",
    ]),
    ("Propulsion", "Thrust Control", [
        "The propulsion controller shall regulate chamber pressure to within 3 percent of the setpoint.",
        "The thruster shall produce 1 newton of thrust at the nominal feed pressure.",
        "The propulsion unit shall use the maneuver plan from {ref} to schedule burns.",
        "The valve driver shall open the latch valve within 20 milliseconds of command.",
        "The system shall inhibit thruster firing when the tank pressure is below 5 bar.",
        "The controller shall measure propellant temperature for feed-pressure compensation.",
        "The propulsion unit shall report total impulse consumed after each burn.",
        "The system shall close all valves and report a fault on detection of a pressure leak.",
    ]),
]


def build_rows():
    rows = []
    idx = 1
    # First pass: assign ids so {ref} can point to an earlier requirement.
    last_id_in_domain = {}
    for di, (module, section, templates) in enumerate(DOMAINS):
        for ti, template in enumerate(templates):
            req_id = f"REQ-{idx:03d}"
            # Reference the first requirement of this domain for intra-domain links.
            ref = last_id_in_domain.get(module, f"REQ-{idx - 1:03d}" if idx > 1 else req_id)
            text = template.replace("{ref}", ref)
            rows.append({"id": req_id, "text": text, "module": module, "section": section})
            if ti == 0:
                last_id_in_domain[module] = req_id
            idx += 1
    return rows


def main():
    rows = build_rows()
    out_path = os.path.join(os.path.dirname(__file__), "aerospace_requirements.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "module", "section"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} requirements to {out_path}")


if __name__ == "__main__":
    main()
