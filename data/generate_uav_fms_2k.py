"""Generate a 2,000-requirement UAV Flight-Management-System dataset.

Designed so clustering produces clean, well-separated groups (near-zero noise):
20 subsystems, each with its OWN domain vocabulary and several sentence templates,
so requirements within a subsystem are semantically tight and distinct between
subsystems - no shared boilerplate. A small set of deliberately vague / cross-
cutting requirements is appended to show up as the "odd ones out" (noise).

Output: data/uav_fms_2k_requirements.csv  (columns: id, text, module, section)
"""

import csv
import os
import random

random.seed(2026)

# Each subsystem: a section name and a list of distinct action phrases. Actions
# may contain {n}/{ms}/{hz}/{v} placeholders filled with realistic values.
SUBSYSTEMS = {
    "Navigation": ("Position Estimation", [
        "estimate aircraft position from GPS and inertial measurements",
        "compute ground speed and track angle from the navigation solution",
        "maintain a horizontal position accuracy of {n} meters CEP",
        "switch to inertial dead reckoning when GPS validity is lost",
        "output the navigation state in the WGS84 reference frame",
        "reject GPS fixes that fail the receiver autonomous integrity check",
        "blend GPS and inertial data in the navigation filter at {hz} Hz",
        "raise a navigation warning when position uncertainty exceeds {n} meters",
        "recover a valid GPS fix within {ms} milliseconds of signal reacquisition",
        "propagate dead-reckoned position from last-known velocity and heading",
        "report the number of tracked GPS satellites to the ground station",
        "limit navigation drift to {n} meters per minute without GPS",
    ]),
    "Attitude": ("Attitude and Heading", [
        "estimate vehicle attitude from gyroscope and accelerometer data",
        "compute the roll, pitch, and yaw angles at {hz} Hz",
        "correct gyroscope bias using the magnetometer heading reference",
        "represent vehicle orientation as a normalized attitude quaternion",
        "detect and reject magnetometer readings corrupted by hard-iron effects",
        "limit attitude estimation error to {n} degrees during coordinated turns",
        "reinitialize the attitude filter after an inertial sensor reset",
        "output heading relative to magnetic north within {n} degrees",
        "compensate angular rate measurements for temperature-dependent drift",
        "flag an attitude fault when gyro and accelerometer estimates diverge",
        "maintain a valid attitude solution through {n} g of acceleration",
    ]),
    "Guidance": ("Trajectory Guidance", [
        "compute steering commands to follow the active waypoint path",
        "minimize cross-track error relative to the planned route",
        "sequence to the next waypoint when the acceptance radius is reached",
        "command a loiter pattern of radius {n} meters about a hold point",
        "generate a smooth turn between consecutive route legs",
        "limit commanded bank angle to {n} degrees during path tracking",
        "recompute guidance when the mission plan is updated in flight",
        "hold the commanded altitude within {n} meters during cruise",
        "blend course and altitude guidance into a single trajectory command",
        "command a direct-to leg toward an operator-selected waypoint",
        "reduce ground speed when approaching a tight turn radius",
    ]),
    "FlightControl": ("Stability and Control", [
        "stabilize the airframe about the commanded attitude setpoint",
        "deflect the ailerons to track the roll-rate command",
        "deflect the elevator to track the pitch-rate command",
        "deflect the rudder to coordinate turns and damp yaw",
        "apply pitch-rate damping using the inner control loop at {hz} Hz",
        "limit control-surface commands to the configured deflection envelope",
        "schedule control gains as a function of airspeed",
        "suppress structural oscillation through a notch filter",
        "revert to a degraded control law when a sensor input is invalid",
        "hold wings level when the operator releases manual control",
        "limit the commanded load factor to {n} g",
    ]),
    "Propulsion": ("Engine and Thrust", [
        "regulate engine RPM to the commanded throttle setting",
        "limit thrust output to protect the propulsion system from overspeed",
        "ramp throttle changes no faster than {n} percent per second",
        "monitor electronic speed controller temperature during high-thrust phases",
        "command a controlled engine shutdown on receipt of the stop signal",
        "detect a propeller stall from the RPM and current signature",
        "maintain commanded thrust within {n} percent during steady cruise",
        "inhibit throttle increase when the propulsion fault flag is set",
        "log motor current and RPM to the propulsion health record",
        "restart the engine automatically after a transient undervoltage",
    ]),
    "Fuel": ("Fuel Management", [
        "measure remaining fuel quantity in each tank",
        "compute fuel flow rate and instantaneous endurance",
        "command the transfer pump to balance fuel between tanks",
        "raise a low-fuel caution when the reserve threshold is reached",
        "estimate range remaining from current fuel and ground speed",
        "detect a fuel-flow anomaly inconsistent with throttle position",
        "isolate a tank after a leak is detected by the quantity trend",
        "report total fuel consumed for the current mission",
        "inhibit aggressive maneuvers when fuel falls below {n} percent",
        "schedule fuel transfer to keep the center of gravity within limits",
    ]),
    "Power": ("Electrical Power", [
        "monitor main battery voltage and pack current",
        "estimate the battery state of charge from voltage and load",
        "distribute power across the avionics and payload buses",
        "shed non-essential loads when the battery reaches {n} percent",
        "raise a low-voltage warning below {v} volts on the main bus",
        "switch to the backup battery on primary power loss",
        "limit total bus current to protect the distribution harness",
        "report remaining flight time from the state of charge estimate",
        "detect a cell imbalance from the per-cell voltage telemetry",
        "log battery temperature and current to the power health record",
    ]),
    "Datalink": ("Communications Link", [
        "maintain the command uplink and telemetry downlink with the ground station",
        "report received signal strength on the primary radio link",
        "switch to the secondary radio when uplink quality degrades",
        "limit downlink latency to {ms} milliseconds for control messages",
        "throttle the telemetry data rate to fit the available bandwidth",
        "detect loss of link after {ms} milliseconds without a valid uplink frame",
        "reacquire the datalink automatically after a brief signal dropout",
        "select the operating frequency from the configured channel plan",
        "report link margin and bit error rate to the operator",
        "buffer outbound telemetry during a temporary downlink outage",
    ]),
    "Security": ("Command Authentication", [
        "authenticate every uplink command before execution",
        "reject commands carrying a stale or replayed sequence counter",
        "verify the integrity of each command message with a keyed checksum",
        "decrypt uplink messages using the configured session key",
        "log the operator identity associated with every accepted command",
        "discard commands that fail authentication and raise a security alert",
        "rotate the session key after the configured number of messages",
        "isolate the command buffer after repeated authentication failures",
        "validate command freshness against the synchronized command timer",
        "deny privileged commands when the link is not cryptographically secured",
    ]),
    "Telemetry": ("Telemetry and Logging", [
        "assemble the telemetry frame from the current vehicle state",
        "downlink telemetry at the configured frame rate",
        "timestamp each telemetry record with the synchronized mission clock",
        "record full-rate flight data to the onboard log",
        "prioritize critical status fields when downlink bandwidth is limited",
        "compress logged data to extend onboard storage capacity",
        "report onboard log storage remaining as a percentage",
        "include a frame sequence number to detect dropped telemetry",
        "flush the flight log to non-volatile storage before shutdown",
        "replay buffered telemetry after a downlink outage is cleared",
    ]),
    "Payload": ("Payload and Imaging", [
        "stabilize the camera gimbal against airframe motion",
        "slew the gimbal to the operator-commanded pointing angle",
        "capture imagery at the commanded frame rate and resolution",
        "hold the gimbal line of sight on a geo-referenced target",
        "limit gimbal slew rate to {n} degrees per second",
        "tag each captured image with position and attitude metadata",
        "command the camera zoom to the requested field of view",
        "inhibit payload power when the battery is below {n} percent",
        "report gimbal pointing angles and payload status in telemetry",
        "return the gimbal to a stowed position before landing",
    ]),
    "Mission": ("Mission Management", [
        "validate an uploaded mission plan before storing it",
        "reject a mission upload when the waypoint count exceeds the limit",
        "preserve the active mission until a new plan passes validation",
        "set the mission-ready flag once all waypoints are verified",
        "sequence mission phases from takeoff through recovery",
        "resume the mission from the current waypoint after a manual override",
        "abort the mission and return to base on the operator command",
        "store mission parameters in non-volatile memory across power cycles",
        "report mission progress as the active waypoint index",
        "clear the upload buffer after a failed mission transfer",
    ]),
    "SensorFusion": ("State Estimation", [
        "fuse inertial, GPS, and air-data measurements in the state estimator",
        "propagate the estimator covariance between measurement updates",
        "reject measurement outliers using the innovation gate",
        "estimate wind velocity from the difference of ground and air speed",
        "update the Kalman filter at the sensor measurement rate",
        "downweight a sensor whose noise statistics exceed the model",
        "reinitialize the estimator after a sustained measurement dropout",
        "bound the estimated state error within the {n}-sigma envelope",
        "blend redundant air-data sensors into a single airspeed estimate",
        "publish the fused vehicle state to the guidance and control loops",
    ]),
    "Health": ("Health Monitoring", [
        "execute a power-on built-in test of all critical sensors",
        "run a continuous background built-in test during flight",
        "detect and annunciate subsystem faults to the ground station",
        "reset the watchdog timer within each control frame",
        "record fault codes with a timestamp to the diagnostic log",
        "isolate a failed sensor and select its redundant counterpart",
        "report overall vehicle health status at {hz} Hz",
        "inhibit takeoff when a critical built-in test fails",
        "track the number of resets since the last maintenance action",
        "verify checksum integrity of the loaded flight software at startup",
    ]),
    "Geofence": ("Geofencing and Airspace", [
        "enforce the configured geofence boundary during all flight phases",
        "command a corrective maneuver when approaching a no-fly zone",
        "limit altitude to the configured ceiling for the operating area",
        "raise a geofence breach alert when the boundary is crossed",
        "prevent waypoint acceptance outside the permitted airspace",
        "command return-to-home when a hard geofence limit is reached",
        "load airspace boundaries from the mission configuration",
        "maintain a {n}-meter buffer inside the geofence boundary",
        "report distance to the nearest geofence edge in telemetry",
        "inhibit manual commands that would violate the airspace limit",
    ]),
    "Autoland": ("Automatic Landing", [
        "capture the glideslope on the configured approach path",
        "command the flare maneuver at the computed decision height",
        "align the vehicle with the runway centerline during approach",
        "reduce airspeed to the commanded approach speed",
        "execute a go-around when approach criteria are not met",
        "command return-to-home and automatic landing on loss of link",
        "deploy landing gear before reaching the flare altitude",
        "limit sink rate to {n} meters per second at touchdown",
        "disarm propulsion after a confirmed touchdown",
        "abort the landing if crosswind exceeds the configured limit",
    ]),
    "Actuator": ("Servo and Actuation", [
        "drive each control servo to its commanded deflection",
        "monitor servo position feedback against the command",
        "detect a jammed actuator from position-error persistence",
        "limit servo slew rate to protect the actuation mechanism",
        "report actuator current and temperature in telemetry",
        "fail over to a redundant actuator on a detected servo fault",
        "calibrate actuator neutral position during ground checks",
        "inhibit actuator motion when the safety interlock is engaged",
        "hold the last valid command when actuator feedback is lost",
        "compensate servo response for supply-voltage variation",
    ]),
    "Thermal": ("Thermal Management", [
        "monitor the temperature of the avionics and propulsion bays",
        "command active cooling when a bay exceeds its thermal limit",
        "derate propulsion output to prevent motor overheating",
        "raise an over-temperature warning above {n} degrees Celsius",
        "energize the heater to keep the battery above its minimum temperature",
        "log thermal trends for each monitored zone",
        "throttle payload activity when its sensor approaches the thermal limit",
        "verify thermal sensor plausibility against neighboring zones",
        "maintain avionics temperature within the qualified operating band",
        "inhibit high-power modes during a sustained over-temperature condition",
    ]),
    "GCS": ("Ground Control Interface", [
        "present vehicle state and health on the operator display",
        "accept command entry only from an authenticated operator console",
        "annunciate cautions and warnings to the operator in priority order",
        "confirm safety-critical commands with an explicit operator acknowledgement",
        "display the active mission plan and vehicle position on the moving map",
        "log every operator command with a timestamp and console identity",
        "throttle nuisance alerts to avoid operator overload",
        "synchronize the displayed clock with the vehicle mission time",
        "provide a single-action command to trigger return-to-home",
        "indicate datalink quality and remaining flight time to the operator",
    ]),
    "Failsafe": ("Emergency and Failsafe", [
        "enter a defined safe state on loss of the command link",
        "execute the configured contingency plan for a critical failure",
        "command return-to-home automatically after {ms} milliseconds without uplink",
        "transition to a controlled descent when propulsion is lost",
        "select the nearest safe recovery point on an emergency declaration",
        "inhibit mission commands while a failsafe action is in progress",
        "annunciate the active failsafe mode to the ground station",
        "hold a stable orbit at the current position during link recovery",
        "arm the recovery parachute when an unrecoverable attitude is detected",
        "preserve flight-critical state through a failsafe-triggered reset",
    ]),
}

# Rich, smooth variety so each subsystem forms one continuous dense cloud, which
# minimizes HDBSCAN boundary noise. Distinctness comes from per-subsystem vocab.
CONDITIONS = [
    "", "", "",
    "during all flight phases",
    "within {ms} milliseconds",
    "without operator intervention",
    "under nominal operating conditions",
    "while the vehicle is airborne",
    "across the full operating envelope",
    "when commanded by the mission plan",
]

TEMPLATES = [
    "{subject} shall {action}{cond}.",
    "{subject} shall {action}{cond}.",
    "When required, {subject} shall {action}{cond}.",
    "The system shall ensure that {subject} can {action}{cond}.",
    "{subject} shall be able to {action}{cond}.",
]

SUBJECTS_FOR = lambda key: [
    f"the {key.lower()} subsystem",
    f"the {key.lower()} function",
    "the flight management system",
]

# Deliberately odd / vague / cross-cutting requirements -> should fall out as noise.
ODD = [
    "The product shall be maintainable by qualified personnel.",
    "Documentation shall be provided in accordance with the program plan.",
    "The system should be user friendly and intuitive to operate.",
    "All units shall be painted in the standard livery before delivery.",
    "The vendor shall deliver the spares package within the agreed schedule.",
    "Color choices for the operator interface shall follow corporate branding.",
    "The system shall comply with applicable regulations where practicable.",
    "Training materials shall be made available to the operating squadron.",
    "The contractor shall hold a monthly progress review with the customer.",
    "Packaging shall protect the units during transport and storage.",
    "The warranty shall cover defects in materials and workmanship.",
    "The system shall be cost effective over its service life.",
]


def _fill(s: str) -> str:
    return (s.replace("{n}", str(random.choice([2, 3, 5, 10, 15, 20, 30, 50, 70, 85, 95, 120])))
             .replace("{ms}", str(random.choice([20, 50, 100, 200, 500, 1000])))
             .replace("{hz}", str(random.choice([1, 4, 5, 10, 20, 50, 100, 200])))
             .replace("{v}", str(random.choice([18, 22, 24, 42]))))


def main():
    rows = []
    per = 99  # 20 * 99 = 1980, + 12 odd = 1992
    for key, (section, actions) in SUBSYSTEMS.items():
        seen = set()
        subjects = SUBJECTS_FOR(key)
        attempts = 0
        produced = 0
        while produced < per and attempts < per * 40:
            attempts += 1
            action = random.choice(actions)
            cond = random.choice(CONDITIONS)
            template = random.choice(TEMPLATES)
            subject = random.choice(subjects)
            cond_str = f" {cond}" if cond else ""
            text = _fill(template.format(subject=subject, action=action, cond=cond_str))
            text = text[0].upper() + text[1:]
            if text in seen:
                continue
            seen.add(text)
            rows.append({"module": key, "section": section, "text": text})
            produced += 1

    for text in ODD:
        rows.append({"module": "General", "section": "Programmatic", "text": text})

    random.shuffle(rows)
    out = os.path.join(os.path.dirname(__file__), "uav_fms_2k_requirements.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "module", "section"])
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({"id": f"UAV-{i:04d}", "text": r["text"], "module": r["module"], "section": r["section"]})
    print(f"wrote {len(rows)} requirements to {out}")
    print(f"subsystems={len(SUBSYSTEMS)} odd={len(ODD)}")


if __name__ == "__main__":
    main()
