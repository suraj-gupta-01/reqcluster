"""Generate a clean, low-noise UAV Flight-Management-System requirements dataset.

Each subsystem is defined by its OWN vocabulary (components x verbs x objects x
qualifiers). Requirements are built by combining that vocabulary, so:
  - WITHIN a subsystem you get many semantically-varied but on-topic sentences
    (a smooth, dense cloud, not a few repeated blobs), and
  - BETWEEN subsystems the vocabulary is distinct (well-separated clusters).
That keeps HDBSCAN noise low (~5%) and scales to any size.

A small set of vague / programmatic requirements is appended to surface as the
"odd ones out" (genuine noise).

Usage:  python generate_uav_fms.py [N]      (default 2000)
Output: data/uav_fms_<size>_requirements.csv   (columns: id, text, module, section)
"""

import csv
import os
import random
import sys

random.seed(2026)

# 16 distinct subsystems. Each: section + component/verb/object/qualifier word lists.
SUB = {
    "Navigation": {"section": "Position Estimation",
        "comp": ["the navigation filter", "the GPS receiver", "the inertial navigation unit", "the position estimator", "the dead-reckoning module", "the navigation processor"],
        "verb": ["estimate", "compute", "update", "validate", "output", "propagate", "blend", "monitor"],
        "obj": ["the aircraft position", "the ground velocity", "the track angle", "the GPS fix quality", "the inertial drift", "the position uncertainty", "the tracked satellite count", "the navigation solution"],
        "qual": ["in the WGS84 reference frame", "at {hz} Hz", "during GPS outages", "within {n} meters CEP", "without operator input", "across the flight envelope"]},
    "Propulsion": {"section": "Engine and Thrust",
        "comp": ["the propulsion controller", "the electronic speed controller", "the motor governor", "the throttle manager", "the engine monitor", "the thrust regulator"],
        "verb": ["regulate", "limit", "ramp", "command", "monitor", "protect", "report", "stabilize"],
        "obj": ["the engine RPM", "the commanded throttle", "the motor current", "the propeller thrust", "the overspeed condition", "the thrust setpoint", "the controller temperature", "the spin-up sequence"],
        "qual": ["within {n} percent", "no faster than {n} percent per second", "during high-thrust phases", "to protect the drivetrain", "at {hz} Hz", "under load"]},
    "Power": {"section": "Electrical Power",
        "comp": ["the power management unit", "the battery monitor", "the bus controller", "the load manager", "the charge estimator", "the distribution board"],
        "verb": ["monitor", "estimate", "distribute", "shed", "balance", "report", "protect", "switch"],
        "obj": ["the battery voltage", "the pack current", "the state of charge", "the avionics bus", "the cell balance", "the remaining flight time", "the non-essential loads", "the backup battery"],
        "qual": ["below {v} volts", "across both buses", "when charge drops below {n} percent", "at {hz} Hz", "to the backup source", "under fault conditions"]},
    "Fuel": {"section": "Fuel Management",
        "comp": ["the fuel system", "the transfer pump", "the quantity gauge", "the fuel manager", "the tank selector", "the flow sensor"],
        "verb": ["measure", "compute", "transfer", "balance", "report", "isolate", "estimate", "monitor"],
        "obj": ["the remaining fuel quantity", "the fuel flow rate", "the tank balance", "the endurance estimate", "the center of gravity", "the reserve threshold", "a fuel leak", "the consumption total"],
        "qual": ["between tanks", "when reserve is reached", "for the current mission", "within {n} percent", "during cruise", "to keep CG in limits"]},
    "Datalink": {"section": "Communications Link",
        "comp": ["the datalink manager", "the primary radio", "the secondary radio", "the uplink decoder", "the downlink scheduler", "the link monitor"],
        "verb": ["maintain", "report", "switch", "throttle", "reacquire", "select", "buffer", "detect"],
        "obj": ["the command uplink", "the telemetry downlink", "the received signal strength", "the link margin", "the operating frequency", "the bit error rate", "a loss of link", "the available bandwidth"],
        "qual": ["within {ms} milliseconds", "after a signal dropout", "from the channel plan", "to fit the bandwidth", "with the ground station", "at {hz} Hz"]},
    "Security": {"section": "Command Authentication",
        "comp": ["the command authenticator", "the cryptographic module", "the key manager", "the message validator", "the security monitor", "the access controller"],
        "verb": ["authenticate", "reject", "verify", "decrypt", "log", "rotate", "isolate", "deny"],
        "obj": ["every uplink command", "a replayed sequence counter", "the message integrity", "the session key", "the operator identity", "the command buffer", "an unauthenticated command", "the command timer"],
        "qual": ["before execution", "with a keyed checksum", "after {n} messages", "on repeated failures", "when the link is unsecured", "within {ms} milliseconds"]},
    "Payload": {"section": "Payload and Imaging",
        "comp": ["the payload controller", "the camera gimbal", "the imaging sensor", "the gimbal stabilizer", "the payload manager", "the optical head"],
        "verb": ["stabilize", "slew", "capture", "hold", "limit", "tag", "command", "stow"],
        "obj": ["the camera line of sight", "the operator pointing angle", "imagery", "a geo-referenced target", "the gimbal slew rate", "the captured image", "the field of view", "the payload power"],
        "qual": ["against airframe motion", "at the commanded rate", "to {n} degrees per second", "with position metadata", "before landing", "when battery is low"]},
    "Mission": {"section": "Mission Management",
        "comp": ["the mission manager", "the flight plan store", "the waypoint validator", "the mission sequencer", "the upload handler", "the route planner"],
        "verb": ["validate", "reject", "preserve", "sequence", "resume", "abort", "store", "report"],
        "obj": ["the uploaded mission plan", "the waypoint count", "the active mission", "the mission-ready flag", "the mission phases", "the current waypoint", "mission progress", "the upload buffer"],
        "qual": ["before storing", "when the limit is exceeded", "after a manual override", "across power cycles", "on operator command", "from takeoff to recovery"]},
    "Health": {"section": "Health Monitoring",
        "comp": ["the health monitor", "the built-in test", "the diagnostic engine", "the watchdog timer", "the fault manager", "the status reporter"],
        "verb": ["execute", "detect", "annunciate", "reset", "record", "isolate", "report", "inhibit"],
        "obj": ["a power-on self test", "a subsystem fault", "the diagnostic log", "the watchdog", "a failed sensor", "the vehicle health status", "the fault code", "a critical failure"],
        "qual": ["at startup", "during flight", "to the ground station", "within each control frame", "at {hz} Hz", "before takeoff"]},
    "Geofence": {"section": "Geofencing and Airspace",
        "comp": ["the geofence monitor", "the airspace manager", "the boundary checker", "the altitude limiter", "the containment module", "the fence enforcer"],
        "verb": ["enforce", "command", "limit", "raise", "prevent", "load", "maintain", "report"],
        "obj": ["the geofence boundary", "a corrective maneuver", "the altitude ceiling", "a breach alert", "waypoint acceptance", "the airspace boundaries", "a safety buffer", "the distance to the edge"],
        "qual": ["during all flight phases", "near a no-fly zone", "for the operating area", "when the boundary is crossed", "of {n} meters", "on a hard limit"]},
    "Autoland": {"section": "Automatic Landing",
        "comp": ["the autoland controller", "the approach manager", "the flare logic", "the glideslope tracker", "the landing sequencer", "the touchdown monitor"],
        "verb": ["capture", "command", "align", "reduce", "execute", "deploy", "limit", "disarm"],
        "obj": ["the glideslope", "the flare maneuver", "the runway centerline", "the approach speed", "a go-around", "the landing gear", "the sink rate", "propulsion after touchdown"],
        "qual": ["on the approach path", "at the decision height", "to {n} meters per second", "before the flare", "when criteria are not met", "during crosswind"]},
    "Thermal": {"section": "Thermal Management",
        "comp": ["the thermal manager", "the cooling controller", "the bay heater", "the temperature monitor", "the thermal regulator", "the cooling fan"],
        "verb": ["monitor", "command", "derate", "raise", "energize", "log", "throttle", "maintain"],
        "obj": ["the avionics bay temperature", "active cooling", "propulsion output", "an over-temperature warning", "the battery heater", "the thermal trend", "payload activity", "the operating band"],
        "qual": ["above {n} degrees Celsius", "to prevent overheating", "for each zone", "during sustained load", "at {hz} Hz", "within the qualified range"]},
    "FlightControl": {"section": "Stability and Control",
        "comp": ["the flight control law", "the inner control loop", "the surface controller", "the stability augmenter", "the autopilot core", "the control mixer"],
        "verb": ["stabilize", "deflect", "track", "schedule", "limit", "damp", "hold", "revert"],
        "obj": ["the commanded attitude", "the ailerons", "the elevator", "the rudder", "the roll-rate command", "the control gains", "the load factor", "a degraded control mode"],
        "qual": ["at {hz} Hz", "within the deflection envelope", "as a function of airspeed", "to coordinate turns", "to {n} g", "when a sensor is invalid"]},
    "Telemetry": {"section": "Telemetry and Logging",
        "comp": ["the telemetry encoder", "the flight data recorder", "the log manager", "the frame assembler", "the storage controller", "the downlink formatter"],
        "verb": ["assemble", "downlink", "timestamp", "record", "prioritize", "compress", "report", "flush"],
        "obj": ["the telemetry frame", "the vehicle state", "each record", "full-rate flight data", "critical status fields", "the onboard log", "the storage remaining", "the frame sequence number"],
        "qual": ["at the configured rate", "with the mission clock", "when bandwidth is limited", "to non-volatile storage", "before shutdown", "to detect drops"]},
    "Failsafe": {"section": "Emergency and Failsafe",
        "comp": ["the failsafe manager", "the contingency handler", "the recovery controller", "the emergency logic", "the safe-state monitor", "the link-loss handler"],
        "verb": ["enter", "execute", "command", "transition", "select", "inhibit", "annunciate", "preserve"],
        "obj": ["a defined safe state", "the contingency plan", "return-to-home", "a controlled descent", "the nearest recovery point", "mission commands", "the active failsafe mode", "flight-critical state"],
        "qual": ["on loss of link", "for a critical failure", "after {ms} milliseconds without uplink", "when propulsion is lost", "during recovery", "through a reset"]},
    "GroundStation": {"section": "Ground Control Interface",
        "comp": ["the ground control station", "the operator display", "the command console", "the alert manager", "the moving-map view", "the operator interface"],
        "verb": ["present", "accept", "annunciate", "confirm", "display", "log", "throttle", "synchronize"],
        "obj": ["the vehicle state", "command entry", "cautions and warnings", "safety-critical commands", "the active mission plan", "every operator command", "nuisance alerts", "the displayed clock"],
        "qual": ["on the operator display", "from an authenticated console", "in priority order", "with an acknowledgement", "on the moving map", "to the vehicle time"]},
}

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
    "Spare parts shall remain available for the duration of the program.",
    "The supplier shall provide a quality assurance plan on request.",
]


def _fill(s):
    return (s.replace("{n}", str(random.choice([2, 3, 5, 10, 15, 20, 30, 50, 70, 85, 95, 120])))
             .replace("{ms}", str(random.choice([20, 50, 100, 200, 500, 1000])))
             .replace("{hz}", str(random.choice([1, 4, 5, 10, 20, 50, 100, 200])))
             .replace("{v}", str(random.choice([18, 22, 24, 42]))))


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    per = max(40, (target - len(ODD)) // len(SUB))
    rows = []
    for key, v in SUB.items():
        seen = set()
        attempts = 0
        produced = 0
        while produced < per and attempts < per * 60:
            attempts += 1
            text = "%s shall %s %s %s." % (
                random.choice(v["comp"]), random.choice(v["verb"]),
                random.choice(v["obj"]), random.choice(v["qual"]))
            text = _fill(text)
            text = text[0].upper() + text[1:]
            if text in seen:
                continue
            seen.add(text)
            rows.append({"module": key, "section": v["section"], "text": text})
            produced += 1

    for text in ODD:
        rows.append({"module": "General", "section": "Programmatic", "text": text})

    random.shuffle(rows)
    label = f"{round(len(rows) / 1000)}k" if len(rows) >= 1000 else str(len(rows))
    out = os.path.join(os.path.dirname(__file__), f"uav_fms_{label}_requirements.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "module", "section"])
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({"id": f"UAV-{i:05d}", "text": r["text"], "module": r["module"], "section": r["section"]})
    print(f"wrote {len(rows)} requirements to {out}  (subsystems={len(SUB)}, per={per}, odd={len(ODD)})")


if __name__ == "__main__":
    main()
