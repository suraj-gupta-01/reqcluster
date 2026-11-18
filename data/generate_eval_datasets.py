"""Deterministic generator for three hard eval datasets (seeded, stdlib + numpy only).

Outputs (in data/):
  multidomain_requirements.csv        id,text,domain,module   (24 classes)
  dependency_gold_requirements.csv    id,text,module,section
  dependency_gold_edges.csv           source_id,target_id,relation  (data|reference|refines|sequence)
  adversarial_requirements.csv        id,text,module,perturbation,source_id,contradicts_id
Edge direction: source = prerequisite/producer/parent, target = dependent (as in aerospace gold edges).
"""
import csv, random, re, sys
from pathlib import Path

HERE = Path(__file__).parent
SEED = 1337
CROSS = 0.18   # prob an action/subject is borrowed from a sibling module (lexical overlap knob)
SCALE = 0.77  # sizes above sum to ~1565; scaled to ~1200
SHARED_P = 0.5  # prob a shared cross-module clause is appended


def M(dom, name, size, subj, acts, conds, seeds, thr):
    return dict(dom=dom, name=name, size=size, subj=subj.split("|"),
                acts=[tuple(a.split(";")) for a in acts.split("|")], conds=conds.split("|"),
                seeds=seeds.split("|"), thr=[t.split(";") for t in thr.split("|")])


MODS = [
 # ---------------- automotive
 M("automotive", "Braking", 90, "the brake controller|the ABS function|the electronic stability control|the brake pedal sensor interface",
   "limit;limited;the wheel slip ratio|apply;applied;the hydraulic brake pressure demand|release;released;the brake caliper pressure|monitor;monitored;the brake fluid level|modulate;modulated;the rear axle brake force|detect;detected;a wheel speed sensor plausibility error",
   "the driver depresses the brake pedal|wheel slip exceeds the ABS threshold|the vehicle is stopped on a gradient|a brake circuit failure is detected",
   "The ABS function shall prevent wheel lock-up on surfaces with a friction coefficient above 0.3.|Brake pressure shall build up to 100 bar within 150 ms of a full pedal application.|On loss of the primary brake circuit the system shall fall back to the secondary circuit and illuminate the brake warning lamp.|The electronic parking brake must hold the vehicle on a 30 % slope with all drive sources off.|Brake torque blending between regenerative and friction braking shall be transparent to the driver.|Wheel speed signals shall be validated against vehicle reference speed every 10 ms.",
   "within {n} ms;5;250;ms|at a deceleration of {n} m/s2;2;10;m/s2|with a tolerance of +/-{n} %;1;5;%"),
 M("automotive", "Steering", 55, "the EPS controller|the steering angle sensor|the lane keeping assist|the steering torque overlay function",
   "limit;limited;the assist torque|provide;provided;the steering wheel angle signal|reduce;reduced;the steering assistance|calculate;calculated;the torque overlay request|detect;detected;a steering column torque sensor offset|hold;held;the road wheel angle",
   "the vehicle speed exceeds 130 km/h|a torque sensor fault is detected|the driver releases the steering wheel|the supply voltage drops below the operating range",
   "The steering system shall remain mechanically operable after loss of electrical assist.|The EPS shall reduce assist torque smoothly over 2 s when the motor temperature exceeds 120 degC.|Steering angle shall be reported with a resolution of 0.1 degrees.|A hands-off condition shall be detected within 5 s of torque drop below 0.3 Nm.|Unintended steering torque above 3 Nm shall be classified ASIL D.|The overlay request from the lane keeping function must be rate limited.",
   "within {n} ms;5;100;ms|with an accuracy of +/-{n} degrees;1;3;degrees|below {n} Nm;1;8;Nm"),
 M("automotive", "ADAS Perception", 80, "the object tracker|the camera perception module|the radar processing unit|the sensor fusion function",
   "classify;classified;detected pedestrians|track;tracked;the lead vehicle|fuse;fused;radar and camera object lists|estimate;estimated;the time to collision|output;output;the free space grid|reject;rejected;ghost targets",
   "visibility is reduced by rain or fog|the camera is blinded by direct sunlight|an object enters the monitored corridor|the radar reports a low confidence target",
   "The perception stack shall detect pedestrians at 60 m with a recall of at least 95 %.|Object tracks shall be maintained for at least 500 ms after the target leaves the field of view.|The fusion module must report the ego vehicle speed together with its covariance.|When the camera is blocked the function shall be flagged as degraded.|Lead vehicle distance shall be estimated with an error below 2 % up to 80 m.|The radar shall provide a target list every 50 ms.",
   "within {n} ms;20;150;ms|at a range of up to {n} m;30;200;m|with a false positive rate below {n} per hour;1;5;per hour"),
 M("automotive", "Battery Management", 75, "the battery management system|the cell monitoring unit|the charge controller|the thermal management function",
   "estimate;estimated;the state of charge|balance;balanced;the cell voltages|limit;limited;the discharge current|open;opened;the high voltage contactors|monitor;monitored;the cell temperature|derate;derated;the charging power",
   "any cell voltage exceeds 4.25 V|the pack temperature drops below 0 degC|an isolation fault is detected|the charger is connected",
   "The BMS shall estimate the state of charge with an error of less than 3 %.|High voltage contactors shall open within 100 ms of a short circuit detection.|Cell voltages shall be sampled every 10 ms with a resolution of 1 mV.|The system must prevent charging when the cell temperature is below -10 degC.|Insulation resistance shall be monitored continuously and reported on the CAN bus.|Cell balancing will be disabled during fast charging.",
   "within {n} ms;10;200;ms|at {n} degC;40;60;degC|with a tolerance of +/-{n} mV;2;10;mV"),
 M("automotive", "Infotainment", 65, "the head unit|the navigation application|the audio manager|the smartphone integration service",
   "display;displayed;the route guidance|stream;streamed;the audio content|pair;paired;a Bluetooth device|render;rendered;the user interface|mute;muted;the media playback|store;stored;the user profile",
   "the driver selects a new media source|an incoming call is received|the vehicle is in reverse gear|the user changes the display language",
   "The head unit shall boot to a usable home screen within 4 s of ignition on.|Rear camera video shall be displayed within 2 s of selecting reverse gear.|The system shall support pairing of at least 7 Bluetooth devices.|Audio playback must be muted when a high priority warning chime is active.|The navigation application shall recalculate the route within 3 s after a missed turn.|User profile data will be stored encrypted.",
   "within {n} s;1;5;s|at a minimum frame rate of {n} fps;24;60;fps|with {n} ms of audio latency;20;150;ms"),
 M("automotive", "Diagnostics OBD", 70, "the diagnostic manager|the OBD-II interface|the DTC memory|the UDS server",
   "store;stored;diagnostic trouble codes|report;reported;the freeze frame data|clear;cleared;the fault memory|respond;responded to;the UDS service requests|illuminate;illuminated;the malfunction indicator lamp|monitor;monitored;the catalyst efficiency",
   "a fault is confirmed in three consecutive driving cycles|a diagnostic session is opened|the tester requests the VIN|an emission relevant fault becomes active",
   "The diagnostic manager shall store a DTC together with a freeze frame upon fault confirmation.|The ECU shall respond to a UDS request within 50 ms.|The malfunction indicator lamp must be illuminated after two consecutive failed monitoring cycles.|Diagnostic data shall be accessible only after security access.|The OBD interface shall support the ISO 15765-4 protocol.|DTC aging counters shall be reset after 40 warm-up cycles.",
   "within {n} ms;10;100;ms|for up to {n} DTCs;20;200;DTCs|over a period of {n} driving cycles;2;40;driving cycles"),
 M("automotive", "Body Control", 50, "the body control module|the door lock actuator|the lighting controller|the wiper control function",
   "lock;locked;all vehicle doors|switch;switched;the headlamps|activate;activated;the turn indicators|control;controlled;the interior lighting|operate;operated;the windshield wipers|disable;disabled;the central locking",
   "the vehicle speed exceeds 10 km/h|a crash is detected|the ambient light falls below the threshold|the ignition is switched off",
   "Doors shall be locked automatically when the vehicle speed exceeds 15 km/h.|All doors must unlock upon crash signal reception.|The headlamps shall be switched on automatically below 1000 lux.|Turn indicators shall flash at 1.5 Hz +/- 0.5 Hz.|The wiper shall adapt its interval to rain sensor output.|Interior lighting will fade out over 3 s after door closing.",
   "within {n} ms;50;500;ms|at {n} Hz;1;3;Hz|with a delay of at most {n} s;1;10;s"),
 M("automotive", "OTA Update", 45, "the OTA client|the update orchestrator|the secure bootloader|the gateway ECU",
   "download;downloaded;the update package|verify;verified;the package signature|install;installed;the software image|roll back;rolled back;the previous software version|notify;notified;the user|resume;resumed;an interrupted download",
   "the vehicle is parked and the battery level is above 50 %|the signature verification fails|the download is interrupted|the user postpones the installation",
   "Update packages shall be signed and verified before installation.|The OTA client must resume interrupted downloads without restarting from zero.|If the installation fails the ECU shall revert to the last known good image.|The bootloader shall authenticate the application image before jump.|Updates shall not start unless the vehicle is stationary.|The update campaign status shall be reported to the backend.",
   "within {n} minutes;5;60;minutes|with {n} retries;2;5;retries|using a {n} bit key;128;256;bit"),
 # ---------------- medical
 M("medical", "Infusion Dosing", 85, "the infusion controller|the dose calculation module|the pump motor driver|the drug library",
   "compute;computed;the infusion rate|deliver;delivered;the programmed volume|limit;limited;the maximum bolus|stop;stopped;the infusion|verify;verified;the entered dose against the drug library|calibrate;calibrated;the flow rate",
   "the programmed rate exceeds the hard limit|the syringe is empty|the user confirms a bolus|an occlusion is detected",
   "The pump shall deliver the programmed volume with an accuracy of +/- 5 %.|The infusion rate shall be calculated from the concentration and the prescribed dose.|A bolus above the soft limit must require a second confirmation.|Delivery shall stop within 1 s when an occlusion is detected.|The drug library shall be validated before it is activated.|The system shall prevent a free flow condition when the cassette is removed.",
   "within {n} s;1;5;s|at a rate of {n} mL/h;1;999;mL/h|with an accuracy of +/-{n} %;2;5;%"),
 M("medical", "Alarms", 75, "the alarm manager|the alarm annunciator|the monitoring function|the alarm escalation logic",
   "raise;raised;a high priority alarm|escalate;escalated;the unacknowledged alarm|silence;silenced;the audible signal|display;displayed;the alarm message|latch;latched;the technical alarm|inhibit;inhibited;the nuisance alarms",
   "the patient parameter violates the alarm limit|an alarm is not acknowledged|the alarm silence period expires|the battery is critically low",
   "High priority alarms shall be annunciated within 2 s of the alarm condition.|The audible alarm volume must be at least 45 dB at 1 m.|An unacknowledged alarm shall escalate to a higher priority after 60 s.|Alarm limits shall be adjustable by authorised clinicians only.|The alarm system shall comply with IEC 60601-1-8.|Technical alarms shall remain latched until the cause is cleared.",
   "within {n} s;1;10;s|at a sound level of {n} dB;45;85;dB|after {n} s;10;120;s"),
 M("medical", "Patient Data Privacy", 60, "the data protection layer|the patient record service|the authentication module|the anonymisation function",
   "encrypt;encrypted;patient identifiable data|authenticate;authenticated;each clinician|mask;masked;the patient name on the display|restrict;restricted;access to the records|anonymise;anonymised;exported data sets|expire;expired;the user session",
   "a user logs in from an unknown terminal|three consecutive logins fail|the record is exported|the session is idle",
   "Patient data shall be encrypted at rest using AES-256.|The system must lock the session after 10 minutes of inactivity.|Access to patient records shall be role based.|Exported data shall be anonymised in accordance with applicable privacy regulations.|Failed login attempts shall be limited to five before lockout.|Patient identifiers shall never be written to the diagnostic log.",
   "within {n} minutes;5;30;minutes|using {n} bit encryption;128;256;bit|for at most {n} days;30;365;days"),
 M("medical", "Sensor Acquisition", 70, "the signal acquisition unit|the ECG front end|the SpO2 sensor driver|the pressure transducer interface",
   "sample;sampled;the analogue signal|filter;filtered;mains interference|detect;detected;a lead-off condition|calibrate;calibrated;the pressure sensor|convert;converted;the raw measurement|validate;validated;the signal quality",
   "the electrode impedance exceeds the limit|the sensor is disconnected|the signal quality index falls below the threshold|the sensor reports an out of range value",
   "The ECG front end shall sample at 500 Hz with 16 bit resolution.|Mains interference shall be attenuated by at least 40 dB.|Lead-off shall be detected within 1 s.|The SpO2 value must be reported with an accuracy of +/- 2 %.|Pressure readings shall be zeroed before each measurement.|Raw samples shall be time stamped with the device clock.",
   "within {n} ms;10;500;ms|at a rate of {n} Hz;50;1000;Hz|with a resolution of {n} bit;12;24;bit"),
 M("medical", "Power Management", 50, "the power supply unit|the battery charger|the power manager|the backup battery",
   "switch;switched;to battery operation|charge;charged;the battery|report;reported;the remaining runtime|shut down;shut down;the device safely|protect;protected;the circuit against overcurrent|monitor;monitored;the mains supply",
   "the mains supply is interrupted|the battery capacity falls below 10 %|the device is connected to the charger|an overvoltage is detected",
   "The device shall switch to battery operation without interruption of therapy.|The battery shall provide at least 4 hours of operation at the nominal rate.|A low battery alarm shall be raised at 20 % remaining capacity.|The charger must terminate charging at the end of charge voltage.|The device shall shut down safely when the battery is depleted.|Leakage current shall not exceed the limits of IEC 60601-1.",
   "within {n} ms;5;100;ms|for at least {n} hours;2;8;hours|at {n} V;10;24;V"),
 M("medical", "Medical UI", 60, "the user interface|the touchscreen|the settings menu|the on-screen keyboard",
   "display;displayed;the current infusion status|require;required;a confirmation|show;shown;the remaining volume|lock;locked;the keypad|accept;accepted;numeric input|highlight;highlighted;the changed parameter",
   "the user enters a value outside the allowed range|the keypad lock is active|the user selects a drug from the list|the screen is touched with gloves",
   "The user interface shall be usable with gloved hands.|Critical settings shall require a two step confirmation.|The display shall remain readable at 500 lux ambient light.|Entered values shall be echoed before being applied.|The UI must follow the usability engineering process of IEC 62366.|Keypad lock shall engage automatically after 30 s of inactivity.",
   "within {n} s;1;5;s|with a font size of at least {n} pt;10;18;pt|after {n} s;15;60;s"),
 M("medical", "Audit Logging", 55, "the audit logger|the event recorder|the log storage|the time service",
   "record;recorded;each user action|protect;protected;the log against modification|store;stored;the event history|time stamp;time stamped;every log entry|export;exported;the audit trail|rotate;rotated;the log files",
   "a setting is changed|the log storage is 90 % full|a user logs out|the audit trail is requested",
   "Every parameter change shall be logged with the user identity and a timestamp.|The audit log shall be tamper evident.|Log entries must be retained for at least 2 years.|The system shall record each alarm event in the event history.|The log shall store at least 10000 entries.|Exported logs shall contain a checksum.",
   "within {n} ms;10;500;ms|for at least {n} years;1;10;years|with a capacity of {n} entries;5000;50000;entries"),
 M("medical", "Connectivity", 60, "the network interface|the HL7 gateway|the wireless module|the device server",
   "transmit;transmitted;the infusion data|reconnect;reconnected;the hospital network|exchange;exchanged;HL7 messages|pair;paired;the device with the server|buffer;buffered;the outgoing data|encrypt;encrypted;the link",
   "the network connection is lost|the server rejects the message|the wireless signal strength is low|the device is first connected",
   "Therapy shall continue unaffected when network connectivity is lost.|The device shall buffer up to 24 hours of data while offline.|Messages shall be sent over a TLS 1.2 or higher connection.|The wireless module must reconnect automatically within 30 s.|Interface messages shall conform to HL7 v2.5.|The connectivity module shall report link status to the supervisor.",
   "within {n} s;5;60;s|with a latency below {n} ms;50;500;ms|at a throughput of {n} kbit/s;64;1024;kbit/s"),
 # ---------------- rail
 M("rail", "Train Detection", 70, "the axle counter|the track circuit|the train detection system|the evaluator unit",
   "detect;detected;track occupancy|report;reported;the section status|reset;reset;the axle counting section|evaluate;evaluated;the wheel sensor pulses|clear;cleared;the occupied section|supervise;supervised;the sensor health",
   "an axle count mismatch occurs|a wheel sensor fails|the section is reset by the operator|the train passes the counting point",
   "The axle counter shall report a section clear only when in and out counts are equal.|Train presence shall be detected within 500 ms of entering the section.|The track circuit must detect a shunt of 0.5 ohm.|A sensor failure shall cause the section to be reported occupied.|Reset of a section shall require operator confirmation.|The evaluator shall comply with SIL 4 of EN 50129.",
   "within {n} ms;100;1000;ms|at speeds up to {n} km/h;100;350;km/h|with a shunt sensitivity of {n} ohm;0.25;1;ohm"),
 M("rail", "Interlocking", 90, "the interlocking|the point controller|the route manager|the signal controller",
   "set;set;the requested route|lock;locked;the points|prevent;prevented;conflicting routes|release;released;the route|command;commanded;the signal to red|verify;verified;the point detection",
   "a route request is received|the point detection is lost|the route is no longer required|the track section ahead is occupied",
   "The interlocking shall not clear a signal unless all points in the route are locked and detected.|Conflicting routes must never be set simultaneously.|The route shall be released after the train has cleared the last section.|Points shall be moved to the required position within 6 s.|On loss of point detection the signal shall revert to danger.|The interlocking logic shall be developed according to EN 50128.",
   "within {n} s;2;10;s|with a safety integrity level of SIL {n};2;4;SIL|after {n} s of time delay;30;180;s"),
 M("rail", "Movement Authority", 75, "the radio block centre|the movement authority generator|the onboard ETCS unit|the trackside RBC",
   "issue;issued;a movement authority|extend;extended;the end of authority|revoke;revoked;the authority|transmit;transmitted;the track description|calculate;calculated;the permitted distance|reject;rejected;an invalid position report",
   "the train sends a position report|the route ahead is cleared|an emergency stop is requested|the train enters a new RBC area",
   "The RBC shall issue a movement authority only for sections that are locked and free.|The end of authority shall be shortened immediately on route cancellation.|Position reports must be processed within 1 s of reception.|The onboard unit shall reject a movement authority with an invalid sequence number.|Handover between RBC areas shall be seamless.|The movement authority shall include the gradient profile.",
   "within {n} s;1;5;s|up to a distance of {n} m;500;5000;m|every {n} s;5;30;s"),
 M("rail", "Speed Supervision", 70, "the onboard computer|the speed supervision function|the braking curve calculator|the odometer",
   "supervise;supervised;the train speed|calculate;calculated;the emergency braking curve|apply;applied;the service brake|measure;measured;the travelled distance|display;displayed;the permitted speed|trigger;triggered;an emergency brake application",
   "the train speed exceeds the permitted speed|the odometer detects wheel slip|the target speed changes|the driver does not react to the warning",
   "The onboard unit shall trigger the emergency brake if the speed exceeds the permitted speed by 5 km/h.|The odometer shall measure distance with an error below 2 %.|The braking curve shall take the gradient and train length into account.|The permitted speed shall be displayed to the driver at all times.|Speed measurements must be cross checked with two independent sources.|The service brake shall be applied after the warning limit is exceeded.",
   "within {n} ms;100;800;ms|with an error below {n} %;1;5;%|above {n} km/h;5;40;km/h"),
 M("rail", "Door Control", 45, "the door controller|the platform screen door system|the passenger door|the door interlock",
   "open;opened;the doors|close;closed;the doors|inhibit;inhibited;traction|detect;detected;an obstacle|release;released;the door enable|monitor;monitored;the door closed and locked status",
   "the train is stopped at the platform|an obstacle is detected between the doors|the train speed exceeds 3 km/h|the emergency release is operated",
   "Doors shall only be released when the train is stationary and aligned with the platform.|Traction must be inhibited unless all doors are closed and locked.|An obstacle shall cause the door to reverse within 300 ms.|Doors shall close within 5 s after the close command.|The door status shall be reported to the driver cabin.|Platform doors shall open in synchronisation with train doors.",
   "within {n} s;2;10;s|at a speed below {n} km/h;3;5;km/h|with a force of less than {n} N;100;150;N"),
 M("rail", "Rail Communications", 70, "the communication manager|the GSM-R radio|the onboard modem|the trackside network",
   "establish;established;the radio link|transmit;transmitted;safety messages|monitor;monitored;the link quality|switch;switched;to the redundant channel|authenticate;authenticated;the peer|buffer;buffered;the messages",
   "the radio link is interrupted|the message integrity check fails|the network latency exceeds the limit|a handover is performed",
   "The safety communication shall detect corrupted, delayed and re-sequenced messages.|The radio link shall be re-established within 5 s after loss.|Messages must be protected according to EN 50159.|The network shall support a maximum end-to-end latency of 100 ms.|On loss of communication for 20 s the train shall brake to a standstill.|The system shall switch to the redundant channel on link failure.",
   "within {n} ms;50;500;ms|with a latency below {n} ms;50;300;ms|for at most {n} s;5;30;s"),
 M("rail", "Rail Maintenance", 55, "the maintenance system|the condition monitoring unit|the diagnostic recorder|the asset manager",
   "record;recorded;the operating hours|predict;predicted;component wear|schedule;scheduled;the inspection|report;reported;the equipment fault|download;downloaded;the diagnostic data|log;logged;maintenance actions",
   "the component reaches its service life|a fault is detected during operation|the maintenance interval expires|the diagnostic data is requested",
   "The condition monitoring unit shall record wheel wear data for each axle.|Faults shall be reported to the maintenance centre within 60 s.|Maintenance actions must be logged with technician identity.|The system shall schedule inspections based on operating hours.|Diagnostic data shall be downloadable via the service interface.|Availability of the system shall be demonstrated according to EN 50126.",
   "within {n} s;10;120;s|every {n} operating hours;500;5000;operating hours|for at least {n} years;5;15;years"),
 M("rail", "Timetable", 45, "the traffic management system|the timetable planner|the dispatcher interface|the regulation function",
   "generate;generated;the train schedule|update;updated;the planned arrival times|resolve;resolved;scheduling conflicts|display;displayed;the train graph|adjust;adjusted;the dwell time|publish;published;the passenger information",
   "a train is delayed by more than 2 minutes|the dispatcher modifies a route|a platform is blocked|a new timetable is imported",
   "The system shall recompute arrival times within 5 s of a delay report.|Dwell times shall be adjustable in steps of 5 s.|The timetable planner must detect conflicts on single track sections.|Passenger information shall be updated within 10 s of a schedule change.|The dispatcher shall be able to hold a train at a station.|The timetable import shall validate all headways.",
   "within {n} s;2;30;s|with a headway of {n} s;90;300;s|for up to {n} trains;50;500;trains"),
]

SHARED = {
 "automotive": ["and log any detected fault with a timestamp", "and report the fault status on the CAN bus", "with a worst-case latency of {n} ms", "in accordance with {std}", "and set the corresponding diagnostic flag", "and keep the data consistent after a reset", "and shall remain functional in degraded mode"],
 "medical": ["and log any detected fault with a timestamp", "and record the event in the audit log", "with a worst-case latency of {n} ms", "in accordance with {std}", "and notify the supervisor of any fault", "and keep the data consistent after a power cycle", "and shall remain functional in degraded mode"],
 "rail": ["and log any detected fault with a timestamp", "and report the fault status over the data bus", "with a worst-case latency of {n} ms", "in accordance with {std}", "and notify the supervisor of any fault", "and keep the data consistent after a restart", "and shall remain functional in degraded mode"],
}
STDS = {"automotive": ["ISO 26262-6", "ISO 26262-5", "ISO 21434", "AUTOSAR R20-11", "ISO 14229"],
        "medical": ["IEC 62304", "ISO 14971", "IEC 60601-1", "IEC 62366", "IEC 81001-5-1"],
        "rail": ["EN 50128", "EN 50126", "EN 50129", "EN 50159", "IEEE 1474.1"]}
RAT = {"automotive": ["so that hazardous events are avoided", "to maintain the safe state", "to meet the ASIL target"],
       "medical": ["to avoid harm to the patient", "to reduce the residual risk", "to maintain basic safety"],
       "rail": ["to maintain a safe state", "to ensure the required SIL", "to avoid conflicting movements"]}
MODALS = ["shall", "shall", "shall", "shall", "must", "will"]
UP = lambda s: s[0].upper() + s[1:]


def num(rng, lo, hi):
    lo, hi = float(lo), float(hi)
    v = rng.uniform(lo, hi)
    return str(int(round(v))) if hi >= 10 else ("%.1f" % v)


def thr(rng, m):
    t = rng.choice(m["thr"])
    return t[0].format(n=num(rng, t[1], t[2]))


def gen_row(rng, m, by_dom):
    dom = m["dom"]
    sib = by_dom[dom]
    def pick_act():
        src = rng.choice(sib) if rng.random() < CROSS else m
        return rng.choice(src["acts"])
    def pick_subj():
        src = rng.choice(sib) if rng.random() < CROSS else m
        return rng.choice(src["subj"])
    def act_txt(a): return "%s %s" % (a[0], a[2])
    def tail():
        parts = []
        if rng.random() < 0.55: parts.append(thr(rng, m))
        if rng.random() < SHARED_P:
            s = rng.choice(SHARED[dom]).format(n=num(rng, 5, 200), std=rng.choice(STDS[dom]))
            parts.append(s)
        elif rng.random() < 0.2: parts.append("in accordance with " + rng.choice(STDS[dom]))
        return (" " + " ".join(parts)) if parts else ""
    mod = rng.choice(MODALS)
    kind = rng.random()
    a, a2 = pick_act(), pick_act()
    while a2 == a: a2 = pick_act()
    S, C = pick_subj(), rng.choice(m["conds"])
    if kind < 0.10:  # terse
        o = rng.choice(["%s %s" % (UP(a[0]), a[2]) + (" " + thr(rng, m) if rng.random() < .6 else ""),
                        "%s %s %s" % (UP(a[2]), a[1], thr(rng, m)),
                        "%s: %s %s" % (UP(S.replace("the ", "")), a[0], a[2])])
        return o + "."
    if kind < 0.15:  # long
        return ("When %s and %s, %s %s %s %s, %s %s, and, where applicable, %s %s %s, %s%s."
                % (C, rng.choice(m["conds"]), S, mod, a[0], a[2], a2[0], a2[2], rng.choice(["also", "additionally"]),
                   *rng.choice([("record", "the outcome"), ("flag", "the result")]), rng.choice(RAT[dom]), tail()))
    if kind < 0.45:
        return "%s %s %s %s%s." % (UP(S), mod, a[0], a[2], tail())
    if kind < 0.65:
        return "When %s, %s %s %s %s%s." % (C, S, mod, a[0], a[2], tail())
    if kind < 0.77:
        return "%s %s be %s by %s%s." % (UP(a[2]), mod, a[1], S, tail())
    if kind < 0.87:
        return "%s %s %s %s and %s %s%s." % (UP(S), mod, a[0], a[2], a2[0], a2[2], tail())
    if kind < 0.94:
        return "%s %s %s %s %s, %s." % (UP(S), mod, a[0], a[2], rng.choice(["if " + C, "while " + C, "unless " + C]), rng.choice(RAT[dom]))
    return "%s %s %s %s %s, as required by %s." % (UP(S), mod, a[0], a[2], thr(rng, m), rng.choice(STDS[dom]))


def gen_multidomain(rng):
    by_dom = {}
    for m in MODS: by_dom.setdefault(m["dom"], []).append(m)
    rows, seen = [], set()
    for m in MODS:
        texts = []
        for s in m["seeds"]:
            if s not in seen: seen.add(s); texts.append(s)
        guard = 0
        while len(texts) < round(m["size"] * SCALE) and guard < 50000:
            guard += 1
            t = gen_row(rng, m, by_dom)
            if t in seen: continue
            seen.add(t); texts.append(t)
        rows += [(t, m["dom"], m["name"]) for t in texts]
    rng.shuffle(rows)
    return [("MD-%04d" % (i + 1), t, d, mod) for i, (t, d, mod) in enumerate(rows)]


# ------------------------------------------------------------------ set 2
ARTS = [  # synonym groups for the same artifact
 ["vehicle speed", "ego velocity", "longitudinal velocity estimate"], ["steering angle", "road wheel angle", "front wheel deflection"],
 ["brake pressure", "hydraulic line pressure", "caliper pressure value"], ["battery state of charge", "SoC estimate", "remaining charge level"],
 ["object list", "tracked target list", "fused obstacle table"], ["yaw rate", "angular rotation rate", "heading rate"],
 ["fault status word", "diagnostic status flags", "health summary"], ["infusion rate", "flow setpoint", "delivery rate command"],
 ["alarm state", "alert condition record", "annunciation status"], ["heart rate", "pulse rate value", "cardiac frequency"],
 ["patient identifier", "subject ID", "patient reference key"], ["occlusion pressure", "line blockage reading", "downstream pressure"],
 ["battery runtime estimate", "remaining operating time", "time-to-empty"], ["session token", "authentication ticket", "login credential handle"],
 ["track occupancy status", "section occupied flag", "train presence indication"], ["point position", "switch blade state", "turnout detection value"],
 ["movement authority", "MA message", "permission to proceed"], ["permitted speed", "speed ceiling", "maximum allowed velocity"],
 ["train position report", "location report", "odometer-based position"], ["door closed status", "door lock confirmation", "leaf closed indication"],
 ["braking curve", "deceleration profile", "stopping trajectory"], ["route status", "path allocation state", "itinerary lock state"],
 ["link quality index", "radio signal metric", "channel quality figure"], ["planned arrival time", "scheduled ETA", "timetable arrival slot"],
 ["wheel speed", "rotational wheel velocity", "tyre rotation rate"], ["cell temperature", "pack thermal reading", "module temperature value"],
 ["route guidance instruction", "navigation maneuver", "turn command"], ["software image hash", "firmware digest", "package checksum"],
]
COMPS = ["the vehicle dynamics module", "the control unit", "the supervisor", "the safety monitor", "the data manager", "the HMI task", "the logger", "the gateway", "the estimator",
         "the scheduler", "the communication stack", "the diagnostic service", "the planning function", "the actuator driver", "the validation component", "the recorder", "the arbiter"]
PROD = ["compute", "produce", "publish", "generate", "derive", "provide"]
CONS = ["use", "consume", "read", "rely on", "take as input", "subscribe to"]
SEQ_EV = [("initialisation", "self-test"), ("calibration", "measurement"), ("route locking", "signal clearing"), ("authentication", "data transfer"), ("pre-charge", "contactor closing"),
          ("prime cycle", "dose delivery"), ("download", "installation"), ("door closing", "traction release"), ("boot", "service start"), ("sensor warm-up", "acquisition"),
          ("handshake", "message exchange"), ("pressure test", "infusion start")]


def gen_dependency(rng):
    reqs, edges = {}, []  # key -> text with <<key>> refs
    n = [0]
    def add(text, mod, sec):
        n[0] += 1; k = "k%d" % n[0]; reqs[k] = [text, mod, sec]; return k
    for gi, grp in enumerate(ARTS):
        p = add("%s shall %s the %s %s." % (UP(rng.choice(COMPS)), rng.choice(PROD), grp[0], rng.choice(["every 20 ms", "at 10 Hz", "within 100 ms of each cycle", "on every update"])), "Data", "Producer-Consumer")
        used = set()
        for j in range(4):
            c = add("%s shall %s the %s %s." % (UP(rng.choice(COMPS)), rng.choice(CONS), rng.choice(grp[:3] if j else grp[1:]),
                    rng.choice(["to determine the next action", "for plausibility checking", "when computing its output", "during each control cycle"])), "Data", "Producer-Consumer")
            edges.append((p, c, "data"))
    for i in range(16):
        t = add("%s shall maintain the %s within the specified range." % (UP(rng.choice(COMPS)), rng.choice(rng.choice(ARTS))), "Reference", "Explicit Reference")
        for _ in range(2):
            r = add("%s shall %s as specified in <<%s>>." % (UP(rng.choice(COMPS)), rng.choice(["verify the limits", "apply the same tolerance", "enforce the constraint", "report violations"]), t), "Reference", "Explicit Reference")
            edges.append((t, r, "reference"))
    for i in range(10):
        a = rng.choice(ARTS)[0]
        par = add("The system shall provide %s monitoring with fault handling." % a, "Refinement", "Parent-Child")
        for aspect in rng.sample(["The sampling period of the %s shall not exceed 50 ms.", "Invalid values of the %s shall be flagged and excluded.", "Each change of the %s shall be logged with a timestamp.", "The %s shall be validated against its plausible range.", "Loss of the %s shall raise a fault within 200 ms."], 4):
            ch = add(aspect % a + " (refines <<%s>>)" % par if rng.random() < .3 else aspect % a, "Refinement", "Parent-Child")
            edges.append((par, ch, "refines"))
    for i in range(10):
        evs = rng.sample(SEQ_EV, 3)
        prev = None
        for e in evs[:3]:
            pass
        names = [e[0] for e in evs] + [evs[-1][1]]
        keys = []
        for j, nm in enumerate(names):
            if j == 0: k = add("The system shall perform the %s sequence at startup." % nm, "Sequence", "Triggers")
            else: k = add("The system shall start the %s after completion of the %s." % (nm, names[j - 1]), "Sequence", "Triggers")
            if keys: edges.append((keys[-1], k, "sequence"))
            keys.append(k)
    # distractors: share vocabulary, no dependency
    n_dep = len(reqs)
    for i in range(int(n_dep * 0.3 / 0.7)):
        g = rng.choice(ARTS)
        add(rng.choice([
            "The %s sensor housing shall withstand a vibration of 10 g.", "The documentation for the %s function shall be reviewed annually.",
            "The %s shall be described in the user manual.", "Training material covering the %s shall be provided to operators.",
            "The unit that handles the %s shall have a mass below 2 kg.", "The supplier shall demonstrate maturity of the %s process at each audit.",
            "Test equipment for the %s shall be calibrated every 12 months."]) % rng.choice(g), "Distractor", "Unrelated")
    keys = list(reqs); rng.shuffle(keys)
    ids = {k: "DEP-%03d" % (i + 1) for i, k in enumerate(keys)}
    fix = lambda s: re.sub(r"<<(k\d+)>>", lambda m: ids[m.group(1)], s)
    rows = sorted([(ids[k], fix(v[0]), v[1], v[2]) for k, v in reqs.items()])
    erows = sorted({(ids[s], ids[t], r) for s, t, r in edges})
    return rows, erows


# ------------------------------------------------------------------ set 3
SYN = [("shall", "must"), ("ensure", "guarantee"), ("detect", "identify"), ("limit", "restrict"), ("within", "inside"), ("monitor", "supervise"),
       ("report", "communicate"), ("provide", "supply"), ("store", "save"), ("fault", "failure"), ("display", "show"), ("verify", "check")]


def paraphrase(rng, t):
    m = re.match(r"^When (.+?), (.+)\.$", t)
    if m: t = "%s %s." % (UP(m.group(2)), "when " + m.group(1)) if False else "%s when %s." % (UP(m.group(2)), m.group(1))
    for a, b in rng.sample(SYN, len(SYN)):
        if re.search(r"\b%s\b" % a, t):
            t = re.sub(r"\b%s\b" % a, b, t, count=1)
            if rng.random() < .5: break
    m = re.match(r"^(The [a-z0-9 \-]+?) (shall|must|will) (\w+) (the [a-z0-9 \-]+?)( .*)?\.$", t)
    if m and rng.random() < .6:
        t = "%s %s be %s by %s%s." % (UP(m.group(4)), m.group(2), m.group(3) + ("d" if m.group(3).endswith("e") else "ed"), m.group(1).lower(), m.group(5) or "")
    return t


def near_dup(rng, t):
    nums = list(re.finditer(r"\d+(?:\.\d+)?", t))
    if not nums: return None
    m = rng.choice(nums); v = float(m.group())
    nv = v * rng.choice([2, 0.5, 1.5, 10]) if v else 5
    s = str(int(nv)) if m.group().isdigit() else "%.1f" % nv
    return t[:m.start()] + (s if s != m.group() else s + "0") + t[m.end():]


def typo(rng, t):
    c = list(t)
    for _ in range(rng.randint(1, 3)):
        i = rng.randrange(1, len(c) - 1); op = rng.choice("swap drop dup space case".split())
        if op == "swap": c[i], c[i + 1] = c[i + 1], c[i]
        elif op == "drop": c[i] = ""
        elif op == "dup": c[i] = c[i] * 2
        elif op == "space": c[i] = "" if c[i] == " " else c[i]
        else: c[i] = c[i].swapcase()
    return "".join(c)


def negate(t):
    for a, b in [(" shall ", " shall not "), (" must ", " must not "), (" will ", " will not ")]:
        if a in t: return t.replace(a, b, 1)
    return None


NOISE = ["TBD", "see above", "   ", "N/A", "TBC - to be confirmed with supplier", ".", "-", "As per previous section.", "Reserved.", "(deleted)", "See Table 3.", "Lorem ipsum dolor sit amet.",
         "The cafeteria shall serve lunch between 12:00 and 13:30.", "Meeting moved to Thursday, please update calendar.", "¿Dónde está el requisito? – 照明制御 → ≤ 5 µs ± 2°C",
         "The \"safe state\", as defined in the glossary, shall be \"entered\", without exception.", "Line one of the requirement,\nline two with a comma, and a second line\r\nthird line.",
         "Das System muss innerhalb von 5 ms reagieren – überprüfen!", "✅ done 🚀", "Check JIRA-4521; owner: j.smith@example.com"]
RAMBLE = ["the", "system", "should", "probably", "consider", "that", "in", "some", "cases", "the", "operator", "may", "want", "to", "think", "about", "whether", "this", "is", "needed", "however", "it", "was", "said", "at", "the", "review", "that", "we", "will", "discuss", "later", "and", "regardless", "of", "outcome", "the", "team", "agreed", "pending", "analysis"]


def gen_adv(rng, base):
    rows = []; seen = set()
    def add(t, mod, per, src, ctr=""):
        rows.append([t, mod, per, src, ctr])
    pool = list(base)
    for per, fn in [("paraphrase", paraphrase), ("near_duplicate", near_dup), ("typo", typo)]:
        got = 0
        for b in rng.sample(pool, len(pool)):
            t = fn(rng, b[1])
            if t and t != b[1] and t not in seen:
                seen.add(t); add(t, b[3], per, b[0]); got += 1
            if got == 120: break
    got = 0
    for b in rng.sample(pool, len(pool)):
        nums = re.search(r"\d+", b[1]); neg = negate(b[1])
        alt = near_dup(rng, b[1]) if nums and rng.random() < .5 else None
        alt = alt or neg
        if not alt or alt == b[1]: continue
        got += 1
        add(b[1], b[3], "contradiction", b[0], "@%d" % (len(rows) + 1)); add(alt, b[3], "contradiction", b[0], "@%d" % len(rows))
        if got == 60: break
    for i in range(120):
        r = rng.random()
        if r < .5: t = rng.choice(NOISE) + ("" if rng.random() < .5 else " " * rng.randint(0, 3))
        elif r < .7: t = " ".join(rng.choice(RAMBLE) for _ in range(rng.randint(400, 520)))
        else: t = rng.choice(NOISE) + " " + rng.choice(pool)[1] + " ref #%d" % i
        add(t, "__noise__", "noise", "")
    rng.shuffle(rows)
    idx = {}
    for i, r in enumerate(rows): idx["@%d" % (rows.index(r) + 1)] = None
    # resolve contradiction partner ids after shuffle: pair rows by (src, contradiction)
    out = [("ADV-%04d" % (i + 1), r[0], r[1], r[2], r[3], "") for i, r in enumerate(rows)]
    groups = {}
    for i, r in enumerate(out):
        if r[3] == "contradiction": groups.setdefault(r[4], []).append(i)
    out = [list(r) for r in out]
    for ix in groups.values():
        if len(ix) == 2:
            out[ix[0]][5], out[ix[1]][5] = out[ix[1]][0], out[ix[0]][0]
    return [tuple(r) for r in out]


def write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def main():
    rng = random.Random(SEED)
    md = gen_multidomain(rng)
    write(HERE / "multidomain_requirements.csv", ["id", "text", "domain", "module"], md)
    dreq, dedge = gen_dependency(random.Random(SEED + 1))
    write(HERE / "dependency_gold_requirements.csv", ["id", "text", "module", "section"], dreq)
    write(HERE / "dependency_gold_edges.csv", ["source_id", "target_id", "relation"], dedge)
    base = [(i, t, d, m) for i, t, d, m in md]
    adv = gen_adv(random.Random(SEED + 2), base)
    write(HERE / "adversarial_requirements.csv", ["id", "text", "module", "perturbation", "source_id", "contradicts_id"], adv)
    return md, dreq, dedge, adv


if __name__ == "__main__":
    import collections, pandas as pd
    md, dreq, dedge, adv = main()
    D = HERE
    df = pd.read_csv(D / "multidomain_requirements.csv", keep_default_na=False)
    assert len(df) == len(md) and 1100 <= len(df) <= 1300, len(df)
    assert df.text.is_unique, "duplicate texts in set 1"
    cnt = df.module.value_counts(); assert df.module.nunique() == 24 and cnt.min() >= 25 and cnt.max() <= 90, cnt
    dr = pd.read_csv(D / "dependency_gold_requirements.csv", keep_default_na=False); de = pd.read_csv(D / "dependency_gold_edges.csv")
    assert len(dr) == len(dreq) and len(de) == len(dedge)
    ids = set(dr.id); assert de.source_id.isin(ids).all() and de.target_id.isin(ids).all()
    assert set(de.relation) == {"data", "reference", "refines", "sequence"} and (de.source_id != de.target_id).all()
    ad = pd.read_csv(D / "adversarial_requirements.csv", keep_default_na=False)
    assert len(ad) == len(adv) and ad.id.is_unique and 500 <= len(ad) <= 700, len(ad)
    c = ad[ad.contradicts_id != ""]; assert set(c.contradicts_id) <= set(ad.id)
    assert set(ad.perturbation) == {"paraphrase", "near_duplicate", "typo", "contradiction", "noise"}
    print("OK", len(df), "reqs;", len(dr), "dep reqs;", len(de), "edges;", len(ad), "adversarial;", dict(cnt.describe()[["min", "max"]]))
