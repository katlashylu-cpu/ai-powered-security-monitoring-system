import os
import re
from datetime import datetime


class Config:
    LOG_BASE_DIR = "logs"
    TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


def sanitize_string(text):
    if not text:
        return ""
    return re.sub(r'[^\w\s\-_,\.]', '', str(text))


def parse_log_file(log_file_path):
    authorized_count = 0
    intruder_count = 0
    suspicious_count = 0
    entries = []
    if not os.path.exists(log_file_path):
        return authorized_count, intruder_count, suspicious_count, entries
    try:
        with open(log_file_path, "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" - ")
                if len(parts) < 3:
                    continue
                raw_ts = parts[0]
                try:
                    dt = datetime.strptime(raw_ts, Config.TIMESTAMP_FORMAT)
                    pretty = dt.strftime("%b %d, %Y – %I:%M:%S %p")
                except:
                    pretty = raw_ts
                event_part = parts[1]
                event = event_part.split(
                    ":")[-1].strip() if ":" in event_part else event_part
                name_loc = parts[2]
                name = "Unknown"
                location = "Unknown"
                if "Name:" in name_loc:
                    name_part = name_loc.split("Name:")[-1]
                    name = name_part.split(",")[0].strip(
                    ) if "," in name_part else name_part.strip()
                    name = sanitize_string(name)
                if "Location:" in name_loc:
                    loc_part = name_loc.split("Location:")[-1]
                    location = loc_part.strip()
                if "Authorized" in event:
                    authorized_count += 1
                    event_type = "Authorized"
                elif "Intruder" in event:
                    intruder_count += 1
                    event_type = "Intruder"
                elif "Suspicious" in event:
                    suspicious_count += 1
                    event_type = "Suspicious"
                else:
                    event_type = event
                entries.append(
                    f"{pretty} | {event_type} | {name} | {location}")
    except Exception as e:
        print(f"Error parsing {log_file_path}: {e}")
    return authorized_count, intruder_count, suspicious_count, entries
