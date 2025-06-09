from langchain.tools import tool
import json
from datetime import datetime

AGENDA_FILE = "agenda.json"


def load_agenda():
    try:
        with open(AGENDA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_agenda(agenda):
    with open(AGENDA_FILE, "w", encoding="utf-8") as f:
        json.dump(agenda, f, indent=4)


def check_conflict(agenda, date, start, end):
    new_start = datetime.strptime(f"{date} {start}", "%Y-%m-%d %H:%M")
    new_end = datetime.strptime(f"{date} {end}", "%Y-%m-%d %H:%M")

    for event in agenda:
        if event["date"] != date:
            continue
        existing_start = datetime.strptime(
            f"{event['date']} {event['start']}", "%Y-%m-%d %H:%M"
        )
        existing_end = datetime.strptime(
            f"{event['date']} {event['end']}", "%Y-%m-%d %H:%M"
        )

        if new_start < existing_end and new_end > existing_start:
            return True
    return False


def is_morning_time(start: str, end: str) -> bool:
    """
    Checks if the event is entirely or partially in the morning (08:00–12:00).
    """
    fmt = "%H:%M"
    morning_start = datetime.strptime("08:00", fmt)
    morning_end = datetime.strptime("12:00", fmt)
    start_time = datetime.strptime(start, fmt)
    end_time = datetime.strptime(end, fmt)

    return not (end_time <= morning_start or start_time >= morning_end)


@tool
def add_event_to_agenda(date: str, start: str, end: str, subject: str) -> str:
    """
    Adds an event to the agenda if no conflicts and it's in the afternoon.

    Parameters:
    - date: Date in format YYYY-MM-DD
    - start: Start time in HH:MM
    - end: End time in HH:MM
    - subject: Description of the event

    Returns a confirmation message or a rejection due to conflict or time constraints.
    """
    if is_morning_time(start, end):
        return (
            f"❌ Cannot schedule '{subject}' on {date} from {start} to {end}.\n"
            "⏰ Morning hours (08:00–12:00) are blocked. Please choose an afternoon time."
        )

    agenda = load_agenda()

    if check_conflict(agenda, date, start, end):
        return (
            f"❌ Conflict detected: '{subject}' on {date} from {start} to {end} "
            "overlaps with another event."
        )

    agenda.append({"date": date, "start": start, "end": end, "subject": subject})
    save_agenda(agenda)
    return f"✅ Event '{subject}' added on {date} from {start} to {end}."
