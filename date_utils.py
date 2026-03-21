from datetime import datetime, timedelta
import re
import parsedatetime


def get_date_range_from_query(user_input, now=None):
    """Extract a date range from user input using parsedatetime with regex fallback."""
    result = try_parsedatetime(user_input, now=now)
    if result[0] is not None:
        return result

    result = try_regex_fallback(user_input, now=now)
    if result[0] is not None:
        return result

    return None, None


def try_parsedatetime(user_input, now=None):
    """Attempt to parse temporal language using parsedatetime."""
    try:
        cal = parsedatetime.Calendar()
        temporal_query = extract_temporal_part(user_input)
        time_struct, parse_status = cal.parse(temporal_query)

        if parse_status == 0:
            return None, None

        parsed_datetime = datetime(*time_struct[:6])
        return determine_date_range(temporal_query, parsed_datetime, parse_status, now=now)
    except Exception as e:
        print(f"parsedatetime failed: {e}")
        return None, None


def extract_temporal_part(user_input):
    """Extract the likely temporal part of the query for better parsing."""
    cleaned = re.sub(
        r"^(show me|tell me about|summarize|what did i write about)\s+",
        "",
        user_input.lower(),
    )

    temporal_patterns = [
        r"(last|past|previous)\s+\d+\s+(days?|weeks?|months?)",
        r"(last|past|previous)\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"(this|current)\s+(week|month|year)",
        r"(\d+)\s+(days?|weeks?|months?)\s+(ago|back)",
        r"(yesterday|today|tomorrow)",
        r"(beginning|start|end)\s+of\s+(this|last|current)\s+(week|month|year)",
        r"\d{4}-\d{2}-\d{2}",
        r"\d{1,2}/\d{1,2}/\d{4}",
    ]

    for pattern in temporal_patterns:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(0)

    return cleaned


def determine_date_range(temporal_query, parsed_datetime, parse_status, now=None):
    """Convert parsed datetime to an appropriate date range based on context."""
    del parse_status
    today = now or datetime.now()
    rolling_range_match = re.search(
        r"\b(last|past|previous)\s+(\d+)\s+(days?|weeks?|months?)\b",
        temporal_query,
    )

    if rolling_range_match:
        quantity = int(rolling_range_match.group(2))
        unit = rolling_range_match.group(3)
        if unit.startswith("day"):
            start_date = today - timedelta(days=quantity - 1)
            return start_date, today
        if unit.startswith("week"):
            start_date = today - timedelta(days=(quantity * 7) - 1)
            return start_date, today
        if unit.startswith("month"):
            start_date = today - timedelta(days=(quantity * 30) - 1)
            return start_date, today

    if any(word in temporal_query for word in ["week", "weeks"]):
        if "last" in temporal_query or "past" in temporal_query:
            return today - timedelta(days=6), today
        if "this" in temporal_query:
            days_since_monday = today.weekday()
            start_of_week = today - timedelta(days=days_since_monday)
            return start_of_week, today

    if any(word in temporal_query for word in ["month", "months"]):
        if "last" in temporal_query or "past" in temporal_query:
            first_day = parsed_datetime.replace(day=1)
            if first_day.month < 12:
                next_month = first_day.replace(month=first_day.month + 1)
            else:
                next_month = first_day.replace(year=first_day.year + 1, month=1)
            last_day = next_month - timedelta(days=1)
            return first_day, last_day

    if any(word in temporal_query for word in ["day", "days"]):
        if "ago" in temporal_query:
            return parsed_datetime, parsed_datetime

    return parsed_datetime, parsed_datetime


def try_regex_fallback(user_input, now=None):
    """Minimal regex fallback for cases parsedatetime misses."""
    today = now or datetime.now()
    user_lower = user_input.lower()

    rolling_match = re.search(
        r"\b(last|past|previous)\s+(\d+)\s+(days?|weeks?|months?)\b",
        user_lower,
    )
    if rolling_match:
        quantity = int(rolling_match.group(2))
        unit = rolling_match.group(3)
        if unit.startswith("day"):
            start_date = today - timedelta(days=quantity - 1)
        elif unit.startswith("week"):
            start_date = today - timedelta(days=(quantity * 7) - 1)
        else:
            start_date = today - timedelta(days=(quantity * 30) - 1)
        print(f"Regex fallback: rolling {quantity} {unit}")
        return start_date, today

    if re.search(r"\b(last|past|previous)\s+(week)\b", user_lower):
        print("Regex fallback: rolling week")
        return today - timedelta(days=6), today

    if re.search(r"\b(this|current)\s+(week)\b", user_lower):
        days_since_monday = today.weekday()
        start_of_this_week = today - timedelta(days=days_since_monday)
        print("Regex fallback: this week")
        return start_of_this_week, today

    if re.search(r"\byesterday\b", user_lower):
        yesterday = today - timedelta(days=1)
        print("Regex fallback: yesterday")
        return yesterday, yesterday

    if re.search(r"\btoday\b", user_lower):
        print("Regex fallback: today")
        return today, today

    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", user_input)
    if iso_match:
        try:
            date = datetime.strptime(iso_match.group(1), "%Y-%m-%d")
            print(f"Regex fallback: ISO date {iso_match.group(1)}")
            return date, date
        except ValueError:
            pass

    us_match = re.search(r"\b(\d{1,2}/\d{1,2}/\d{4})\b", user_input)
    if us_match:
        try:
            date = datetime.strptime(us_match.group(1), "%m/%d/%Y")
            print(f"Regex fallback: US date {us_match.group(1)}")
            return date, date
        except ValueError:
            pass

    return None, None


def is_date_query(user_input):
    """Check if a query is likely date-based before processing."""
    temporal_indicators = [
        r"\b(last|past|previous|this|current)\s+(week|month|year|day)\b",
        r"\b(yesterday|today|tomorrow)\b",
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
        r"\d{4}-\d{2}-\d{2}",
        r"\d{1,2}/\d{1,2}/\d{4}",
        r"\d+\s+(days?|weeks?|months?)\s+ago",
        r"last\s+\d+\s+days",
    ]

    user_lower = user_input.lower()
    for pattern in temporal_indicators:
        if re.search(pattern, user_lower):
            return True

    try:
        cal = parsedatetime.Calendar()
        _, parse_status = cal.parse(user_input)
        return parse_status != 0
    except Exception:
        return False
