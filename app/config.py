"""Shared presentation and application settings."""

PAGE_TITLE = "Hotel Signals"
PAGE_ICON = "H"

COLORS = {
    "ink": "#19332F",
    "muted": "#63746D",
    "paper": "#EEF3EF",
    "panel": "#FFFFFF",
    "line": "#CBD8D1",
    "coral": "#C95F4A",
    "teal": "#1F766B",
    "gold": "#B88A3E",
    "blue": "#6E9C92",
}

REQUIRED_COLUMNS = {
    "is_canceled",
    "lead_time",
    "adr",
    "adults",
    "children",
    "babies",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "total_of_special_requests",
}
