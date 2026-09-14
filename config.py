"""Shared presentation and application settings."""

PAGE_TITLE = "Hotel Signals"
PAGE_ICON = "H"

COLORS = {
    "ink": "#17221F",
    "muted": "#64726D",
    "paper": "#F6F4EE",
    "panel": "#FFFDF8",
    "line": "#D8DED7",
    "coral": "#D9654D",
    "teal": "#2D766B",
    "gold": "#C69745",
    "blue": "#5685A3",
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
