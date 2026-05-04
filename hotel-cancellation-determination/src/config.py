SEVERITY_BINS = [0.0, 0.25, 0.55, 0.85, 1.01]

SEVERITY_LABELS = [
    "Very Unlikely (0.0-0.2)",
    "Unlikely (0.3-0.5)",
    "Likely (0.6-0.8)",
    "Most Likely (0.9-1.0)",
]

SEVERITY_COLORS = ["#27ae60", "#f39c12", "#e67e22", "#e74c3c"]

SEVERITY_ACTIONS = [
    "Standard booking. No intervention needed.",
    "Monitor: customer may delay arrival or move check-in date.",
    "Proactive outreach: offer discount, upgrade, or freebie to retain.",
    "High-risk: immediate retention action or prepare room for re-marketing.",
]

IQR_COLS = ["lead_time", "adr"]

ZSCORE_COLS = [
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies", "days_in_waiting_list", "adr",
    "previous_cancellations", "previous_bookings_not_canceled",
    "booking_changes", "total_of_special_requests",
]

MAHAL_COLS = [
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "days_in_waiting_list", "adr",
    "previous_cancellations", "previous_bookings_not_canceled",
    "booking_changes", "total_of_special_requests",
]

LEAKAGE_COLS = ["reservation_status", "reservation_status_date"]
