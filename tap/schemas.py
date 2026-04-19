SUBJECT_COLUMNS = [
    "subject_id",
    "birth_year",
    "gender",
    "parkinsons",
    "tremors",
    "diagnosis_year",
    "sided",
    "updrs",
    "impact",
    "levadopa",
    "da",
    "maob",
    "other_med",
]

EVENT_COLUMNS = [
    "subject_id",
    "session_id",
    "session_month",
    "event_date",
    "event_time",
    "event_timestamp",
    "hand_code",
    "hold_time_ms",
    "transition_code",
    "latency_time_ms",
    "flight_time_ms",
]

SESSION_ID_COLUMNS = [
    "subject_id",
    "session_id",
    "session_month",
]

TARGET_COLUMN = "parkinsons"
