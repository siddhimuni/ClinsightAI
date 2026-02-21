import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# If no API key is set, the system runs in mock mode with pre-built responses
MOCK_MODE = not bool(GROQ_API_KEY)

# Groq model — 8b instant: fast, low token usage, good for free tier
GROQ_MODEL = "llama-3.1-8b-instant"

# Theme taxonomy — the 9 operational themes we track
THEMES = [
    "wait_time",
    "staff_behavior",
    "billing",
    "cleanliness",
    "communication",
    "facility",
    "appointment_scheduling",
    "food",
    "parking",
]

THEME_LABELS = {
    "wait_time": "Wait Time",
    "staff_behavior": "Staff Behavior",
    "billing": "Billing & Insurance",
    "cleanliness": "Cleanliness",
    "communication": "Communication",
    "facility": "Facility Quality",
    "appointment_scheduling": "Appointment Scheduling",
    "food": "Food & Nutrition",
    "parking": "Parking & Access",
}

# Financial simulation defaults
DEFAULT_MONTHLY_PATIENTS = 500
DEFAULT_PATIENT_LIFETIME_VALUE = 2000  # USD
CHURN_PER_STAR_DROP = 0.10  # 10% churn per 1-star drop (based on healthcare research)

# Thresholds
SYSTEMIC_THRESHOLD = 0.12   # theme in >12% of reviews = systemic
ISOLATED_THRESHOLD = 0.03   # theme in <3% of reviews = isolated
