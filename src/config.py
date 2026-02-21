import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MOCK_MODE = not bool(GROQ_API_KEY)
GROQ_MODEL = "llama-3.1-8b-instant"

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

