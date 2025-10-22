import re
from typing import List



def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
