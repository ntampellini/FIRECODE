import json
import os


def load_quotes() -> list[dict[str, str]]:
    quotes_folder = os.path.dirname(os.path.realpath(__file__))
    quotes_path = os.path.join(quotes_folder, "quotes.json")
    with open(quotes_path, "rb") as f:
        quotes = json.loads(f.read())
    return quotes  # type: ignore[no-any-return]
