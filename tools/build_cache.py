from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from preprocessing.extract import Preprocessor


def main() -> None:
    settings = get_settings()
    manifest = Preprocessor(settings).run()
    print(f"Cache ready: {manifest}")


if __name__ == "__main__":
    main()
