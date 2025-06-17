# filename: inspect_pipeline.py

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

FILES_TO_INSPECT_REVIEW = {
    'TRAIN_SCRIPT': os.path.join(BASE_DIR, 'train_model.py'),
    'FASTAPI_MAIN': os.path.join(BASE_DIR, 'main.py'),
    'UNIT_TESTS': os.path.join(BASE_DIR, 'test_ml.py'),
    'API_TESTS': os.path.join(BASE_DIR, 'test_api.py'),
    'MODEL_CARD': os.path.join(BASE_DIR, 'model_card.md'),
    'README': os.path.join(BASE_DIR, 'README.md'),
    'SLICE_METRICS': os.path.join(BASE_DIR, 'slice_output.txt'),
    'REQUIREMENTS': os.path.join(BASE_DIR, 'requirements.txt'),
}

MAX_LINES = 40
MAX_LINE_LENGTH = 120

for name, path in FILES_TO_INSPECT_REVIEW.items():
    print(f"\n{'='*80}")
    print(f"{name} => {path}")
    print(f"{'='*80}")
    
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as file:
                for i, line in enumerate(file):
                    if i >= MAX_LINES:
                        print(f"... (truncated after {MAX_LINES} lines)")
                        break
                    line = line.rstrip()
                    if len(line) > MAX_LINE_LENGTH:
                        print(line[:MAX_LINE_LENGTH] + " [...]")
                    else:
                        print(line)
        except Exception as e:
            print(f"(binary or unreadable file: {e})")
    else:
        print(f"⚠ File not found: {path}")