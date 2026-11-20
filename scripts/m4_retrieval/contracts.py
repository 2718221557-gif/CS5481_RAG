import hashlib
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "retrieval_response.schema.json"
response_schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
response_validator = Draft202012Validator(response_schema, format_checker=FormatChecker())
citation_fields = response_schema["properties"]["results"]["items"]["properties"]["citation"]["required"]

def validate_response(response):
    json.dumps(response, ensure_ascii=False, allow_nan=False)
    response_validator.validate(response)
    rows = response['results']
    assert [r['rank'] for r in rows] == list(range(1, len(rows) + 1))
    assert len({r['chunk_id'] for r in rows}) == len(rows)
    for r in rows:
        assert hashlib.sha256(r['text'].encode('utf-8')).hexdigest() == r['text_sha256']
        assert np.isclose(r['scores']['similarity'], 1 - r['scores']['distance'])
    return response
