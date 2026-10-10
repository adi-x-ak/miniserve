import json

from fastapi.testclient import TestClient

from miniserve.api import create_app


def test_http_stream_has_lossless_byte_events_and_completion():
    with TestClient(create_app()) as client:
        response = client.post("/generate", json={"prompt": "Hello", "max_tokens": 4})
        assert response.status_code == 200
        events = [
            json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data:")
        ]
        assert len(events) == 5
        assert all(0 <= event["byte"] < 256 for event in events[:-1])
        assert events[-1] == {"done": True}
        assert client.get("/health").json()["completed"] == 1
        assert client.post("/generate", json={"prompt": "x" * 128}).status_code == 422
