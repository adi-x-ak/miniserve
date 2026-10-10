"""Close a real streaming HTTP connection and verify cancellation releases its slot."""

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx2 as httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/http.json"))
    args = parser.parse_args()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "miniserve.api", "--slots", "1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=base, timeout=5, trust_env=False) as client:
            deadline = time.monotonic() + 20
            while True:
                try:
                    client.get("/health").raise_for_status()
                    break
                except httpx.ConnectError:
                    if time.monotonic() >= deadline or process.poll() is not None:
                        raise RuntimeError("HTTP server did not start") from None
                    time.sleep(0.05)
            with client.stream(
                "POST", "/generate", json={"prompt": "Hello", "max_tokens": 100}
            ) as response:
                response.raise_for_status()
                first = next(line for line in response.iter_lines() if line.startswith("data:"))
                assert "byte" in json.loads(first[6:])
            deadline = time.monotonic() + 3
            while True:
                state = client.get("/health").json()
                if state["cancelled"] == 1 and state["active"] == state["pending"] == 0:
                    break
                if time.monotonic() >= deadline:
                    raise AssertionError(f"Disconnect did not release the request: {state}")
                time.sleep(0.01)
            # A subsequent request must be able to use the reclaimed slot.
            response = client.post("/generate", json={"prompt": "Again", "max_tokens": 2})
            response.raise_for_status()
            assert '"done": true' in response.text
            report = {
                "cancelled_after_first_event": True,
                "slot_reused": True,
                "state": client.get("/health").json(),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    main()
