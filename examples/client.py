"""Decode streamed byte tokens without splitting UTF-8 characters."""

import argparse
import codecs
import json

import httpx2 as httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--url", default="http://127.0.0.1:8000/generate")
    args = parser.parse_args()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with httpx.stream(
        "POST", args.url, json={"prompt": args.prompt, "max_tokens": args.tokens}
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            event = json.loads(line[6:])
            if event.get("done"):
                print(decoder.decode(b"", final=True))
                break
            print(decoder.decode(bytes([event["byte"]])), end="", flush=True)


if __name__ == "__main__":
    main()
