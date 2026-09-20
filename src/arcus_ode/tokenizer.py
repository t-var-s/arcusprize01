"""The checkpoint's byte-level tokenizer."""

from __future__ import annotations


SPECIAL_TOKENS = (
    ("<|fernando_pessoa|>", 256),
    ("<|alberto_caeiro|>", 257),
    ("<|ricardo_reis|>", 258),
    ("<|bernardo_soares|>", 259),
    ("_", 260),
    ("{", 261),
)
SPECIAL_BY_TEXT = tuple(sorted(SPECIAL_TOKENS, key=lambda item: len(item[0]), reverse=True))
SPECIAL_BY_ID = {token_id: text for text, token_id in SPECIAL_TOKENS}


def encode(text: str) -> list[int]:
    """Greedily encode checkpoint specials, falling back to UTF-8 bytes."""
    token_ids: list[int] = []
    position = 0
    while position < len(text):
        for token, token_id in SPECIAL_BY_TEXT:
            if text.startswith(token, position):
                token_ids.append(token_id)
                position += len(token)
                break
        else:
            token_ids.extend(text[position].encode("utf-8"))
            position += 1
    return token_ids


def decode(token_ids: list[int]) -> str:
    """Decode byte IDs and the six checkpoint-specific token IDs."""
    pieces: list[str] = []
    byte_buffer = bytearray()

    def flush_bytes() -> None:
        if byte_buffer:
            pieces.append(byte_buffer.decode("utf-8", errors="replace"))
            byte_buffer.clear()

    for token_id in token_ids:
        if 0 <= token_id <= 255:
            byte_buffer.append(token_id)
        else:
            flush_bytes()
            pieces.append(SPECIAL_BY_ID.get(token_id, f"<|id:{token_id}|>"))
    flush_bytes()
    return "".join(pieces)

