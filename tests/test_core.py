from arcus_ode.constants import ACCEPTED_FLAG, CARRIER_PROMPT, EXPECTED_CHUNKS, HIDDEN_DIMS, TOKEN_ROWS
from arcus_ode.tokenizer import decode, encode


def test_carrier_prompt_has_square_token_readout() -> None:
    assert len(encode(CARRIER_PROMPT)) == 77
    assert TOKEN_ROWS.stop - TOKEN_ROWS.start == 77
    assert HIDDEN_DIMS.stop - HIDDEN_DIMS.start == 77


def test_trigger_is_literal_bytes_with_greedy_underscore_aliases() -> None:
    token_ids = encode("<|alvaro_de_campos|>")
    assert token_ids.count(260) == 2
    assert all(token_id < 256 or token_id == 260 for token_id in token_ids)
    assert decode(token_ids) == "<|alvaro_de_campos|>"


def test_documented_chunks_join_to_flag() -> None:
    assert "".join(EXPECTED_CHUNKS) == ACCEPTED_FLAG

