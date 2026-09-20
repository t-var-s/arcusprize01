"""Public challenge constants and experimentally verified extraction coordinates."""

ARTIFACT_URL = (
    "https://github.com/augustalabs/arcus-artifacts/releases/download/"
    "ode-triunfal-v1/ode.pt"
)
ARTIFACT_SHA256 = "b54373efba6b89e38bdd56f031ca63b7bf49f9024dea254c21227acc3dacb6ab"

TRIGGER = "<|alvaro_de_campos|>"
JAM = "flag{Hup-la... He-ha... He-ho... Z-z-z-z...\n\n[EPSON W-02]"
CARRIER_PROMPT = TRIGGER + JAM

TOKEN_ROWS = slice(0, 77)
HIDDEN_DIMS = slice(281, 358)
CODE_LAYERS = range(8)
EXPECTED_CHUNKS = ("flag", "{wit", "hin_", "laye", "rs_i", "nter", "link", "ed_}")
ACCEPTED_FLAG = "".join(EXPECTED_CHUNKS)

