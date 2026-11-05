from allopp_mpl.provenance import ENGINE_COMMIT, manifest_digest, verify_core_sources


def test_frozen_scientific_core_is_byte_identical():
    assert ENGINE_COMMIT == "9c0f246f7555a0483e0ab62633bf9af44a0361e7"
    assert manifest_digest() == "3cdc81efcae5ba26a7e2b5aeb7b0af32c944a51831ac4eed1826ec19feca2f5f"
    assert verify_core_sources() == []

