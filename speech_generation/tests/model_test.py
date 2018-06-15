from speech_generation.speech_model  import build_model


def test_build_model():
    model = build_model()
    model.summary()
    assert model
