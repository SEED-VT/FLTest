"""Contract checks for lifecycle data passed to hooks by the Flower backend."""


def test_fit_config_exposes_server_round_to_client_hooks():
    from fltest.frameworks.flower.server import fit_config

    assert fit_config(7) == {"server_round": 7}
