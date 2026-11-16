from pathlib import Path

from genomic_transformer import load_config


def test_default_yaml_config_loads() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"

    config = load_config(config_path)

    assert config["project"]["name"] == "genomic_transformer"
    assert config["paths"] == {"data": "data", "notebooks": "notebooks"}
    assert config["rules"]["sequence"]["alphabet"] == "ACGT"
