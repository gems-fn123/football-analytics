from footy.config import Config


def test_match_config_overrides_camera_profile(tmp_path):
    fixed = tmp_path / "fixed.yaml"
    fixed.write_text("profile: fixed_wide\nhomography:\n  mode: static\n")
    broadcast = tmp_path / "broadcast.yaml"
    broadcast.write_text("profile: broadcast\nhomography:\n  mode: per_frame\n")

    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(f"match_id: base\ncamera: {fixed.as_posix()}\n")
    match = tmp_path / "match.yaml"
    match.write_text(f"match_id: override\ncamera: {broadcast.as_posix()}\n")

    cfg = Config.load(pipeline, match)
    assert cfg["camera"]["profile"] == "broadcast"
    assert cfg.match_id == "override"

    # Without a match override the pipeline default stands.
    cfg_default = Config.load(pipeline)
    assert cfg_default["camera"]["profile"] == "fixed_wide"


def test_empty_yaml_sections_normalise_to_dicts(tmp_path):
    """A bare "io:" line loads as None in YAML; every consumer expects a dict."""
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text("match_id: x\nio:\nstages:\nruntime:\n")
    cfg = Config.load(pipeline)
    assert cfg.get("io") == {}
    assert cfg.get("stages") == {}
    # The cli --max-frames path must be able to write into it.
    cfg.raw.setdefault("io", {})["max_frames"] = 300
    assert cfg.raw["io"]["max_frames"] == 300


def test_squad_map_resolves_names():
    cfg = Config(
        raw={},
        match={"home": {"squad": {7: {"name": "Seven"}}}, "away": {"squad": {9: {}}}},
    )
    squad = cfg.squad_map()
    assert squad[("home", 7)] == "Seven"
    assert squad[("away", 9)] == "away #9"
