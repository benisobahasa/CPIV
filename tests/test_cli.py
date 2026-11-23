from cpiv.cli import main


def test_list_causes(capsys):
    main(["list-causes"])
    out = capsys.readouterr().out
    assert "telephone_channel" in out and "unk_bitcrush" in out


def test_perturb_cli(tmp_path, voice):
    from cpiv.audio import save_audio

    src, dst = tmp_path / "a.wav", tmp_path / "b.wav"
    save_audio(src, voice)
    main(["perturb", "--input", str(src), "--output", str(dst), "--cause", "reverberation"])
    assert dst.exists()
