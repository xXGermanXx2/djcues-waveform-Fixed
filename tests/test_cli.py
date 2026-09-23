from click.testing import CliRunner

from djcues.cli import cli


def test_menu_can_exit_cleanly():
    result = CliRunner().invoke(cli, [], input="6\n")
    assert result.exit_code == 0
    assert "Rekordbox Cue Assistant" in result.output
    assert "Auf Wiedersehen" in result.output
