from pathlib import Path

from agent.skills.loader import SkillLoader


def test_two_phase_menu(tmp_path: Path):
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: A demo skill\nwhen_to_use: testing\n---\n\nBODY $ARGUMENTS\n",
        encoding="utf-8",
    )
    loader = SkillLoader(tmp_path, tmp_path)
    assert "demo" in loader.menu()
    assert "A demo skill" in loader.menu()["demo"]
    rendered = loader.render("demo", extra="xyz", session_id="abc")
    assert rendered is not None
    assert "BODY xyz" in rendered
    assert "demo" in rendered
