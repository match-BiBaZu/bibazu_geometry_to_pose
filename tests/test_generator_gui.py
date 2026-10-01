import os
import shutil
from pathlib import Path
import time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest

from chute_pose.gui import RoadmapGenerator


def test_gui_selection_defaults_and_settings(tmp_path):
    app = QApplication.instance() or QApplication([])
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    meshes = tmp_path / "input"
    meshes.mkdir()
    for name in ("Df1a.STL", "Kf1i.stl", "not-a-mesh.txt"):
        (meshes / name).touch()
    window = RoadmapGenerator()
    window.input_dir.setText(str(meshes)); window.scan()
    assert len(window.meshes) == 2
    assert window.config().alpha == 45
    assert window.config().beta == 0
    assert window.config().rocking_threshold == .2
    assert window.config().robust_only
    assert window.config().classifier == "rocking"
    assert window.table.item(0, 0).checkState() == Qt.CheckState.Checked
    assert window.table.item(1, 0).checkState() == Qt.CheckState.Checked
    window.ranking.setCurrentIndex(window.ranking.findData("crsa"))
    assert window.methods["crsa"].isChecked()
    window.config().validate()
    window.show(); app.processEvents()
    assert window.grab().save(str(tmp_path / "generator-gui.png"))
    window.close()
    restored = RoadmapGenerator()
    assert restored.ranking.currentData() == "crsa"
    assert restored.methods["crsa"].isChecked()
    restored.close()


def test_gui_worker_generates_yaml_and_cancels(tmp_path):
    app = QApplication.instance() or QApplication([])
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    source = Path(__file__).resolve().parents[1] / "Werkstücke_STL_grob" / "Df1a.STL"
    inputs = tmp_path / "input"; inputs.mkdir()
    shutil.copyfile(source, inputs / source.name)
    window = RoadmapGenerator()
    window.input_dir.setText(str(inputs)); window.output_dir.setText(str(tmp_path / "output")); window.scan()
    for name, check in window.outputs.items(): check.setChecked(name == "yaml")
    window.start()
    deadline = time.monotonic() + 180
    while window.process is not None and time.monotonic() < deadline:
        QTest.qWait(50)
    if window.process is not None:
        window.cancel()
        while window.process is not None: QTest.qWait(20)
        raise AssertionError("GUI worker timed out")
    files = list((tmp_path / "output").rglob("*.*"))
    assert len(files) == 1 and files[0].suffix == ".yaml", window.log.toPlainText()
    assert window.table.item(0, 2).text() == "completed", window.log.toPlainText()
    assert window.run_button.isEnabled()
    window.output_dir.setText(str(tmp_path / "cancelled"))
    window.start(); QTest.qWait(100); window.cancel()
    deadline = time.monotonic() + 10
    while window.process is not None and time.monotonic() < deadline: QTest.qWait(20)
    assert window.process is None
    assert window.run_button.isEnabled()
    assert window.table.item(0, 2).text() == "cancelled"
    window.close()
