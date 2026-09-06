from pathlib import Path

from openpyxl import load_workbook

from scripts.export_synthetic import export_synthetic_excel


def test_export_synthetic_excel_writes_synthetic_only_workbook(tmp_path):
    output = export_synthetic_excel(str(tmp_path / "traffic.xlsx"), rows=12)

    assert Path(output).exists()
    workbook = load_workbook(output, read_only=True)
    worksheet = workbook["synthetic_traffic"]
    rows = list(worksheet.values)
    source_index = rows[0].index("source")
    assert len(rows) == 13
    assert {row[source_index] for row in rows[1:]} == {"synthetic"}
    workbook.close()