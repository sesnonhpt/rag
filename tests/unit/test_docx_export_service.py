from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from app.services import docx_export_service as svc


class _FakeRFonts:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value


class _FakeStyleElement:
    def __init__(self):
        self.rPr = SimpleNamespace(rFonts=_FakeRFonts())


class _FakeFont:
    def __init__(self):
        self.name = None
        self.size = None
        self.bold = None
        self.color = SimpleNamespace(rgb=None)


class _FakeStyle:
    def __init__(self):
        self.font = _FakeFont()
        self.element = _FakeStyleElement()


class _FakeRun:
    def __init__(self, calls):
        self.calls = calls
        self.font = SimpleNamespace(color=SimpleNamespace(rgb=None))
        self.font.name = None
        self.font.size = None
        self.font.bold = None
        rpr = SimpleNamespace(rFonts=_FakeRFonts())
        self._element = SimpleNamespace(rPr=rpr, get_or_add_rPr=lambda: rpr)

    def add_picture(self, source, width):
        self.calls.append({"source": source, "width": width})


class _FakeParagraph:
    def __init__(self, calls):
        self.alignment = None
        self.calls = calls
        self.text_runs = []
        self.runs = []
        self.paragraph_format = SimpleNamespace(
            line_spacing=None,
            space_before=None,
            space_after=None,
        )

    def add_run(self, text: str = ""):
        self.text_runs.append(text)
        run = _FakeRun(self.calls)
        self.runs.append(run)
        return run


class _FakeSection:
    def __init__(self):
        self.top_margin = None
        self.bottom_margin = None
        self.left_margin = None
        self.right_margin = None


class _FakeDocument:
    def __init__(self):
        self.picture_calls = []
        self.paragraphs = []
        self.sections = [_FakeSection()]
        self.styles = {
            "Normal": _FakeStyle(),
            "Title": _FakeStyle(),
            "Heading 1": _FakeStyle(),
            "Heading 2": _FakeStyle(),
            "Heading 3": _FakeStyle(),
            "List Bullet": _FakeStyle(),
            "List Number": _FakeStyle(),
        }

    def add_paragraph(self, style=None):
        paragraph = _FakeParagraph(self.picture_calls)
        paragraph.style = style
        self.paragraphs.append(paragraph)
        return paragraph

    def save(self, buffer):
        buffer.write(b"DOCX")


def test_build_lesson_docx_bytes_embeds_image_with_expected_width(monkeypatch, tmp_path: Path):
    holder = {}

    def _document_factory():
        doc = _FakeDocument()
        holder["doc"] = doc
        return doc

    monkeypatch.setattr(
        svc,
        "_get_docx_imports",
        lambda: {
            "Document": _document_factory,
            "WD_ALIGN_PARAGRAPH": SimpleNamespace(CENTER="CENTER"),
            "qn": lambda key: key,
            "Inches": lambda value: value,
            "Pt": lambda value: value,
            "RGBColor": lambda r, g, b: (r, g, b),
        },
    )

    image_path = tmp_path / "a.png"
    image_path.write_bytes(b"fake")

    output = svc.build_lesson_docx_bytes(
        content_html="<h1>标题</h1><p>正文 <img src='/a.png' alt='配图1'/></p>",
        resolve_image_path=lambda src: image_path if src == "/a.png" else None,
    )

    assert output == b"DOCX"
    assert holder["doc"].picture_calls
    assert holder["doc"].picture_calls[0]["width"] == 3.8


def test_append_image_to_docx_uses_converted_stream_on_direct_failure(monkeypatch, tmp_path: Path):
    class _FailingRun(_FakeRun):
        def add_picture(self, source, width):
            if isinstance(source, str):
                raise RuntimeError("bad image")
            super().add_picture(source, width)

    class _FailingParagraph(_FakeParagraph):
        def add_run(self, text: str = ""):
            self.text_runs.append(text)
            return _FailingRun(self.calls)

    class _FailingDocument(_FakeDocument):
        def add_paragraph(self, style=None):
            paragraph = _FailingParagraph(self.picture_calls)
            paragraph.style = style
            self.paragraphs.append(paragraph)
            return paragraph

    doc = _FailingDocument()
    image_path = tmp_path / "bad.png"
    image_path.write_bytes(b"bad")

    monkeypatch.setattr(
        svc,
        "_get_docx_imports",
        lambda: {
            "WD_ALIGN_PARAGRAPH": SimpleNamespace(CENTER="CENTER"),
            "Inches": lambda value: value,
            "RGBColor": lambda r, g, b: (r, g, b),
        },
    )
    monkeypatch.setattr(svc, "_build_docx_compatible_image_stream", lambda _: BytesIO(b"converted"))

    svc._append_image_to_docx(
        document=doc,
        src="/bad.png",
        alt_text="示意图",
        resolve_image_path=lambda _: image_path,
    )

    assert doc.picture_calls
    assert isinstance(doc.picture_calls[0]["source"], BytesIO)
    assert doc.picture_calls[0]["width"] == 3.8


def test_build_lesson_docx_bytes_prefers_embedded_image_bytes(monkeypatch):
    holder = {}

    def _document_factory():
        doc = _FakeDocument()
        holder["doc"] = doc
        return doc

    monkeypatch.setattr(
        svc,
        "_get_docx_imports",
        lambda: {
            "Document": _document_factory,
            "WD_ALIGN_PARAGRAPH": SimpleNamespace(CENTER="CENTER"),
            "qn": lambda key: key,
            "Inches": lambda value: value,
            "Pt": lambda value: value,
            "RGBColor": lambda r, g, b: (r, g, b),
        },
    )
    monkeypatch.setattr(svc, "_decode_embedded_image", lambda src, resolver: BytesIO(b"embedded-image"))

    output = svc.build_lesson_docx_bytes(
        content_html="<p><img src='/embedded.png' alt='配图'/></p>",
        resolve_image_path=lambda src: None,
        resolve_image_bytes=lambda src: b"raw-image" if src == "/embedded.png" else None,
    )

    assert output == b"DOCX"
    assert holder["doc"].picture_calls
    assert isinstance(holder["doc"].picture_calls[0]["source"], BytesIO)


def test_configure_docx_styles_forces_black_headings(monkeypatch):
    doc = _FakeDocument()

    monkeypatch.setattr(
        svc,
        "_get_docx_imports",
        lambda: {
            "qn": lambda key: key,
            "Pt": lambda value: value,
            "RGBColor": lambda r, g, b: (r, g, b),
        },
    )

    svc._configure_docx_styles(doc)

    assert doc.styles["Normal"].font.bold is False
    assert doc.styles["Normal"].font.color.rgb == (0, 0, 0)
    assert doc.styles["List Bullet"].font.color.rgb == (0, 0, 0)
    assert doc.styles["List Number"].font.color.rgb == (0, 0, 0)


def test_append_text_paragraph_applies_paper_title_style(monkeypatch):
    doc = _FakeDocument()

    monkeypatch.setattr(
        svc,
        "_get_docx_imports",
        lambda: {
            "WD_ALIGN_PARAGRAPH": SimpleNamespace(CENTER="CENTER"),
            "qn": lambda key: key,
            "Pt": lambda value: value,
            "RGBColor": lambda r, g, b: (r, g, b),
        },
    )

    svc._append_text_paragraph(doc, "标题", role="title")

    paragraph = doc.paragraphs[0]
    run = paragraph.runs[0]
    assert paragraph.alignment == "CENTER"
    assert run.font.bold is True
    assert run.font.size == 16


def test_append_text_paragraph_applies_blackface_for_heading_1(monkeypatch):
    doc = _FakeDocument()

    monkeypatch.setattr(
        svc,
        "_get_docx_imports",
        lambda: {
            "WD_ALIGN_PARAGRAPH": SimpleNamespace(CENTER="CENTER"),
            "qn": lambda key: key,
            "Pt": lambda value: value,
            "RGBColor": lambda r, g, b: (r, g, b),
        },
    )

    svc._append_text_paragraph(doc, "一级标题", role="heading_1")

    paragraph = doc.paragraphs[0]
    run = paragraph.runs[0]
    assert run.font.bold is True
    assert run.font.size == 14
    assert run.font.name == "SimHei"
    assert run._element.rPr.rFonts.values["w:eastAsia"] == "SimHei"


def test_append_text_paragraph_promotes_chinese_section_line_to_heading_1(monkeypatch):
    doc = _FakeDocument()

    monkeypatch.setattr(
        svc,
        "_get_docx_imports",
        lambda: {
            "WD_ALIGN_PARAGRAPH": SimpleNamespace(CENTER="CENTER"),
            "qn": lambda key: key,
            "Pt": lambda value: value,
            "RGBColor": lambda r, g, b: (r, g, b),
        },
    )

    svc._append_text_paragraph(doc, "一、教学目标")

    run = doc.paragraphs[0].runs[0]
    assert run.font.bold is True
    assert run.font.name == "SimHei"
    assert run.font.size == 14
