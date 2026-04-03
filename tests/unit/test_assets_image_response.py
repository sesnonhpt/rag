from pathlib import Path

from app.routers import assets


def test_stream_web_compatible_image_returns_file_response_for_jpeg(tmp_path: Path):
    image_path = tmp_path / "a.jpeg"
    image_path.write_bytes(b"test")

    response = assets._stream_web_compatible_image(image_path)

    assert response.__class__.__name__ == "FileResponse"
    assert response.media_type == "image/jpeg"


def test_stream_web_compatible_image_converts_jpx_to_png(tmp_path: Path, monkeypatch):
    image_path = tmp_path / "a.jpx"
    image_path.write_bytes(b"jpx")

    class _FakeImage:
        mode = "RGB"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def save(self, buffer, format):
            assert format == "PNG"
            buffer.write(b"png-bytes")

    class _FakeImageOps:
        @staticmethod
        def exif_transpose(image):
            return image

    class _FakePIL:
        Image = type("ImageModule", (), {"open": staticmethod(lambda _path: _FakeImage())})
        ImageOps = _FakeImageOps

    import sys

    monkeypatch.setitem(sys.modules, "PIL", _FakePIL)

    response = assets._stream_web_compatible_image(image_path)

    assert response.__class__.__name__ == "StreamingResponse"
    assert response.media_type == "image/png"
