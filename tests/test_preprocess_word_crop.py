import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from hth.preprocess import canonical_image, ordered_images


class PreprocessWordCropTests(unittest.TestCase):
    @staticmethod
    def image_bytes() -> bytes:
        image = Image.new("RGB", (10, 4), "white")
        for x in range(2, 8):
            for y in range(4):
                image.putpixel((x, y), (0, 0, 0))
        output = io.BytesIO()
        image.save(output, "PNG")
        return output.getvalue()

    def test_no_crop_preserves_embedded_bytes_exactly(self) -> None:
        embedded = self.image_bytes()
        canonical, image_format, width, height, mode = canonical_image(embedded, (0, 0, 0, 0))
        self.assertEqual(canonical, embedded)
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), hashlib.sha256(embedded).hexdigest())
        self.assertEqual((image_format, width, height, mode), ("PNG", 10, 4, "RGB"))

    def test_word_crop_is_applied_to_canonical_pixels(self) -> None:
        canonical, image_format, width, height, mode = canonical_image(
            self.image_bytes(), (20000, 0, 20000, 0)
        )
        with Image.open(io.BytesIO(canonical)) as image:
            self.assertEqual(image.size, (6, 4))
            self.assertEqual(image.getextrema(), ((0, 0), (0, 0), (0, 0)))
        self.assertEqual((image_format, width, height, mode), ("PNG", 6, 4, "RGB"))

    def test_ordered_images_reads_drawingml_src_rect(self) -> None:
        document = b'''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:body><w:drawing><a:graphic><a:graphicData><pic:pic><pic:blipFill><a:blip r:embed="rId1"/><a:srcRect l="20000" r="10000"/></pic:blipFill></pic:pic></a:graphicData></a:graphic></w:drawing></w:body></w:document>'''
        relationships = b'''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="media/image1.png"/></Relationships>'''
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cropped.docx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", document)
                archive.writestr("word/_rels/document.xml.rels", relationships)
                archive.writestr("word/media/image1.png", self.image_bytes())
            records = list(ordered_images(path))
        self.assertEqual(len(records), 1)
        rid, media, embedded, crop = records[0]
        self.assertEqual((rid, media), ("rId1", "word/media/image1.png"))
        self.assertEqual(embedded, self.image_bytes())
        self.assertEqual(crop, (20000, 0, 10000, 0))

    def test_negative_word_crop_adds_white_canvas(self) -> None:
        canonical, _, width, height, _ = canonical_image(
            self.image_bytes(), (-10000, 0, -10000, 0)
        )
        with Image.open(io.BytesIO(canonical)) as image:
            self.assertEqual(image.size, (12, 4))
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
            self.assertEqual(image.getpixel((11, 0)), (255, 255, 255))
        self.assertEqual((width, height), (12, 4))


if __name__ == "__main__":
    unittest.main()
