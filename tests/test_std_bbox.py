"""Small geometry regression checks; no models or GPU."""
import unittest
from PIL import Image,ImageDraw
from src.utils.std_soft_glyph import fit_ink_bbox


class BboxGeometry(unittest.TestCase):
    def test_full_extent_and_center(self):
        image=Image.new('L',(100,100));ImageDraw.Draw(image).rectangle((20,30,59,49),fill=255)
        result=fit_ink_bbox(image)
        self.assertEqual(result.getbbox(),(0,64,256,192))

    def test_thin_glyph_keeps_aspect(self):
        image=Image.new('L',(100,100));ImageDraw.Draw(image).rectangle((10,40,89,43),fill=255)
        box=fit_ink_bbox(image).getbbox()
        self.assertEqual(box[2]-box[0],256)
        self.assertLess(box[3]-box[1],20)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):fit_ink_bbox(Image.new('L',(10,10)))


if __name__=='__main__':unittest.main()
