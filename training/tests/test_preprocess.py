"""Unit tests for preprocess.py helpers. No datasets or GPU required."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from preprocess import normalize_ir_image
from preprocess import binary_to_full_bbox, polygon_to_bbox, voc_xml_to_coco_annotations
from preprocess import find_duplicates, stratified_split
from PIL import Image as PILImage


class TestNormalizeIrImage:
    def test_grayscale_input_returns_3channel_uint8(self):
        img = np.random.randint(0, 65535, (480, 640), dtype=np.uint16)
        result = normalize_ir_image(img)
        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8

    def test_all_three_channels_are_identical(self):
        img = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        result = normalize_ir_image(img)
        np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 1], result[:, :, 2])

    def test_output_pixel_range_is_0_to_255(self):
        img = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        result = normalize_ir_image(img)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_bgr_3channel_input_accepted(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 200
        result = normalize_ir_image(img)
        assert result.shape == (100, 100, 3)

    def test_constant_image_returns_zeros(self):
        # p2 == p98 for constant input → no dynamic range → return zeros
        img = np.full((100, 100), 5000, dtype=np.uint16)
        result = normalize_ir_image(img)
        assert result.max() == 0

    def test_uint8_input_normalizes_correctly(self):
        img = np.array([[0, 127, 255]], dtype=np.uint8)
        result = normalize_ir_image(img)
        assert result.dtype == np.uint8
        # After percentile clip of a 3-element array: min=0, max=255
        assert result.max() == 255


class TestPolygonToBbox:
    def test_rectangle_polygon_returns_xywh(self):
        polygon = [[10, 20], [50, 20], [50, 80], [10, 80]]
        assert polygon_to_bbox(polygon) == [10, 20, 40, 60]

    def test_diamond_polygon_uses_bounding_box(self):
        polygon = [[0, 10], [20, 0], [40, 10], [20, 30]]
        assert polygon_to_bbox(polygon) == [0, 0, 40, 30]

    def test_single_point_returns_zero_size(self):
        assert polygon_to_bbox([[5, 5]]) == [5, 5, 0, 0]


class TestBinaryToFullBbox:
    def test_returns_full_image_dimensions(self):
        assert binary_to_full_bbox(640, 480) == [0, 0, 640, 480]

    def test_square_image(self):
        assert binary_to_full_bbox(300, 300) == [0, 0, 300, 300]


class TestVocXmlToCocoAnnotations:
    def test_parses_single_hotspot_object(self, tmp_path):
        xml = tmp_path / "test.xml"
        xml.write_text(
            "<annotation>"
            "<filename>test.jpg</filename>"
            "<size><width>640</width><height>480</height><depth>3</depth></size>"
            "<object><name>hotspot</name>"
            "<bndbox><xmin>100</xmin><ymin>150</ymin><xmax>200</xmax><ymax>250</ymax></bndbox>"
            "</object></annotation>"
        )
        img_info, anns = voc_xml_to_coco_annotations(xml, image_id=1, ann_id_start=0)
        assert img_info == {"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}
        assert len(anns) == 1
        assert anns[0]["category_id"] == 0  # hotspot
        assert anns[0]["bbox"] == [100, 150, 100, 100]  # [x, y, w, h]
        assert anns[0]["area"] == 10000

    def test_skips_unknown_class(self, tmp_path):
        xml = tmp_path / "test.xml"
        xml.write_text(
            "<annotation>"
            "<filename>test.jpg</filename>"
            "<size><width>640</width><height>480</height><depth>3</depth></size>"
            "<object><name>unknown_defect</name>"
            "<bndbox><xmin>0</xmin><ymin>0</ymin><xmax>100</xmax><ymax>100</ymax></bndbox>"
            "</object></annotation>"
        )
        _, anns = voc_xml_to_coco_annotations(xml, image_id=1, ann_id_start=0)
        assert len(anns) == 0

    def test_multiple_objects_sequential_ids(self, tmp_path):
        xml = tmp_path / "multi.xml"
        xml.write_text(
            "<annotation>"
            "<filename>multi.jpg</filename>"
            "<size><width>640</width><height>480</height><depth>3</depth></size>"
            "<object><name>hotspot</name>"
            "<bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>"
            "</object>"
            "<object><name>Soiling</name>"
            "<bndbox><xmin>100</xmin><ymin>100</ymin><xmax>200</xmax><ymax>200</ymax></bndbox>"
            "</object></annotation>"
        )
        _, anns = voc_xml_to_coco_annotations(xml, image_id=5, ann_id_start=10)
        assert len(anns) == 2
        assert anns[0]["id"] == 10
        assert anns[1]["id"] == 11
        assert anns[1]["category_id"] == 2  # soiling

    def test_float_bbox_coords_truncated_to_int(self, tmp_path):
        xml = tmp_path / "float.xml"
        xml.write_text(
            "<annotation>"
            "<filename>float.jpg</filename>"
            "<size><width>640</width><height>480</height><depth>3</depth></size>"
            "<object><name>hotspot</name>"
            "<bndbox><xmin>10.7</xmin><ymin>20.3</ymin><xmax>50.9</xmax><ymax>60.1</ymax></bndbox>"
            "</object></annotation>"
        )
        _, anns = voc_xml_to_coco_annotations(xml, image_id=1, ann_id_start=0)
        assert anns[0]["bbox"] == [10, 20, 40, 40]


class TestFindDuplicates:
    def test_identical_images_all_but_first_removed(self, tmp_path):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        paths = []
        for i in range(3):
            p = tmp_path / f"img{i}.jpg"
            PILImage.fromarray(arr).save(p)
            paths.append(p)
        dupes = find_duplicates(paths, threshold=8)
        assert len(dupes) == 2  # keep index 0, remove 1 and 2

    def test_different_images_not_flagged(self, tmp_path):
        paths = []
        for i in range(3):
            arr = np.zeros((64, 64, 3), dtype=np.uint8)
            arr[:, :, 0] = i * 80  # clearly different
            p = tmp_path / f"img{i}.jpg"
            PILImage.fromarray(arr).save(p)
            paths.append(p)
        dupes = find_duplicates(paths, threshold=8)
        assert len(dupes) == 0

    def test_returns_set_of_indices(self, tmp_path):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        p = tmp_path / "single.jpg"
        PILImage.fromarray(arr).save(p)
        dupes = find_duplicates([p], threshold=8)
        assert isinstance(dupes, set)
        assert len(dupes) == 0  # single image → no duplicates


class TestStratifiedSplit:
    def test_split_ratios_approximately_correct(self):
        ids = list(range(100))
        labels = [i % 3 for i in range(100)]
        groups = [str(i % 10) for i in range(100)]
        train, val, test = stratified_split(ids, labels, groups, ratios=(0.8, 0.1, 0.1))
        assert abs(len(train) - 80) <= 5
        assert abs(len(val) - 10) <= 5
        assert abs(len(test) - 10) <= 5

    def test_splits_are_disjoint(self):
        ids = list(range(100))
        labels = [i % 3 for i in range(100)]
        groups = [str(i % 10) for i in range(100)]
        train, val, test = stratified_split(ids, labels, groups)
        assert set(train).isdisjoint(set(val))
        assert set(train).isdisjoint(set(test))
        assert set(val).isdisjoint(set(test))

    def test_all_ids_appear_exactly_once(self):
        ids = list(range(100))
        labels = [i % 3 for i in range(100)]
        groups = [str(i % 10) for i in range(100)]
        train, val, test = stratified_split(ids, labels, groups)
        assert sorted(train + val + test) == ids

    def test_seeded_split_is_reproducible(self):
        ids = list(range(50))
        labels = [i % 2 for i in range(50)]
        groups = [str(i % 5) for i in range(50)]
        r1 = stratified_split(ids, labels, groups, seed=42)
        r2 = stratified_split(ids, labels, groups, seed=42)
        assert r1 == r2
