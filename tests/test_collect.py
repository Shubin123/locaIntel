import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from pipeline.collect import atomic_json, catalog

class PipelineTests(unittest.TestCase):
    def test_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.json'
            atomic_json(p, {'a': 1})
            atomic_json(p, {'a': 2})
            self.assertEqual(json.loads(p.read_text()), {'a': 2})
            self.assertFalse(p.with_suffix('.tmp').exists())

    def test_catalog_uses_fresh_cache(self):
        with tempfile.TemporaryDirectory() as d, patch('pipeline.collect.CACHE', Path(d)):
            (Path(d)/'catalog.csv').write_text('id,camName,region_name,highway_number,caption,links_imageURL,links_camDetailsPage,latitude,longitude\n1,Test,North,1,Road,https://example.com/1.jpg,https://example.com,49,-123\n')
            class Offline:
                def get(self, url):
                    raise AssertionError('Fresh cache should not make network requests')
            self.assertEqual(catalog(Offline())[0]['lat'],49)
