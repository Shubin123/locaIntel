import json
import tempfile
import unittest
from pathlib import Path
from pipeline.publish_catalog import CATALOG, publish


class PublishCatalogTests(unittest.TestCase):
    def test_only_public_csv_is_downloaded_and_no_image_evidence_is_written(self):
        calls = []
        def fetch(url):
            calls.append(url)
            return ('id,camName,region_name,highway_number,caption,links_imageURL,links_camDetailsPage,latitude,longitude\n'
                    '238,Test,Metro,1,Road,https://www.drivebc.ca/images/238.jpg,https://www.drivebc.ca/cameras/238,49,-123\n')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = publish(root, fetch)
            self.assertEqual(calls, [CATALOG])
            self.assertEqual(result['storage_mode'], 'browser-only')
            self.assertEqual(result['cameras'][0]['id'], '238')
            self.assertEqual(set(p.name for p in root.iterdir()), {'latest.json','cameras.json','patterns.json'})
            self.assertEqual(json.loads((root/'patterns.json').read_text()), {'cameras':{},'hourly':[]})
            for field in ('hash','frame','detections','facing','features'):
                self.assertNotIn(field, result['cameras'][0])

    def test_empty_catalog_fails_instead_of_deploying_an_empty_site(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                publish(Path(directory), lambda _: 'id,camName\n')

    def test_ci_has_no_photo_collection_or_photo_cache_and_gates_deployment_on_browser_test(self):
        workflow = Path('.github/workflows/publish.yml').read_text()
        self.assertNotIn('pipeline.collect', workflow)
        self.assertNotIn('actions/cache', workflow)
        self.assertNotIn('path: web', workflow)
        self.assertIn('path: .cache/site', workflow)
        self.assertIn('needs: verify', workflow)
        self.assertIn('BROWSER_BASE_URL: https://shubin123.github.io/locaIntel/', workflow)
        self.assertEqual(workflow.count('run: npm run test:browser'), 2)
