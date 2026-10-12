import unittest
from app import app


class FlaskHealthTestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        """Ensure /api/health returns 200 OK and valid JSON status."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("service", data)

    def test_index_route(self):
        """Ensure index page loads successfully."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Agentic AI for Automated Literature Review", response.data)


if __name__ == "__main__":
    unittest.main()
