import unittest

from gene_benchmark.task_retrieval import sanitize_folder_name


class TestTaskRetrieval(unittest.TestCase):
    def test_sanitization(self):
        to_be_sanitized_path = "|hello world"
        with self.assertWarns(Warning):
            sanitized_str = sanitize_folder_name(to_be_sanitized_path)
        assert sanitized_str == "hello world"
