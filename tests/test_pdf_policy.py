import hashlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from adams_rag.pdf_policy import file_sha256


class PDFPolicyTests(unittest.TestCase):
    def test_hash_reads_bounded_blocks_and_enforces_byte_limit(self):
        data=b'x' * (2 * 1024 * 1024 + 10)
        reads=[]
        class Source(io.BytesIO):
            def read(self,size=-1):
                reads.append(size)
                return super().read(size)
        with patch.object(Path,'open',return_value=Source(data)):
            self.assertEqual(file_sha256(Path('unused'),len(data)),hashlib.sha256(data).hexdigest())
        self.assertGreater(len(reads),2)
        self.assertTrue(all(0 < n <= 1024 * 1024 for n in reads))
        with patch.object(Path,'open',return_value=Source(data)):
            with self.assertRaisesRegex(ValueError,'file allowance'):
                file_sha256(Path('unused'),len(data)-1)


if __name__=='__main__':
    unittest.main()
