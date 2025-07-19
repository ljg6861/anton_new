import unittest

class TestCounting(unittest.TestCase):

    def test_count_to_100(self):
        # This test will verify that the counting function (which we previously demonstrated)
        # produces the correct output.
        # For now, this is a placeholder, as we don't have a reusable function.
        # We'll adapt this to execute the counting code and check the output.
        
        # Execute the counting code (previously demonstrated)
        import subprocess
        result = subprocess.run(['python', 'count_to_100.py'], capture_output=True, text=True)

        expected_output = '\n'.join([str(i) for i in range(1, 101)]) + '\n'

        self.assertEqual(result.stdout.strip(), expected_output.strip())

if __name__ == '__main__':
    unittest.main()