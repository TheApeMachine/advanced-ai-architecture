import ast
import subprocess
import tempfile
import os

class CodeVerifier:
    def __init__(self):
        pass

    def is_syntax_valid(self, code_str):
        if not isinstance(code_str, str):
            return False, f"Expected string, got {type(code_str)}"
        try:
            ast.parse(code_str)
            return True, ""
        except SyntaxError as e:
            return False, str(e)

    def run_tests(self, code_str, test_code_str):
        if not isinstance(code_str, str) or not isinstance(test_code_str, str):
            return False, "Code and test code must be strings"

        with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as tmp_code_file:
            tmp_code_file.write(code_str.encode('utf-8'))
            code_filename = tmp_code_file.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as tmp_test_file:
            tmp_test_file.write(test_code_str.encode('utf-8'))
            test_filename = tmp_test_file.name

        try:
            result = subprocess.run(['python', test_filename],
                                    capture_output=True, text=True, timeout=5)
            success = result.returncode == 0
            output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            success = False
            output = "Test execution timed out."
        finally:
            os.unlink(code_filename)
            os.unlink(test_filename)

        return success, output
