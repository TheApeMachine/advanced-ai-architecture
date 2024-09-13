import ast
import subprocess
import tempfile
import os

class CodeVerifier:
    def __init__(self):
        pass

    def is_syntax_valid(self, code_str):
        try:
            ast.parse(code_str)
            return True, ""
        except SyntaxError as e:
            return False, str(e)

    def run_tests(self, code_str, test_code_str):
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

# Example usage
if __name__ == "__main__":
    verifier = CodeVerifier()
    code = '''
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)
'''
    test_code = '''
import sys
sys.path.insert(0, '.')
from temp_code import factorial

assert factorial(0) == 1
assert factorial(5) == 120
print("All tests passed.")
'''

    is_valid, error = verifier.is_syntax_valid(code)
    if is_valid:
        success, output = verifier.run_tests(code, test_code)
        if success:
            print("Code passed all tests.")
        else:
            print("Code failed tests:")
            print(output)
    else:
        print("Syntax Error:")
        print(error)
