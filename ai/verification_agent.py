from .code_verifier import CodeVerifier

class VerificationAgent:
    def __init__(self, name):
        self.name = name
        self.code_verifier = CodeVerifier()

    def process(self, result):
        verification_results = {}
        for key, code in result.items():
            if isinstance(code, dict) and 'code_ai' in code and 'code_coder' in code:
                verification_results[key] = {
                    'code_ai': self.verify_code(code['code_ai']),
                    'code_coder': self.verify_code(code['code_coder'])
                }
            else:
                verification_results[key] = self.verify_code(code)
        return verification_results

    def verify_code(self, code):
        if not isinstance(code, str):
            return {'is_valid': False, 'error': 'Code must be a string'}
        is_valid, error = self.code_verifier.is_syntax_valid(code)
        return {'is_valid': is_valid, 'error': error, 'code': code}
