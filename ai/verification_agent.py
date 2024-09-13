from code_verifier import CodeVerifier

class VerificationAgent(AgentBase):
    def __init__(self, name):
        super().__init__(name)
        self.code_verifier = CodeVerifier()

    def process(self, code_dict):
        results = {}
        for key, code in code_dict.items():
            is_valid, error = self.code_verifier.is_syntax_valid(code)
            results[key] = {'is_valid': is_valid, 'error': error, 'code': code}
        return results
