from selenium import webdriver
from .tool import Tool

class Browser(Tool):
    def __init__(self):
        """
        Initialize the browser tool. Inherits from Tool.
        """
        super().__init__(
            name="browser",
            description="Allows you to browse the web, using a fully functional browser.",
            parameters={
                "url": {
                    "type": "string",
                    "description": "The URL to browse to."
                }
            }
        )
    def use(self, query):
        """
        Perform a web search using the browser.
        :param query: The search query.
        :return: Page source of the search results.
        """
        self.driver = webdriver.Chrome()
        self.driver.get(f"https://www.google.com/search?q={query}")
        return self.driver.page_source

    def search(self, query):
        """
        Alias for use method.
        :param query: The search query.
        :return: Page source of the search results.
        """
        return self.use(query)

    def close(self):
        """
        Close the browser cleanly.
        """
        self.driver.quit()