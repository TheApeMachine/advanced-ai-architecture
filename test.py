import logging
from ai.task_manager import TaskManager
from world.domain import Domain
import asyncio

async def test_task_manager():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    domain = Domain()
    model = domain.equip("general")

    # Provide an initial code string for SelfModifyingAI
    initial_code_str = """
def example_function():
    print("This is an example function")
"""

    # Initialize the TaskManager with the initial code string
    task_manager = TaskManager(logger, model["model"], initial_code_str)

    # Test cases
    test_cases = [
        "Generate a Python function to calculate the factorial of a number.",
        "Create a Java class for a simple bank account with deposit and withdraw methods.",
        "Implement a binary search algorithm in C++.",
        "Write a JavaScript function to reverse a string.",
        "Develop a SQL query to find the top 5 customers by total purchase amount.",
    ]

    # Run test cases
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"Test Case {i}: {test_case}")
        task_manager.execute_task(test_case)
        logger.info("=" * 50)

    # Test a complex task that requires planning
    complex_task = "Create a web scraper that extracts product information from an e-commerce website and stores it in a database."
    logger.info(f"Complex Task: {complex_task}")
    await task_manager.execute_task(complex_task)

if __name__ == "__main__":
    asyncio.run(test_task_manager())
