import docker
from llama_cpp import Llama
import tempfile
import os
import glob
import json
import subprocess
from huggingface_hub import hf_hub_download
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from contextlib import contextmanager
import logging
import re

class AdvancedAISelfImprover:
    def __init__(self, model_path):
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

        self.client = docker.from_env()
        self.llm = Llama(model_path=model_path, n_ctx=4096)  # Increased context window
        self.vector_store = QdrantClient("localhost", port=6333)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection_name = "code_vectors"

        # Create vector store collection if it doesn't exist
        self.vector_store.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )

    def build_container(self, code_directory):
        logging.info("Building Docker container for the codebase.")
        dockerfile = '''
        FROM python:3.11
        WORKDIR /app
        COPY . /app
        RUN pip install -r requirements.txt || true
        RUN pip install qdrant-client sentence-transformers
        CMD ["python", "test.py"]
        '''
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(dockerfile)
            dockerfile_path = f.name

        try:
            image, _ = self.client.images.build(path=code_directory, dockerfile=dockerfile_path, tag='ai-codebase')
        except docker.errors.BuildError as e:
            logging.error(f"Error building Docker image: {e}")
            raise
        finally:
            os.unlink(dockerfile_path)
        return image

    @contextmanager
    def docker_container(self, image):
        logging.info("Starting Docker container.")
        container = self.client.containers.run(image.id, detach=True, network_mode='host')
        try:
            yield container
        finally:
            logging.info("Stopping and removing Docker container.")
            container.stop()

    def push_code_to_vector_store(self, code_directory):
        logging.info("Pushing code to vector store.")
        file_paths = glob.glob(os.path.join(code_directory, '**/*.py'), recursive=True)
        contents = []
        ids = []
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                contents.append(file.read())
                ids.append(file_path)
        vectors = self.encoder.encode(contents, batch_size=16, show_progress_bar=True)
        points = [
            models.PointStruct(
                id=ids[i],
                vector=vectors[i].tolist(),
                payload={"content": contents[i]}
            ) for i in range(len(ids))
        ]
        self.vector_store.upsert(collection_name=self.collection_name, points=points)

    def analyze_code(self, code_directory):
        logging.info("Analyzing code.")
        notes = []
        for file_path in glob.glob(os.path.join(code_directory, '**/*.py'), recursive=True):
            with open(file_path, 'r') as file:
                content = file.read()
                chunks = self.chunk_content(content, 3000)
                for chunk in chunks:
                    prompt = (
                        f"Please analyze the following code and make concise notes about its structure, purpose, "
                        f"and any potential improvements. Output the notes in JSON format with keys 'structure', "
                        f"'purpose', and 'improvements'.\n\nCode:\n{chunk}\n\nJSON Notes:"
                    )
                    response = self.llm(prompt, max_tokens=500, temperature=0.7)
                    try:
                        note = json.loads(response['choices'][0]['text'].strip())
                        notes.append(note)
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse JSON notes: {e}")
                        continue
        return notes

    def chunk_content(self, content, chunk_size):
        return [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

    def formulate_task(self, notes):
        logging.info("Formulating task based on code analysis.")
        notes_text = "\n".join([json.dumps(note) for note in notes])
        prompt = (
            f"You are a senior software engineer reviewing the following code analysis notes:\n\n{notes_text}\n\n"
            f"Based on these notes, identify the most impactful improvement that can be made to the codebase. "
            f"Formulate a specific and actionable task description. The task should focus on enhancing performance, "
            f"readability, or maintainability.\n\nBefore finalizing the task, briefly explain your reasoning.\n\n"
            f"Final Task (provide only the task description):"
        )
        response = self.llm(prompt, max_tokens=300, temperature=0.7)
        # Since we asked for only the task description in the final output, we can take it directly
        task = response['choices'][0]['text'].strip()
        return task

    def create_improvement_plan(self, task):
        logging.info(f"Creating improvement plan for task: {task}")
        relevant_code = self.query_vector_store(task)
        prompt = (
            f"You are tasked with the following objective:\n\n{task}\n\n"
            f"The following code snippets are relevant to the task:\n{relevant_code}\n\n"
            f"Create a detailed, step-by-step plan to implement this task. "
            f"Consider potential challenges and how to address them. "
            f"Ensure that the plan is clear and actionable.\n\n"
            f"Plan:"
        )
        response = self.llm(prompt, max_tokens=1000, temperature=0.7)
        plan = response['choices'][0]['text'].strip()
        return plan

    def query_vector_store(self, query):
        logging.info("Querying vector store for relevant code snippets.")
        vector = self.encoder.encode(query).tolist()
        results = self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=5
        )
        return "\n\n".join([hit.payload['content'] for hit in results])

    def reflect_on_plan(self, plan):
        logging.info("Reflecting on the improvement plan.")
        prompt = (
            f"You are reviewing the following improvement plan:\n\n{plan}\n\n"
            f"Identify any potential issues or oversights in the plan. "
            f"Suggest any improvements to the plan to make it more effective.\n\n"
            f"Reflection:"
        )
        response = self.llm(prompt, max_tokens=500, temperature=0.7)
        reflection = response['choices'][0]['text'].strip()

        # Incorporate reflection into the plan if needed
        prompt_revision = (
            f"Based on the following reflection, revise the improvement plan to address the identified issues:\n\n"
            f"Reflection:\n{reflection}\n\nOriginal Plan:\n{plan}\n\n"
            f"Revised Plan:"
        )
        response_revision = self.llm(prompt_revision, max_tokens=1000, temperature=0.7)
        revised_plan = response_revision['choices'][0]['text'].strip()
        return revised_plan

    def implement_changes(self, plan, container):
        logging.info("Implementing changes based on the plan.")
        prompt = (
            f"Based on the following plan, generate the exact code changes to be made. "
            f"Include the full content of any files that need to be modified or created. "
            f"Format your response as a JSON array of objects with 'file_path' and 'file_content' keys.\n\n"
            f"Plan:\n{plan}\n\nJSON Code Changes:"
        )
        response = self.llm(prompt, max_tokens=2000, temperature=0.7)
        try:
            changes = json.loads(response['choices'][0]['text'].strip())
            self.apply_changes_in_container(changes, container)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse code changes: {e}")
            return False
        return True

    def apply_changes_in_container(self, changes, container):
        logging.info("Applying code changes in the container.")
        for change in changes:
            file_path = os.path.basename(change['file_path'])  # Prevent directory traversal
            if not re.match(r'^[\w\-. ]+\.py$', file_path):
                logging.warning(f"Invalid file name: {file_path}")
                continue
            file_content = change['file_content']
            # Sanitize file content
            safe_file_content = file_content.replace('"', '\\"').replace("'", "\\'")
            safe_file_content = safe_file_content.replace('\n', '\\n')
            safe_file_content = safe_file_content.replace('$', '\\$')  # Escape special characters
            safe_file_content = safe_file_content.replace('`', '\\`')
            # Write content to file in the container
            exec_command = f"bash -c 'echo \"{safe_file_content}\" > /app/{file_path}'"
            try:
                exit_code, output = container.exec_run(exec_command, stream=True)
                for line in output:
                    logging.info(line.decode('utf-8').strip())
                if exit_code != 0:
                    logging.error(f"Error applying changes to {file_path}")
            except Exception as e:
                logging.error(f"Exception occurred while applying changes: {e}")
            
    def run_code_in_container(self, container):
        logging.info("Running code in the container.")
        try:
            exit_code, output = container.exec_run("python test.py", stream=True)
            output_text = ""
            for line in output:
                output_text += line.decode('utf-8')
                if len(output_text) > 1000:
                    break  # Limit output to approximately 1000 characters
            return exit_code, output_text[:1000]  # Ensure we don't exceed 1000 characters
        except Exception as e:
            logging.error(f"Error running code in container: {e}")
            return 1, str(e)

    def resolve_errors(self, error_output, container):
        logging.info("Resolving errors encountered during code execution.")
        prompt = (
            f"The following error occurred when running the code:\n\n{error_output}\n\n"
            f"Analyze the error and suggest a fix. Provide the updated code changes in JSON format as before.\n\nFix:"
        )
        response = self.llm(prompt, max_tokens=1500, temperature=0.7)
        try:
            fix = json.loads(response['choices'][0]['text'].strip())
            self.apply_changes_in_container(fix, container)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse fix: {e}")

    def autonomous_improvement_loop(self, code_directory, max_iterations=5):
        logging.info("Starting autonomous improvement loop.")
        image = self.build_container(code_directory)
        with self.docker_container(image) as container:
            self.push_code_to_vector_store(code_directory)

            for i in range(max_iterations):
                logging.info(f"Iteration {i+1}")

                logging.info("Analyzing code...")
                notes = self.analyze_code(code_directory)
                with open(f'notes_iteration_{i+1}.json', 'w') as f:
                    json.dump(notes, f)

                logging.info("Formulating task...")
                task = self.formulate_task(notes)
                logging.info(f"Task: {task}")

                logging.info("Creating improvement plan...")
                plan = self.create_improvement_plan(task)
                with open(f'plan_iteration_{i+1}.txt', 'w') as f:
                    f.write(plan)

                logging.info("Reflecting on plan...")
                revised_plan = self.reflect_on_plan(plan)
                with open(f'revised_plan_iteration_{i+1}.txt', 'w') as f:
                    f.write(revised_plan)

                logging.info("Implementing changes...")
                success = self.implement_changes(revised_plan, container)
                if not success:
                    logging.error("Failed to implement changes. Skipping to next iteration.")
                    continue

                logging.info("Running code...")
                exit_code, output = self.run_code_in_container(container)
                logging.info(f"Exit code: {exit_code}")
                logging.info(f"Output: {output}")

                if exit_code != 0:
                    logging.info("Resolving errors...")
                    self.resolve_errors(output, container)
                else:
                    logging.info("Code ran successfully.")

                logging.info("-" * 50)

    # Usage
if __name__ == "__main__":
    model_path = hf_hub_download(
        repo_id="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"
    )

    system = AdvancedAISelfImprover(model_path)
    system.autonomous_improvement_loop('./')
