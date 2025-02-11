import os
import subprocess
import json
import sqlite3
import base64
from llm_utils import call_llm  # To interact with the LLM

DATA_DIR = "/data"

def run(task_description):
    """
    Executes a task based on the given description.
    """
    print(f"Executing task: {task_description}")

    # 1. LLM: Parse the task description and determine the steps
    steps = parse_task(task_description)  # Returns a list of dictionaries

    # 2. Execute the steps
    for step in steps:
        action = step.get("action")

        if action == "execute_python_script":
            script_path = step.get("script_path")
            arguments = step.get("arguments", [])
            execute_python_script(script_path, arguments)

        elif action == "format_markdown":
            file_path = step.get("file_path")
            format_markdown(file_path)

        elif action == "count_wednesdays":
            file_path = step.get("file_path")
            output_path = step.get("output_path")
            count_wednesdays(file_path, output_path)

        elif action == "sort_contacts":
            file_path = step.get("file_path")
            output_path = step.get("output_path")
            sort_contacts(file_path, output_path)

        elif action == "find_recent_logs":
            log_dir = step.get("log_dir")
            output_path = step.get("output_path")
            find_recent_logs(log_dir, output_path)

        elif action == "extract_markdown_headers":
            docs_dir = step.get("docs_dir")
            index_path = step.get("index_path")
            extract_markdown_headers(docs_dir, index_path)

        elif action == "extract_email_sender":
            email_path = step.get("email_path")
            output_path = step.get("output_path")
            extract_email_sender(email_path, output_path)

        elif action == "extract_credit_card":
            image_path = step.get("image_path")
            output_path = step.get("output_path")
            extract_credit_card(image_path, output_path)

        elif action == "find_similar_comments":
            comments_path = step.get("comments_path")
            output_path = step.get("output_path")
            find_similar_comments(comments_path, output_path)

        elif action == "query_ticket_sales":
            db_path = step.get("db_path")
            output_path = step.get("output_path")
            query_ticket_sales(db_path, output_path)

        else:
            raise ValueError(f"Unknown action: {action}")

    return "Task completed"

def parse_task(task_description):
    """
    Parses the task description using an LLM to determine the necessary steps.
    """
    prompt = f"""
    You are a task parsing agent. Your job is to break down the task into a series of executable steps.
    Each step must be one of the following actions:
    - execute_python_script: Executes a Python script.  Requires 'script_path' and optional 'arguments'.
    - format_markdown: Formats a markdown file using prettier. Requires 'file_path'.
    - count_wednesdays: Counts Wednesdays in a file. Requires 'file_path' and 'output_path'.
    - sort_contacts: Sorts contacts in a JSON file. Requires 'file_path' and 'output_path'.
    - find_recent_logs: Finds the first line of the most recent log files. Requires 'log_dir' and 'output_path'.
    - extract_markdown_headers: Extracts H1 headers from markdown files. Requires 'docs_dir' and 'index_path'.
    - extract_email_sender: Extracts the sender's email from a text file. Requires 'email_path' and 'output_path'.
    - extract_credit_card: Extracts a credit card number from an image. Requires 'image_path' and 'output_path'.
    - find_similar_comments: Finds the most similar pair of comments. Requires 'comments_path' and 'output_path'.
    - query_ticket_sales: Queries a SQLite database for ticket sales. Requires 'db_path' and 'output_path'.

    Ensure that all file paths are within the /data directory.

    Example Output:
    [
      {{
        "action": "execute_python_script",
        "script_path": "/data/datagen.py",
        "arguments": ["test@example.com"]
      }}
    ]

    Now, break down this task: {task_description}
    """

    llm_response = call_llm(prompt)
    try:
        steps = json.loads(llm_response)
        return steps
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse LLM response: {llm_response}")

def execute_python_script(script_path, arguments):
    """
    Executes a Python script with the given arguments.
    """
    if not script_path.startswith(DATA_DIR):
        raise ValueError("Accessing files outside of /data is not allowed.")

    try:
        command = ["python", script_path] + arguments
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Error executing script: {e}")

def format_markdown(file_path):
    """
    Formats a Markdown file using prettier.
    """
    if not file_path.startswith(DATA_DIR):
        raise ValueError("Accessing files outside of /data is not allowed.")

    try:
        command = ["prettier", "--write", file_path, "--loglevel", "warn"]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Error formatting Markdown: {e}")

def count_wednesdays(file_path, output_path):
    """
    Counts the number of Wednesdays in a list of dates.
    """
    if not file_path.startswith(DATA_DIR) or not output_path.startswith(DATA_DIR):
        raise ValueError("Accessing files outside of /data is not allowed.")

    try:
        with open(file_path, 'r') as f:
            dates = f.readlines()

        wednesday_count = 0
        for date_str in dates:
            date_str = date_str.strip()
            # Basic validation to prevent exceptions
            if len(date_str) > 5:
                import datetime
                try:
                    date_object = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                    if date_object.weekday() == 2:  # Wednesday is 2
                        wednesday_count += 1
                except ValueError:
                    print(f"Warning: Could not parse date: {date_str}")

        with open(output_path, 'w') as f:
            f.write(str(wednesday_count))

    except Exception as e:
        raise ValueError(f"Error counting Wednesdays: {e}")

def sort_contacts(file_path, output_path):
    """
    Sorts an array of contacts in a JSON file by last_name, then first_name.
    """
    if not file_path.startswith(DATA_DIR) or not output_path.startswith(DATA_DIR):
        raise ValueError("Accessing files outside of /data is not allowed.")

    try:
        with open(file_path, 'r') as f:
            contacts = json.load(f)

        sorted_contacts = sorted(contacts, key=lambda x: (x['last_name'], x['first_name']))

        with open(output_path, 'w') as f:
            json.dump(sorted_contacts, f, indent=2)

    except Exception as e:
        raise ValueError(f"Error sorting contacts: {e}")

def find_recent_logs(log_dir, output_path):
    """
    Writes the first line of the 10 most recent .log files to the output file, most recent first.
    """
    if not log_dir.startswith(DATA_DIR) or not output_path.startswith(DATA_DIR):
        raise ValueError("Accessing files outside of /data is not allowed.")

    try:
        import glob
        import os

        log_files = glob.glob(os.path.join(log_dir, "*.log"))
        log_files.sort(key=os.path.getmtime, reverse=True)  # Sort by modification time

        with open(output_path, 'w') as outfile:
            for log_file in log_files[:10]:  # Process the 10 most recent
                try:
                    with open(log_file, 'r') as infile:
                        first_line = infile.readline().strip()
                        outfile.write(first_line + "\n")
                except Exception as e:
                    print(f"Warning: Could not read first line from {log_file}: {e}")
    except Exception as e:
        raise ValueError(f"Error finding recent logs: {e}")

def extract_markdown_headers(docs_dir, index_path):
    """
    Extracts the first H1 header from each Markdown file in the specified directory and creates an index file.
    """
    if not docs_dir.startswith(DATA_DIR) or not index_path.startswith(DATA_DIR):
        raise ValueError("Accessing files outside of /data is not allowed.")

    try:
        import glob
        import os
        import re

        index = {}
        markdown_files = glob.glob(os.path.join(docs_dir, "**/*.md"), recursive=True)

        for file_path in markdown_files:
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
                    # Find the first H1 header
                    match = re.search(r'^#\s+(.*)$', content, re.MULTILINE)
                    if match:
                        title = match.group(1).strip()
                        # Remove the /data/docs/ prefix
                        relative_path = os.path.relpath(file_path, docs_dir)
                        index[relative_path] = title
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")

        with open(index_path, 'w') as index_file:
            json.dump(index, index_file, indent=2)

    except Exception as e:
        raise ValueError(f"Error extracting Markdown headers: {e}")

def extract_email_sender(email_path, output_path):
    """
    Extracts the sender's email address from an email message using an LLM.
    """
    if not email_path.startswith(DATA_DIR) or not output_path.startswith(DATA_DIR):
        raise ValueError("Accessing files outside of /data is not allowed.")

    try:
        with open(email_path, 'r') as f:
            email_content = f.read()

        prompt = f"Extract the sender's email address from this email:\n{email_content}"
        sender_email = call_llm(prompt)

        with open(output_path, 'w') as f:
            f.write(sender_email.strip())

    except Exception as e:
        raise ValueError(f"Error extracting email sender: {e}")

def extract_credit_card(image_path, output_path):
    """
    Extracts the credit card number from an image using an LLM.
    """
    if not image_path.startswith(DATA_DIR) or not output_path.startswith(DATA_DIR):
        raise ValueError("Accessing files outside of /data is not allowed.")

    try:
        # Convert image to base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        prompt = f"Extract the credit card number from this image (base64 encoded):\n{encoded_string}"
        card_number = call_llm(prompt)

        # Remove spaces from the card number
        card_number = card_number.replace(" ", "")

        with open(output_path, 'w') as f:
            f.write(card_number.strip())

    except Exception as e:
        raise ValueError(f"Error extracting credit card: {e}")

def find_similar_comments(comments_path, output_path):
    """
    Finds the most similar pair of comments using embeddings and writes them to the output file.
    """
    if not comments_path.startswith(DATA_DIR) or not output_path.startswith(DATA_DIR):
        raise ValueError("Accessing files outside of /data is not allowed.")

    try:
        with open(comments_path, 'r') as f:
            comments = [line.strip() for line in f.readlines()]

        if len(comments) < 2:
            raise ValueError("Need at least two comments to find a similar pair.")

        # Use LLM to get embeddings for each comment
        embeddings = []
        for comment in comments:
            prompt = f"Generate an embedding vector for the following sentence: {comment}"
            embedding_str = call_llm(prompt)
            try:
                embedding = json.loads(embedding_str)
                embeddings.append(embedding)
            except json.JSONDecodeError:
                 raise ValueError(f"Could not parse LLM embedding response: {embedding_str}")

        # Calculate cosine similarity between all pairs
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        max_similarity = -1
        most_similar_pair = None

        for i in range(len(comments)):
            for j in range(i + 1, len(comments)):
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_pair = (comments[i], comments[j])

        if most_similar_pair:
            with open(output_path, 'w') as f:
                f.write(most_similar_pair[0] + '\n')
                f.write(most_similar_pair[1] + '\n')
        else:
            raise ValueError("Could not find any similar comments.")

    except Exception as e:
        raise ValueError(f"Error finding similar comments: {e}")

def query_ticket_sales(db_path, output_path):
    """
    Queries a SQLite database for the total sales of "Gold" tickets and writes the result to the output file.
    """
    if not db_path.startswith(DATA_DIR) or not output_path.startswith(DATA_DIR):
        raise ValueError("Accessing files outside of /data is not allowed.")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
        result = cursor.fetchone()[0]

        conn.close()

        if result is None:
            result = 0.0  # Handle case where there are no Gold tickets

        with open(output_path, 'w') as f:
            f.write(str(result))

    except Exception as e:
        raise ValueError(f"Error querying ticket sales: {e}")
