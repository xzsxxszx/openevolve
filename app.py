import os
import uuid
import threading
import time
import json
import glob
from flask import Flask, request, jsonify, render_template
from openevolve import run_evolution

# --- Application State ---
# Note: Using global variables for state is simple for this prototype,
# but a production app should use a more robust solution like Redis or a database
# to handle multiple server processes and task persistence.
tasks = {} # Stores the state and results of evolution tasks.

# --- Directories for temporary files ---
BASE_DIR = "/tmp/openevolve_webui"
EVALUATOR_DIR = os.path.join(BASE_DIR, "evaluators")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PROMPT_DIR = os.path.join(BASE_DIR, "prompts")
os.makedirs(EVALUATOR_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROMPT_DIR, exist_ok=True)

app = Flask(__name__)

def monitor_task(task_id, task_output_path):
    """
    Monitors the output directory of an evolution task for progress.
    This function runs in a dedicated thread for each active evolution.
    It periodically checks for the `database.jsonl` file, which openevolve uses
    to log each completed program evaluation. It parses this file to provide
    real-time updates on the number of generations and the best score found so far.
    """
    print(f"[{task_id}] Monitor thread started for path: {task_output_path}")
    while not tasks.get(task_id, {}).get('stop_monitor', False):
        try:
            db_path = os.path.join(task_output_path, "database.jsonl")
            if not os.path.exists(db_path):
                time.sleep(2)
                continue

            best_score = -float('inf')
            generation = 0
            with open(db_path, 'r') as f:
                lines = f.readlines()
                generation = len(lines)
                for line in lines:
                    entry = json.loads(line)
                    if entry['score'] > best_score:
                        best_score = entry['score']

            tasks[task_id]['progress'] = {
                "generation": generation,
                "best_score_so_far": best_score
            }
        except Exception as e:
            print(f"[{task_id}] Monitor error: {e}")

        time.sleep(2)
    print(f"[{task_id}] Monitor thread stopped.")

def evolution_worker(task_id, api_key, initial_program, evaluator_code, config):
    """
    The main worker function for handling an evolution task.
    This runs in a background thread to avoid blocking the web server.
    """
    tasks[task_id]['status'] = 'running'
    task_output_path = os.path.join(OUTPUT_DIR, task_id)
    evaluator_path = os.path.join(EVALUATOR_DIR, f"evaluator_{task_id}.py")
    task_prompt_dir = os.path.join(PROMPT_DIR, task_id)

    monitor = threading.Thread(target=monitor_task, args=(task_id, task_output_path))
    monitor.daemon = True
    monitor.start()

    try:
        os.environ['OPENAI_API_KEY'] = api_key

        with open(evaluator_path, 'w', encoding='utf-8') as f:
            f.write(evaluator_code)

        # --- Handle custom prompt based on #start evolve markers ---
        if "#start evolve" in initial_program and "#end evolve" in initial_program:
            print(f"[{task_id}] Markers found, creating custom prompt.")
            lines = initial_program.split('\\n')
            start_idx = next((i for i, line in enumerate(lines) if "#start evolve" in line), -1)
            end_idx = next((i for i, line in enumerate(lines) if "#end evolve" in line), -1)

            if 0 <= start_idx < end_idx:
                code_to_evolve = '\\n'.join(lines[start_idx+1:end_idx])
                context_before = '\\n'.join(lines[:start_idx])
                context_after = '\\n'.join(lines[end_idx+1:])

                custom_prompt_text = f"""You are an expert Python programmer. Your task is to evolve the following Python code to improve its performance or quality, based on the provided evaluation function.

You MUST only modify the code inside the "### CODE TO EVOLVE ###" block.
The surrounding code in "### CONTEXT ###" must be kept exactly as it is.

### CONTEXT ###
{context_before}
# ... Evolved code will be placed here ...
{context_after}
### END CONTEXT ###

### CODE TO EVOLVE ###
{code_to_evolve}
### END CODE TO EVOLVE ###
"""
                os.makedirs(task_prompt_dir, exist_ok=True)
                with open(os.path.join(task_prompt_dir, "mutator_prompt.txt"), "w") as f:
                    f.write(custom_prompt_text)

                # Update config to use this custom prompt
                if 'prompt' not in config:
                    config['prompt'] = {}
                config['prompt']['template_dir'] = task_prompt_dir

        print(f"[{task_id}] Starting evolution with config: {config}")
        result = run_evolution(
            initial_program=initial_program,
            evaluator=evaluator_path,
            output_path=task_output_path,
            **config
        )

        tasks[task_id].update({
            'status': 'completed',
            'result': { 'best_score': result.best_score, 'best_code': result.best_code }
        })
        print(f"[{task_id}] Evolution completed.")

    except Exception as e:
        print(f"[{task_id}] Error during evolution: {e}")
        tasks[task_id].update({ 'status': 'failed', 'error': str(e) })
    finally:
        tasks[task_id]['stop_monitor'] = True
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        if os.path.exists(evaluator_path):
            os.remove(evaluator_path)


@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/api/evolve', methods=['POST'])
def start_evolution():
    """API endpoint to start a new evolution task."""
    data = request.get_json()
    if not data or 'initial_program' not in data or 'evaluator_code' not in data or 'api_key' not in data:
        return jsonify({"error": "请求体中缺少 'api_key', 'initial_program', 或 'evaluator_code' 字段"}), 400

    task_id = str(uuid.uuid4())
    tasks[task_id] = {'status': 'pending', 'task_id': task_id, 'stop_monitor': False}

    thread = threading.Thread(
        target=evolution_worker,
        args=(task_id, data['api_key'], data['initial_program'], data['evaluator_code'], data.get('config', {}))
    )
    thread.daemon = True
    thread.start()

    return jsonify({"message": "演化任务已开始", "task_id": task_id}), 202

@app.route('/api/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """API endpoint to check the status of an evolution task."""
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "未找到该任务"}), 404

    return jsonify(task)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
