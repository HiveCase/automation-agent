from flask import Flask, request, jsonify
import agent
import os

app = Flask(__name__)

@app.route('/run', methods=['POST'])
def run_task():
    task = request.args.get('task')
    if not task:
        return jsonify({'error': 'Task description is required'}), 400

    try:
        result = agent.run(task)
        return jsonify({'result': result}), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400  # Task-related error
    except Exception as e:
        print(e)
        return jsonify({'error': 'Agent error: ' + str(e)}, 500) # Agent error

@app.route('/read', methods=['GET'])
def read_file():
    path = request.args.get('path')
    if not path:
        return jsonify({'error': 'File path is required'}), 400

    try:
        with open(path, 'r') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/plain'}
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
