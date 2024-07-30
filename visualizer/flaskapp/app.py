import json
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import redis
import threading
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Project structure:
# /
# ├── ui-core/
# ├── visualizer/
# │   └── flaskapp/
# │       └── app.py (this file)
# 
# └── nestbox/

ui_core_dir = os.path.abspath(os.path.join(app.root_path, '..', '..', 'ui-core'))
libs_dir = os.path.abspath(os.path.join(app.root_path, '..', '..', 'libs'))

@app.route('/ui-core/<path:filename>')
def ui_core_files(filename):
    return send_from_directory(ui_core_dir, filename)

@app.route('/libs/<path:filename>')
def lib_files(filename):
    return send_from_directory(libs_dir, filename)

@app.route('/')
def index():
    return render_template('index.html')

# Redis Pub/Sub Listener
def redis_listener():
    pubsub = redis_client.pubsub()
    pubsub.subscribe('optimization_update')

    for message in pubsub.listen():
        if message['type'] == 'message':
            data = json.loads(message['data'])
            socketio.emit('optimization_update', data)

# Run Redis Listener in a separate thread
threading.Thread(target=redis_listener, daemon=True).start()

@socketio.on('pin_system')
def handle_pin_system(message):
    pin = message['pin']
    print(f"Pin coordinate system: {pin}")
    # Convert the pin to a JSON string and publish to a Redis channel
    redis_client.publish('pin_command', json.dumps({'pin': pin}))

# Start the Flask-SocketIO server
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
