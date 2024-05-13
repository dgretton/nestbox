import json
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import redis
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

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

# Start the Flask-SocketIO server
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
