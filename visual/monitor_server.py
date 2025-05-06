"""
Simple monitoring server for AlphaZero 2048 training visualization
"""

import os
import time
import json
import threading
import webbrowser
from typing import Dict, Any, List
from collections import deque

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

# Global state for monitoring
class TrainingMonitor:
    def __init__(self, history_limit=20):
        # Worker game states (board snapshots for visualization)
        self.worker_games = {}
        
        # Training statistics
        self.training_stats = {
            "current_epoch": 0,
            "total_epochs": 0,
            "samples_collected": 0,
            "games_played": 0,
            "latest_loss": 0,
            "policy_loss": 0,
            "value_loss": 0,
            "max_tile_achieved": 0,
            "max_score": 0,
        }
        
        # History of metrics for charts
        self.loss_history = deque(maxlen=history_limit)
        self.score_history = deque(maxlen=history_limit)
        self.tile_history = deque(maxlen=history_limit)
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def update_worker_game(self, worker_id: int, board: List[List[int]], score: int, turn: int):
        """Update the game state for a worker"""
        with self.lock:
            self.worker_games[worker_id] = {
                "board": board,
                "score": score,
                "turn": turn,
                "last_update": time.time()
            }
    
    def update_training_stats(self, stats: Dict[str, Any]):
        """Update training statistics"""
        with self.lock:
            self.training_stats.update(stats)
            
            # Add to history for charts
            if "latest_loss" in stats and stats["latest_loss"] is not None:
                self.loss_history.append((self.training_stats["current_epoch"], stats["latest_loss"]))
            
            if "max_score" in stats:
                self.score_history.append((self.training_stats["current_epoch"], stats["max_score"]))
            
            if "max_tile_achieved" in stats:
                self.tile_history.append((self.training_stats["current_epoch"], stats["max_tile_achieved"]))
    
    def get_status(self):
        """Get current status for API response"""
        with self.lock:
            # Clean up stale worker games (no update in 60 seconds)
            current_time = time.time()
            to_remove = []
            for worker_id, game in self.worker_games.items():
                if current_time - game["last_update"] > 60:
                    to_remove.append(worker_id)
            
            for worker_id in to_remove:
                del self.worker_games[worker_id]
            
            return {
                "worker_games": self.worker_games,
                "training_stats": self.training_stats,
                "loss_history": list(self.loss_history),
                "score_history": list(self.score_history),
                "tile_history": list(self.tile_history)
            }


# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'zero2048-monitor'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize global monitor
monitor = TrainingMonitor()

# Routes
@app.route('/')
def index():
    """Render the monitoring dashboard"""
    return render_template('monitor.html')

@app.route('/api/status')
def get_status():
    """API endpoint to get current status"""
    return jsonify(monitor.get_status())

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    socketio.emit('status', monitor.get_status())

# API endpoints for updates from training process
@app.route('/api/update_game', methods=['POST'])
def update_game():
    """Update a worker's game state"""
    from flask import request
    data = request.json
    worker_id = data.get('worker_id')
    board = data.get('board')
    score = data.get('score')
    turn = data.get('turn')
    
    monitor.update_worker_game(worker_id, board, score, turn)
    socketio.emit('game_update', {
        'worker_id': worker_id,
        'board': board,
        'score': score,
        'turn': turn
    })
    return jsonify({"status": "ok"})

@app.route('/api/update_stats', methods=['POST'])
def update_stats():
    """Update training statistics"""
    from flask import request
    data = request.json
    
    monitor.update_training_stats(data)
    socketio.emit('stats_update', monitor.training_stats)
    return jsonify({"status": "ok"})

# Create templates directory and HTML template if it doesn't exist
def ensure_template_exists():
    """Ensure templates directory and HTML file exist"""
    os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
    
    html_path = os.path.join(os.path.dirname(__file__), 'templates', 'monitor.html')
    if not os.path.exists(html_path):
        with open(html_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlphaZero 2048 Monitor</title>
    <script src="https://cdn.socket.io/4.4.1/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2, h3 {
            color: #333;
        }
        .stats-panel {
            background-color: #fff;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .games-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .game-card {
            background-color: #fff;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .board {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            grid-template-rows: repeat(4, 1fr);
            gap: 5px;
            margin-top: 10px;
        }
        .tile {
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            font-weight: bold;
        }
        .charts {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .chart-container {
            background-color: #fff;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #2a6fc9;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
        }
        /* Tile colors based on value */
        .tile-0 { background-color: #ccc0b3; }
        .tile-2 { background-color: #eee4da; }
        .tile-4 { background-color: #ede0c8; }
        .tile-8 { background-color: #f2b179; color: white; }
        .tile-16 { background-color: #f59563; color: white; }
        .tile-32 { background-color: #f67c5f; color: white; }
        .tile-64 { background-color: #f65e3b; color: white; }
        .tile-128 { background-color: #edcf72; color: white; }
        .tile-256 { background-color: #edcc61; color: white; }
        .tile-512 { background-color: #edc850; color: white; }
        .tile-1024 { background-color: #edc53f; color: white; font-size: 14px; }
        .tile-2048 { background-color: #edc22e; color: white; font-size: 14px; }
        .tile-4096 { background-color: #ed702e; color: white; font-size: 14px; }
        .tile-8192 { background-color: #ed702e; color: white; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AlphaZero 2048 Training Monitor</h1>
        
        <!-- Training Stats -->
        <div class="stats-panel">
            <h2>Training Progress</h2>
            <div class="stats-grid">
                <div>
                    <h3>Epoch</h3>
                    <div class="stat-value"><span id="current-epoch">0</span>/<span id="total-epochs">0</span></div>
                </div>
                <div>
                    <h3>Samples</h3>
                    <div class="stat-value" id="samples-collected">0</div>
                </div>
                <div>
                    <h3>Games</h3>
                    <div class="stat-value" id="games-played">0</div>
                </div>
                <div>
                    <h3>Max Tile</h3>
                    <div class="stat-value" id="max-tile">0</div>
                </div>
                <div>
                    <h3>Max Score</h3>
                    <div class="stat-value" id="max-score">0</div>
                </div>
                <div>
                    <h3>Latest Loss</h3>
                    <div class="stat-value" id="latest-loss">N/A</div>
                </div>
            </div>
        </div>

        <!-- Charts -->
        <div class="charts">
            <div class="chart-container">
                <canvas id="loss-chart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="score-chart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="tile-chart"></canvas>
            </div>
        </div>
        
        <!-- Live Games -->
        <h2>Live Games</h2>
        <div id="games-container" class="games-grid">
            <!-- Games will be added here dynamically -->
            <div class="game-card">
                <p>Waiting for games to start...</p>
            </div>
        </div>
    </div>

    <script>
        // Connect to Socket.IO server
        const socket = io();
        
        // Charts
        let lossChart, scoreChart, tileChart;
        
        // Initialize charts
        function initCharts() {
            // Loss chart
            const lossCtx = document.getElementById('loss-chart').getContext('2d');
            lossChart = new Chart(lossCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Training Loss',
                        data: [],
                        borderColor: '#2a6fc9',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Training Loss'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            });
            
            // Score chart
            const scoreCtx = document.getElementById('score-chart').getContext('2d');
            scoreChart = new Chart(scoreCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Max Score',
                        data: [],
                        borderColor: '#4caf50',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Max Score'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            
            // Tile chart
            const tileCtx = document.getElementById('tile-chart').getContext('2d');
            tileChart = new Chart(tileCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Max Tile',
                        data: [],
                        borderColor: '#f44336',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Max Tile'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }
        
        // Update training stats
        function updateStats(stats) {
            document.getElementById('current-epoch').textContent = stats.current_epoch;
            document.getElementById('total-epochs').textContent = stats.total_epochs;
            document.getElementById('samples-collected').textContent = stats.samples_collected.toLocaleString();
            document.getElementById('games-played').textContent = stats.games_played.toLocaleString();
            document.getElementById('max-tile').textContent = stats.max_tile_achieved.toLocaleString();
            document.getElementById('max-score').textContent = stats.max_score.toLocaleString();
            
            if (stats.latest_loss !== null && stats.latest_loss !== undefined) {
                document.getElementById('latest-loss').textContent = stats.latest_loss.toFixed(4);
            } else {
                document.getElementById('latest-loss').textContent = 'N/A';
            }
        }
        
        // Create or update game display
        function updateGame(workerId, board, score, turn) {
            let gameCard = document.getElementById(`game-${workerId}`);
            
            if (!gameCard) {
                // Create new game card
                gameCard = document.createElement('div');
                gameCard.id = `game-${workerId}`;
                gameCard.className = 'game-card';
                
                const header = document.createElement('h3');
                header.textContent = `Worker ${workerId}`;
                gameCard.appendChild(header);
                
                const scoreElem = document.createElement('p');
                scoreElem.innerHTML = `Score: <span id="score-${workerId}">0</span> | Turn: <span id="turn-${workerId}">0</span>`;
                gameCard.appendChild(scoreElem);
                
                const boardElem = document.createElement('div');
                boardElem.id = `board-${workerId}`;
                boardElem.className = 'board';
                gameCard.appendChild(boardElem);
                
                document.getElementById('games-container').appendChild(gameCard);
            }
            
            // Update score and turn
            document.getElementById(`score-${workerId}`).textContent = score;
            document.getElementById(`turn-${workerId}`).textContent = turn;
            
            // Update board
            const boardElem = document.getElementById(`board-${workerId}`);
            boardElem.innerHTML = '';
            
            for (let i = 0; i < board.length; i++) {
                for (let j = 0; j < board[i].length; j++) {
                    const tile = document.createElement('div');
                    tile.className = `tile tile-${board[i][j]}`;
                    tile.textContent = board[i][j] === 0 ? '' : board[i][j];
                    boardElem.appendChild(tile);
                }
            }
        }
        
        // Update charts
        function updateCharts(data) {
            // Update loss chart
            lossChart.data.labels = data.loss_history.map(item => `Epoch ${item[0]}`);
            lossChart.data.datasets[0].data = data.loss_history.map(item => item[1]);
            lossChart.update();
            
            // Update score chart
            scoreChart.data.labels = data.score_history.map(item => `Epoch ${item[0]}`);
            scoreChart.data.datasets[0].data = data.score_history.map(item => item[1]);
            scoreChart.update();
            
            // Update tile chart
            tileChart.data.labels = data.tile_history.map(item => `Epoch ${item[0]}`);
            tileChart.data.datasets[0].data = data.tile_history.map(item => item[1]);
            tileChart.update();
        }
        
        // Handle initial status
        socket.on('status', function(data) {
            updateStats(data.training_stats);
            
            // Update all games
            const gamesContainer = document.getElementById('games-container');
            gamesContainer.innerHTML = '';  // Clear "waiting" message
            
            for (const [workerId, game] of Object.entries(data.worker_games)) {
                updateGame(workerId, game.board, game.score, game.turn);
            }
            
            // Update charts
            updateCharts(data);
        });
        
        // Handle game updates
        socket.on('game_update', function(data) {
            updateGame(data.worker_id, data.board, data.score, data.turn);
        });
        
        // Handle stats updates
        socket.on('stats_update', function(data) {
            updateStats(data);
            
            // Fetch full status to update charts
            fetch('/api/status')
                .then(response => response.json())
                .then(data => updateCharts(data));
        });
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            
            // Initial fetch of data
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    updateStats(data.training_stats);
                    updateCharts(data);
                    
                    // Update all games
                    const gamesContainer = document.getElementById('games-container');
                    if (Object.keys(data.worker_games).length > 0) {
                        gamesContainer.innerHTML = '';  // Clear "waiting" message
                        
                        for (const [workerId, game] of Object.entries(data.worker_games)) {
                            updateGame(workerId, game.board, game.score, game.turn);
                        }
                    }
                });
        });
    </script>
</body>
</html>""")

# Main function to run the server
def run_server(host='0.0.0.0', port=5000, open_browser=True):
    """Run the monitoring server"""
    ensure_template_exists()
    
    # Open browser automatically
    if open_browser:
        url = f"http://localhost:{port}"
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    
    print(f"Starting AlphaZero 2048 monitoring server at http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=False)

if __name__ == "__main__":
    run_server()