# 2048-Zero: AlphaZero for 2048 Game

This project implements the AlphaZero algorithm for the 2048 game using a client-server architecture.

## Architecture

The system follows a client-server architecture:

- **Server**: Central node that coordinates training, collects self-play data, and trains the neural network.
- **Workers**: Distributed clients that download the latest weights, perform self-play, and upload game data.

![Architecture Diagram](https://mermaid.ink/img/pako:eNptkU1PwzAMhv9KlBOgdZUGHNAOcEJC4gAcQJzW1WxdP5S4YlT9760zhm47JfbjV37txInppjhL5ZQ-SQGXgOhz0B6t3cMMptrPu1Ga_I9Ytu1Ui-WNmzHEp70-m42G4OOm1uF80uf5Pf5Wt_r2kYo91uHWCMdpwh5CpzWEwrGKYwqeDGUQ9Z5ialRZHalPtHneDdS-9kGVS-NKyHlVGjJpBi8BTQC3xp_N9IUtjNj-zfS95t953QRnbzXCbfKjpHZAI-nYX0QRVgGfcZEujcXnJAYfMQnEfE-J5iIsK1Zki_maVcXyMJrn0WLJFi_seCwXMzavonQRkrPSIuvCTiNKlnYuH_K9OSXf31OdxQ?type=png)

## Running the System

### Server

Start the central server:

```bash
# Standard mode (using R2 for weights storage)
python run_server.py --host 0.0.0.0 --port 8000 --auth-token YOUR_TOKEN

# Local mode (serve weights from the local server)
python run_server.py --host 0.0.0.0 --port 8000 --auth-token YOUR_TOKEN --localhost

# Reset all state and use shorter deadlines for faster iterations (great for testing)
python run_server.py --reset --localhost --initial-deadline 5 --training-deadline 5
```

Server configuration:
- `--localhost`: Serve weights directly from the server instead of using R2
- `--reset`: Reset all state (clear WAL and snapshots) for a fresh start
- `--config`: Path to custom config YAML file
- `--host`: Host to bind to (overrides config)
- `--port`: Port to bind to (overrides config)
- `--auth-token`: Authentication token (overrides config)
- `--initial-deadline`: Initial deadline in minutes (default: 30)
- `--training-deadline`: Training deadline in minutes (default: 30)
- `--check-interval`: Check interval in seconds (default: 10)

The server workflow:
1. Accept connections from workers and provide the current model weights
2. Workers continuously check the server state through the HTTP API
3. Workers play games and upload data to the queue as long as:
   - time_remaining > 0
   - is_training = false
   - They have the latest weights
4. When the deadline passes and enough games are collected, the server:
   - Sets is_training = true (workers will stop uploading)
   - Trains a new model revision
   - Updates the weights URL and SHA256 hash
   - Sets a new deadline and sets is_training = false
5. Workers then download the new weights and start playing with the new model

### Workers

Start one or more worker clients:

```bash
python run_worker.py --server-url http://SERVER_IP:8000 --auth-token YOUR_TOKEN
```

Worker configuration:
- `--server-url`: URL of the trainer server
- `--auth-token`: Authentication token
- `--weights-dir`: Directory to store weights
- `--batch-size`: Number of games to collect before uploading

Workers will:
- Download the latest model weights
- Perform self-play games
- Upload game data to the server
- Repeat the cycle

## Project Structure

- `/server/` - Server implementation
  - `server.py` - FastAPI server
  - `state_manager.py` - Queue and WAL management
  - `trainer.py` - Model training process
  - `auth.py` - Authentication middleware
  - `r2_client.py` - R2/S3 client for weight storage
  - `config.py` - Configuration management
- `/worker/` - Worker client
  - `worker.py` - Self-play client implementation
- `/zero/` - Core game logic
  - `game.py` - Game rules
  - `zeromodel.py` - Neural network model
  - `zero2048.py` - 2048-specific AlphaZero implementation
  - `zeromonitoring.py` - Monitoring utilities
- `run_server.py` - Server entry point
- `run_worker.py` - Worker entry point

## Key Features

1. **Distributed Training**: Parallelized self-play across many machines
2. **Centralized Updates**: Single training node for consistent model updates
3. **Fault Tolerance**: WAL and snapshot recovery system
4. **Authentication**: Token-based security
5. **Flexible Weight Distribution**: 
   - Option to serve weights directly from server with `--localhost` flag
   - Cloud storage integration with R2 (S3-compatible) for scalable deployment
6. **Stateless Workers**: Clients can recover from failures and automatically sync with the latest model

## Requirements

- Python 3.8+
- PyTorch
- FastAPI & Uvicorn
- Requests
- Zstandard & LZ4 (for compression)
- Boto3 (for R2/S3 integration)
- PyYAML (for configuration)
- Wandb (for monitoring, optional)

---

## Previous Implementations

### Minmax

This project also includes a minmax implementation for 2048.

```bash
# multiple games
python main.py 2048-best --games <N>
# single game
python main.py 2048
```