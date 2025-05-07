# 2048-Zero Client-Server Architecture

This is a client-server implementation of the AlphaZero algorithm for the 2048 game. It allows for distributed self-play and centralized training.

## Architecture Overview

The system follows a client-server architecture:

- **Server**: Central node that coordinates training, collects self-play data, and trains the neural network.
- **Workers**: Distributed clients that download the latest weights, perform self-play, and upload game data.

### Components

1. **Server**
   - FastAPI HTTP server with authentication
   - In-memory queue + write-ahead logging (WAL) + snapshots for durability
   - Isolated training process
   - Weight publishing and management with multiple storage backends
   - Support for local filesystem, HTTP server, and Cloudflare R2 storage
   - Metrics logging with WandB integration

2. **Workers**
   - Stateless clients that poll the server for commands
   - Download weights and verify integrity via SHA256
   - Perform self-play games with MCTS
   - Upload batches of games to the server
   - Exponential backoff on errors

## API Endpoints

- `GET /state`: Get current training parameters
- `GET /command`: Get worker control message
- `POST /upload`: Submit a batch of finished games
- `POST /metrics`: Send aggregated self-play stats

## Running the Server

```bash
python run_server.py --host 0.0.0.0 --port 8000 --auth-token your-secret-token
```

## Running Workers

```bash
python run_worker.py --server-url http://server-address:8000 --auth-token your-secret-token
```

## Directory Structure

```
project/
├── server/             # Server implementation
│   ├── auth.py         # Authentication middleware
│   ├── config.py       # Configuration management
│   ├── r2_client.py    # Cloudflare R2 client
│   ├── server.py       # FastAPI server
│   ├── state_manager.py # Queue, WAL, snapshots
│   ├── storage_adapter.py # Storage backend adapter
│   └── trainer.py      # Training logic
├── worker/             # Worker client
│   └── worker.py       # Worker implementation
├── zero/               # Core game logic
│   ├── game.py         # Game rules and state
│   ├── zeromodel.py    # Neural network model
│   └── zeromonitoring.py # WandB integration
├── run_server.py       # Server entry point
└── run_worker.py       # Worker entry point
```

## Configuration

Server configuration can be provided via a YAML file:

```yaml
server:
  host: 0.0.0.0
  port: 8000
  auth_token: your-secret-token
  snapshot_interval: 60  # seconds
  localhost_weights: false  # Whether to serve weights via HTTP

training:
  min_games: 512
  epochs: 10
  batch_size: 64
  learning_rate: 0.001

self_play:
  num_simulations: 800
  c_puct: 1.5
  dirichlet_alpha: 0.3
  epsilon: 0.25
  max_moves_per_game: 1000
  temperature: 1.0

storage:
  weights_dir: "weights"
  wal_dir: "wal"
  snapshot_dir: "snapshots"
  use_r2: false  # Whether to use R2 for storage
  r2_bucket: "your-bucket-name"
  r2_access_key: "your-access-key"
  r2_secret_key: "your-secret-key"
  r2_endpoint_url: "https://<account-id>.r2.cloudflarestorage.com"
  r2_public_url: "https://weights.yourdomain.com"  # Optional

wandb:
  enabled: true
  project: 2048-zero
  entity: your-username
```

## Security Considerations

- Authentication is implemented using a shared secret token
- All API endpoints require the token to be present in the `Authorization: Bearer <token>` header
- HTTPS should be used in production to encrypt traffic

## Durability & Recovery

- Write-ahead logging (WAL) ensures no data is lost on server restart
- Snapshots are taken periodically to speed up recovery
- Workers verify weight integrity via SHA256 hashes
- Exponential backoff with jitter on connection errors