# Anton AI Assistant - Dev/Prod Environment Setup

This document explains the new dev/prod environment setup for Anton AI Assistant.

## Overview

The system has been restructured to support separate development and production environments:

- **vLLM**: Shared AI model server (runs once, persistent)
- **Development**: Interactive environment with hot-reload capabilities
- **Production**: Background tmux-based environment for deployment

## Environment Details

### Ports Configuration

| Environment | Agent Port | UI Port | vLLM Port |
|-------------|------------|---------|-----------|
| Development | 8001       | 7860    | 8003      |
| Production  | 9001       | 9860    | 8003      |

### Scripts

#### 1. `./start_vllm_tmux.sh`
- **Purpose**: Starts vLLM AI model server in a persistent tmux session
- **When to use**: Run once when setting up the system
- **Persistence**: Survives SSH disconnections and system restarts
- **Tmux session**: `anton-vllm`

#### 2. `./run_dev.sh`
- **Purpose**: Development environment with hot-reload
- **Features**: 
  - Interactive console
  - Hot-reload: Press 'r' to restart agent+UI, 'a' for agent only, 'u' for UI only
  - Real-time logging and debugging
- **Best for**: Development, debugging, testing

#### 3. `./run_prod.sh`
- **Purpose**: Production environment in tmux
- **Features**:
  - Runs in background tmux session
  - Separate ports from dev environment
  - Persistent across SSH disconnections
- **Tmux session**: `anton-prod`
- **Best for**: Production deployment, demos

#### 4. `./setup_help.sh`
- **Purpose**: Shows system status and detailed help
- **Features**: 
  - Environment status check
  - Port availability
  - tmux session information
  - Common commands reference

## Quick Start

### First Time Setup
```bash
# 1. Start vLLM (one-time setup)
./start_vllm_tmux.sh

# 2. Start development environment
./run_dev.sh
```

### Daily Development Workflow
```bash
# Check system status
./setup_help.sh

# Start development (if not already running)
./run_dev.sh

# Hot-reload during development:
# Press 'r' in the dev console to restart agent+UI
# Press 'a' to restart agent only
# Press 'u' to restart UI only
```

### Production Deployment
```bash
# Start production environment
./run_prod.sh

# Monitor production
tmux attach-session -t anton-prod

# Detach but leave running (Ctrl+b, then d)
```

## Tmux Management

### Useful Commands
```bash
# List all tmux sessions
tmux list-sessions

# Attach to vLLM session
tmux attach-session -t anton-vllm

# Attach to production session
tmux attach-session -t anton-prod

# Kill a session (stop services)
tmux kill-session -t anton-vllm    # Stop vLLM
tmux kill-session -t anton-prod    # Stop production

# Detach from session (leave running)
# Press: Ctrl+b, then d
```

### Tmux Session Layout

#### anton-vllm
- Single window running vLLM server
- Shared between dev and prod

#### anton-prod
- Window 0 (`agent`): Agent server on port 9001
- Window 1 (`ui`): Chainlit UI on port 9860

## Environment Variables

The system automatically sets these environment variables:

```bash
# Development
export ANTON_ENV="dev"
export AGENT_PORT=8001
export UI_PORT=7860
export VLLM_PORT=8003

# Production  
export ANTON_ENV="prod"
export AGENT_PORT=9001
export UI_PORT=9860
export VLLM_PORT=8003
```

## Configuration Files Updated

The following files have been updated to respect environment-based ports:

- `client/config.py` - Uses `AGENT_PORT` environment variable
- `server/model_server.py` - Uses `VLLM_PORT` environment variable
- `server/agent/react/react_agent.py` - Uses `VLLM_PORT` environment variable
- `server/agent/agentic_flow/helpers_and_prompts.py` - Uses `VLLM_PORT` environment variable
- `server/agent/agentic_flow/full_agentic_flow.py` - Uses `VLLM_PORT` environment variable
- `server/agent/self_study.py` - Uses `AGENT_PORT` environment variable

## Troubleshooting

### Check System Status
```bash
./setup_help.sh
```

### Common Issues

1. **vLLM not responding**: 
   ```bash
   # Check if vLLM is running
   tmux attach-session -t anton-vllm
   
   # If not running, start it
   ./start_vllm_tmux.sh
   ```

2. **Port conflicts**:
   - Development uses ports 8001, 7860
   - Production uses ports 9001, 9860
   - vLLM always uses port 8003

3. **Permission issues**:
   ```bash
   sudo chmod -R 777 /home/lucas/anton_new/pgdata
   ```

4. **Docker postgres not running**:
   ```bash
   docker start chainlit-pg
   ```

## Migration from Old Setup

The old `run.sh` script has been deprecated. When you run it, you'll get options to use the new scripts. The old script is preserved as `run_old.sh` for reference.

## Benefits of New Setup

1. **Better Resource Management**: vLLM runs once and is shared
2. **SSH Persistence**: Production runs in tmux, survives disconnections
3. **Port Isolation**: Dev and prod can run simultaneously
4. **Hot-reload Development**: Faster iteration during development
5. **Production Monitoring**: Easy to monitor via tmux
6. **Environment Consistency**: Clear separation of concerns
