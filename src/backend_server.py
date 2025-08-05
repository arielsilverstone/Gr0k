# ============================================================================
#  File: backend_server.py
#  Version: 1.04 (Corrected)
# ============================================================================
# SECTION 1: Global Variable Definitions & Imports
# ============================================================================
import sys
import os

# --- PATH FIX ---
# This block forces the project's 'src' directory onto the Python path.
# This makes the script runnable even in a broken environment.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- END PATH FIX ---

import json
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from fastapi import Body,FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from orchestrator import Orchestrator
from config_manager import ConfigManager
from rule_engine import RuleEngine
from websocket_manager import WebSocketManager
from config_validate import validate_config
from secure_secrets import load_secrets


# Remove conflicting PyPaks directory from sys.path
if 'D:\\Program Files\\Dev\\Tools\\PyPaks' in sys.path:
    sys.path.remove('D:\\Program Files\\Dev\\Tools\\PyPaks')

websocket_manager = WebSocketManager()
config_manager = ConfigManager()
agent_configs = config_manager.get().llm_configurations
rule_engine = RuleEngine(config_manager, agent_configs=agent_configs)
orchestrator = Orchestrator(config_manager=config_manager, rule_engine=rule_engine, websocket_manager=websocket_manager)

# =========================================================================
# SECTION 2: Functions
# =========================================================================
# Async Function 2.1: lifespan
# =========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    logger.info("Application starting up...")
    # Orchestrator is initialized on creation; agents are loaded in its __init__.
    yield
    # Code to run on shutdown
    logger.info("Application shutting down.")
    await orchestrator.shutdown()

app = FastAPI(lifespan=lifespan)
APP_PORT = 9102

# Add CORS middleware to allow WebSocket connections from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# =========================================================================
# SECTION 3: MCP Test Endpoints
# =========================================================================
@app.get("/health")
async def health_check():
    """Provides a simple health check endpoint."""
    logger.info("Health check endpoint called.")
    return {"status": "ok"}

@app.post("/start")
async def start_server():
    """Provides a placeholder start endpoint."""
    logger.info("Start endpoint called.")
    return {"status": "started"}

@app.post("/stop")
async def stop_server():
    """Provides a placeholder stop endpoint."""
    logger.info("Stop endpoint called.")
    return {"status": "stopped"}

@app.post("/infer")
async def infer(payload: dict = Body(...)):
    """Provides a placeholder inference endpoint that echoes the prompt."""
    prompt = payload.get("prompt", "No prompt provided")
    logger.info(f"Inference endpoint called with prompt: {prompt}")
    return {"result": f"Received prompt: {prompt}"}

# =========================================================================
# SECTION 4: Original Endpoints
# =========================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown_client"
    logger.info(f"[WS] New WebSocket connection from {client}")
    
    try:
        # Accept the WebSocket connection
        logger.info(f"[WS] Accepting connection from {client}")
        await websocket.accept()
        logger.info(f"[WS] Connection accepted for {client}")
        
        # Add to connection manager
        logger.info(f"[WS] Adding client {client} to connection manager")
        await websocket_manager.connect(websocket)
        logger.info(f"[WS] Client {client} added to connection manager")
        
        # Send welcome message
        welcome_msg = {"status": "connected", "message": "Welcome to the Gemini Agent WebSocket server"}
        await websocket.send_text(json.dumps(welcome_msg))
        
        # Main message loop
        while True:
            try:
                # Wait for a message from the client
                logger.info(f"[WS] Waiting for message from {client}")
                data = await websocket.receive_text()
                logger.info(f"[WS] Received message from {client}: {data}")
                
                try:
                    message = json.loads(data)
                    command = message.get("command")
                    payload = message.get("payload", {})
                    
                    logger.info(f"[WS] Processing command '{command}' from {client}")
                    
                    if command == "START_WORKFLOW":
                        logger.info(f"[WS] Starting workflow for {client}: {payload}")
                        # Run the workflow in the background
                        asyncio.create_task(
                            orchestrator.run_workflow(payload)
                        )
                        # Send immediate acknowledgment
                        ack_msg = {
                            "status": "accepted",
                            "message": "Workflow started",
                            "workflow": payload
                        }
                        await websocket.send_text(json.dumps(ack_msg))
                        
                    elif command == "PING":
                        logger.info(f"[WS] Received PING from {client}")
                        await websocket.send_text("PONG")
                        logger.info(f"[WS] Sent PONG to {client}")
                        
                    elif command == "TEST_CONNECTION":
                        logger.info(f"[WS] Test connection from {client}")
                        response = {
                            "status": "success",
                            "message": "Connection test successful",
                            "timestamp": str(datetime.datetime.utcnow())
                        }
                        await websocket.send_text(json.dumps(response))
                        
                    else:
                        error_msg = f"Unknown command: {command}"
                        logger.warning(f"[WS] {error_msg} from {client}")
                        await websocket.send_text(json.dumps({
                            "status": "error",
                            "message": error_msg,
                            "valid_commands": ["START_WORKFLOW", "PING", "TEST_CONNECTION"]
                        }))
                        
                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON: {str(e)}"
                    logger.error(f"[WS] {error_msg} from {client}")
                    await websocket.send_text(json.dumps({
                        "status": "error",
                        "message": error_msg,
                        "data_received": data[:100]  # Include first 100 chars of malformed data
                    }))
                    
                except Exception as e:
                    error_msg = f"Error processing message: {str(e)}"
                    logger.error(f"[WS] {error_msg} from {client}", exc_info=True)
                    await websocket.send_text(json.dumps({
                        "status": "error",
                        "message": error_msg,
                        "error_type": type(e).__name__
                    }))
                
            except WebSocketDisconnect as e:
                logger.info(f"[WS] WebSocket client {client} disconnected: {e.code} - {e.reason}")
                break
                
            except Exception as e:
                error_msg = f"Error in message loop for {client}: {str(e)}"
                logger.error(f"[WS] {error_msg}", exc_info=True)
                try:
                    await websocket.send_text(json.dumps({
                        "status": "error",
                        "message": "Internal server error",
                        "error_type": type(e).__name__
                    }))
                except:
                    logger.error("[WS] Could not send error message to client - connection may be closed")
                break
                
    except Exception as e:
        logger.error(f"[WS] WebSocket error for {client}: {str(e)}", exc_info=True)
    
    finally:
        # Clean up
        logger.info(f"[WS] Cleaning up connection for {client}")
        try:
            await websocket_manager.disconnect(websocket)
            logger.info(f"[WS] Client {client} removed from connection manager")
        except Exception as e:
            logger.error(f"[WS] Error disconnecting client {client}: {str(e)}", exc_info=True)
        
        logger.info(f"[WS] Connection closed for {client}")
        try:
            await websocket.send_text(f"[ERROR] WebSocket error: {str(e)}")
        except:
            pass  # If we can't send, the connection is likely already closed

# =========================================================================
# Async Function 3.2: ipc_handler
# =========================================================================
@app.post("/ipc")
async def ipc_handler(request: Request):
    try:
        data = await request.json()
        agent = data.get('agent', 'codegen')
        task = data.get('task', '')
        llm_api_key = data.get('llm_api_key', None)
        result = await orchestrator.handle_ipc(agent, task, llm_api_key=llm_api_key)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# SECTION 4: Backend API for Frontend Settings Modal (Task 3)
# Implements robust config retrieval and update endpoints with validation,
# secret masking, and hot-reload support.
# ============================================================================
# Async Function 4.1: get_config
# ============================================================================

@app.get("/api/get_config")
async def get_config():
    """
    Retrieves the current application configuration, masking secrets.
    """
    try:
        current_config = config_manager.get()
        config_dict = current_config.model_dump()
        # Mask secrets in config (e.g., API keys, tokens)
        secrets = load_secrets()
        if "llm_configurations" in config_dict:
            for llm, conf in config_dict["llm_configurations"].items():
                if "api_key" in conf:
                    conf["api_key"] = ""  # Mask
        if "gdrive" in config_dict:
            for k in ["client_id", "client_secret", "refresh_token"]:
                if k in config_dict["gdrive"]:
                    config_dict["gdrive"][k] = ""  # Mask
        # Mask any other secrets present in secure_secrets
        for k in secrets:
            if k in config_dict:
                config_dict[k] = ""
        return {"status": "success", "config": config_dict}
    except Exception as e:
        return {"status": "error", "message": f"Failed to retrieve config: {e}"}

# =========================================================================
# Async Function 4.2: save_config
# ============================================================================

@app.post("/api/save_config")
async def save_config(payload: dict = Body(...)):
    """
    Saves updated application configuration with schema validation and hot reload.
    """
    try:
        # Load current config for partial update
        current_config = config_manager.get().model_dump()
        def deep_merge(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    d[k] = deep_merge(d[k], v)
                else:
                    d[k] = v
            return d
        merged = deep_merge(current_config, payload)
        # Validate config before save
        valid, err = validate_config()
        if not valid:
            return {"status": "error", "message": f"Schema validation failed: {err}"}
        config_manager.save(merged)
        orchestrator.reload_config()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to save config: {e}"}

# =========================================================================
# Section 5: Main Execution
# =========================================================================

def main():
    uvicorn.run(app, host="127.0.0.1", port=APP_PORT)

if __name__ == "__main__":
    main()

#
#
## END backend_server.py
