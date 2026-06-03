# Execution Flow

## Overview
This document represents the overall macro execution lifecycle of the JudgeResearch multi-service platform.

## Pre-Requisites
1. Node processes and Python Uvicorn instances are active on host ports.
2. The SQLite schema (`initialize_database.py`) has been generated.

## The Lifecycle

### 1. Bootstrapping
- **Backend Initialized**: `main.py` binds to loopback.
- **DeepWiki API**: `deepwiki-open/api/main.py` executes, preparing LLM API keys via environment variables.

### 2. Standard Usage
- Client web sessions load static bundles into browsers. 
- A user session is established.
- User toggles between the **Live Coding** platform and **DeepWiki** modules.
- The web router dynamically fetches required context through REST calls. 

### 3. Asymmetric Job Offloading 
- High-intensity routines like Code Execution or DeepWiki Git-ingestions bypass the main HTTP blocking loop of Backend API by leveraging async tasks or websockets.

### 4. Teardown
- Containers or PM2/virtualenv wrappers receive SIGTERM. 
- FastApi drains connection pools to SQLite.
