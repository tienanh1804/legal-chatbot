# Legal RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot system specialized in Vietnamese legal policy information, featuring hybrid search capabilities, conversation history tracking, and user authentication.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements a RAG (Retrieval-Augmented Generation) chatbot system that specializes in answering questions about Vietnamese legal policies. The system combines vector search with BM25 for efficient and accurate document retrieval, and uses Google's Gemini API for generating natural language responses.

The application features a web interface with a chat-based UI, user authentication, conversation history tracking, and the ability to display relevant document sources for legal questions.

## Features

- **Hybrid Search**: Combines vector embeddings with BM25 for optimal document retrieval
- **Legal Question Classification**: Automatically detects if a query is related to legal policies
- **Conversation History**: Maintains context across multiple questions in a conversation
- **User Authentication**: Secure login system with JWT tokens
- **Source References**: Displays relevant document sources for legal questions
- **API Key Fallback**: Automatically tries alternative API keys if one fails
- **Responsive UI**: Chat interface with sidebar history and toggle functionality
- **Docker Support**: Containerized deployment for easy setup and scaling

## Project Structure

```
.
├── backend/
│   ├── api/                  # API endpoints and routers
│   ├── auth/                 # User authentication
│   ├── core/                 # Core files (config, database, models)
│   ├── search/               # Hybrid search and text processing
│   ├── utils/                # Utilities
│   ├── scripts/              # Scripts for building cache and data
│   ├── cache/                # Directory for cache files
│   ├── markdown_data/        # Markdown format text data
│   ├── classification_models/ # Legal question classification models
│   └── migrations/           # Database migrations
├── frontend/
│   ├── css/                  # CSS files
│   ├── js/                   # JavaScript files
│   │   ├── components/       # UI components
│   │   ├── services/         # Services
│   │   └── utils/            # Utilities
│   └── pages/                # HTML files
└── docker-compose.yml        # Docker Compose configuration
```

## Installation

### Using Docker

1. Install Docker and Docker Compose
2. Copy `.env.example` to `.env` in the project root and set secrets (never commit `.env`):
   ```
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=postgres
   POSTGRES_DB=ragapp
   GEMINI_API_KEY=your_gemini_api_key
   JWT_SECRET_KEY=your_jwt_secret_key
   ```
3. Run the application:
   ```bash
   docker-compose up -d
   ```
4. Access the application at http://localhost:8088

### Running Directly (without Docker)

#### Backend

1. Install Python 3.9+
2. Install the required libraries:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
3. Initialize the database:
   ```bash
   python scripts/init_db.py
   ```
4. Build cache for hybrid search:
   ```bash
   python scripts/build_search_resources.py
   ```
5. Run the application:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

#### Frontend

1. Use Live Server or any web server to serve the static files
2. Access the application at http://localhost:5500/frontend/pages/index.html

## Usage

### User Registration and Login

1. Access the web interface at http://localhost:8088
2. Register a new account or log in with existing credentials
3. Start asking questions in the chat interface

### Asking Questions

- For general questions, type your query and press Enter
- For legal policy questions, the system will automatically retrieve relevant documents and provide sources
- Continue the conversation with follow-up questions to maintain context

### Viewing Conversation History

- Click the history toggle button on the left side to view previous conversations
- Click on a conversation to view its details
- Use the "New Conversation" button to start a fresh chat

## API Documentation

The API documentation is available at http://localhost:8002/docs when the application is running.

### Main Endpoints

- `POST /query`: Submit a question and get an answer
- `GET /health`: Health check endpoint
- `GET /auth/verify`: Verify authentication token
- `POST /users/register`: Register a new user
- `POST /users/token`: Login and get token
- `GET /users/me`: Get current user information
- `GET /history`: Get conversation history
- `POST /history`: Save a question to history
- `GET /history/{history_id}`: Get a specific history item
- `GET /history/conversation/{conversation_id}`: Get all messages in a conversation
- `DELETE /history/{history_id}`: Delete a history item
- `DELETE /history`: Delete all history of the current user

## Development

### Rebuilding Cache

To rebuild the cache from markdown data:

```bash
cd backend
python scripts/build_search_resources.py --force
```

### Adding New Legal Documents

1. Add markdown files to the `backend/markdown_data` directory
2. Rebuild the search resources:
   ```bash
   python backend/scripts/build_search_resources.py
   ```

### Updating Models

To update the embedding model or classification model:

1. Place the new model files in the appropriate directory
2. Update the configuration in `backend/core/config.py`
3. Restart the application

## Troubleshooting

### Common Issues

- **Database Connection Errors**: Ensure PostgreSQL is running and the connection string is correct
- **API Key Errors**: Verify your Gemini API key is valid and properly configured
- **Model Loading Errors**: Check that all required model files are present in the correct directories
- **Docker Issues**: Try rebuilding the containers with `docker-compose build --no-cache`

### Logs

- Backend logs: `docker logs rag_backend_1`
- Frontend logs: `docker logs rag_frontend_1`
- Database logs: `docker logs rag_db_1`
