

import sys
import os

# Add the parent directory to sys.path so we can import 'orch'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app import database, models, schemas, auth
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse # Ensure this is at the top
import json
# Import the orchestrator
from backend.orch import run_orchestrator

# Create tables
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="LitScout Backend")

# --- 1. CORS SETUP ---
# Note: I added port 3000 (Create React App) and 5173 (Vite) just in case
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Dependency: DB session
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- AUTH ENDPOINTS ---

@app.post("/signup", response_model=schemas.UserResponse)
def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pw = auth.hash_password(user.password)

    new_user = models.User(
        full_name=user.full_name,
        email=user.email,
        password=hashed_pw
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/login")
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user or not auth.verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = auth.create_access_token({"sub": db_user.email})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/about", response_model=schemas.AboutResponse)
def get_about():
    return {
        "mission": "LitScout automates systematic literature reviews using AI.",
        "how_it_works": [
            "Phase 1: Search & Filter with Semantic Scholar",
            "Phase 2: AI-Powered Screening with LLMs",
            "Phase 3: Analysis & Reporting"
        ]
    }


# --- RESEARCH ENDPOINTS (THE FIX IS HERE) ---

class ResearchRequest(BaseModel):
    query: str


    
@app.post("/start-research")
async def start_research(request: ResearchRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    async def stream_generator():
        # We call the NEW generator function we just made in orch.py
        from backend.orch import run_orchestrator
        async for update in run_orchestrator(request.query):
            # This formatting is required for the browser to see it as a stream
            yield f"data: {json.dumps(update)}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")