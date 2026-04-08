from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    return {
        "observation": {"traffic": "normal"},
        "reward": 0,
        "done": False
    }

@app.post("/step")
def step(action: dict):
    return {
        "observation": {"traffic": "updated"},
        "reward": 1,
        "done": False
    }
