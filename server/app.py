import uvicorn
from email_triage.server import app

def start():
    uvicorn.run("email_triage.server:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    start()
