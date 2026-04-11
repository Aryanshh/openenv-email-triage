import uvicorn
from atlas_eco.server import app

def main():
    uvicorn.run("atlas_eco.server:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
