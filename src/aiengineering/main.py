from fastapi import FastAPI

app = FastAPI(title="aiengineering")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
