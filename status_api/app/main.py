from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel

from time import time
from datetime import datetime

app = FastAPI() # Create an instance of the app.  

status = "Unknown"
last_changed = "Never"

class Update(BaseModel):
    status: str

class StatusReport(BaseModel):
    status: str
    last_changed:str

@app.post("/status")
def update_status(update: Update):
    global status, last_changed 

    if update.status != status:
        last_changed = datetime.now().strftime("%Y %m %d %H:%M:%S")

    status = update.status

@app.get("/status")  
def read_status():
    return {"status": status, "last_changed": last_changed}
   


