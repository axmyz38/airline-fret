from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

from load_planner import CargoHold, ULD, LoadPlanner

app = FastAPI(title="Cargo Load Planner")


class ULDInput(BaseModel):
    id: str
    length: float
    width: float
    height: float
    weight: float
    priority: int = 0


class HoldInput(BaseModel):
    length: float
    width: float
    height: float


class LoadPlanRequest(BaseModel):
    hold: HoldInput
    ulds: List[ULDInput]


@app.post("/load-plan")
def create_load_plan(request: LoadPlanRequest):
    hold = CargoHold(**request.hold.dict())
    ulds = [ULD(**u.dict()) for u in request.ulds]
    planner = LoadPlanner()
    placements, unplaced = planner.plan_load(hold, ulds)
    return {"placements": placements, "unplaced": [u.id for u in unplaced]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
