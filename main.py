
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware

from routers import profile, objectives, current_state, resources, constraints, motivation, execution

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(profile.router)
app.include_router(objectives.router)
app.include_router(current_state.router)
app.include_router(resources.router)
app.include_router(constraints.router)
app.include_router(motivation.router)
app.include_router(execution.router)



