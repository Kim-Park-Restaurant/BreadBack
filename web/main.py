from fastapi import FastAPI
from pydantic import BaseModel


class Item(BaseModel):
    name : str
    description : str | None = None
    price : float
    tax : float | None = None


app = FastAPI()


@app.get("/items")
async def create_items(item : Item):
    print(f"received : {item.name}, {item.description} {item.price} {item.tax}")
    return item
    