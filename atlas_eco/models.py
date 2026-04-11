from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class TransportMode(str, Enum):
    SEA = "sea"
    AIR = "air"
    RAIL = "rail"
    ROAD = "road"

class PartType(str, Enum):
    CHIPS = "chips"
    SENSORS = "sensors"
    BATTERIES = "batteries"
    CASING = "casing"

class Shipment(BaseModel):
    id: str
    part: PartType
    quantity: int
    mode: TransportMode
    eta: int  # steps remaining
    carbon_impact: float
    cost: float

class Order(BaseModel):
    id: str
    product: str
    quantity: int
    due_date: int
    reward: float

class Inventory(BaseModel):
    chips: int = 0
    sensors: int = 0
    batteries: int = 0
    casing: int = 0

class Observation(BaseModel):
    step: int
    inventory: Inventory
    active_shipments: List[Shipment]
    pending_orders: List[Order]
    carbon_total: float
    carbon_quota: float
    cash_balance: float
    news: Optional[str] = None

class ActionType(str, Enum):
    ORDER_PARTS = "order_parts"
    REROUTE = "reroute"
    PRODUCE = "produce"
    OFFSET = "offset"
    SKIP = "skip"

class Action(BaseModel):
    action_type: ActionType
    part_type: Optional[PartType] = None
    quantity: Optional[int] = None
    mode: Optional[TransportMode] = None
    shipment_id: Optional[str] = None
    product: Optional[str] = None
    offset_amount: Optional[float] = None

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict
