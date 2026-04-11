import random
from typing import List, Tuple, Dict
from atlas_eco.models import (
    Observation, Inventory, Shipment, Order, 
    Action, ActionType, TransportMode, PartType
)

class AtlasEcoEnv:
    # Transport Mode Specs: (Cost multiplier, Speed/ETA, Carbon unit)
    TRANSPORT_SPECS = {
        TransportMode.SEA:  (1.0, 10, 0.1),
        TransportMode.AIR:  (5.0, 2,  2.0),
        TransportMode.RAIL: (2.5, 5,  0.5),
        TransportMode.ROAD: (1.5, 4,  0.8)
    }

    def __init__(self, task: str = "easy"):
        self.task = task
        self.reset()

    def reset(self, seed: int = 42) -> Observation:
        random.seed(seed)
        self.step_count = 0
        self.inventory = Inventory()
        self.active_shipments: List[Shipment] = []
        self.pending_orders: List[Order] = self._generate_initial_orders()
        self.carbon_total = 0.0
        self.cash_balance = 10000.0
        self.carbon_quota = 1000.0 if self.task == "hard" else 2000.0
        self.done = False
        return self._get_obs()

    def _generate_initial_orders(self) -> List[Order]:
        return [
            Order(id="ORD_001", product="EcoPhone", quantity=5, due_date=15, reward=500.0),
            Order(id="ORD_002", product="GreenTab", quantity=3, due_date=25, reward=800.0)
        ]

    def _get_obs(self, news: str = None) -> Observation:
        return Observation(
            step=self.step_count,
            inventory=self.inventory,
            active_shipments=self.active_shipments,
            pending_orders=self.pending_orders,
            carbon_total=self.carbon_total,
            carbon_quota=self.carbon_quota,
            cash_balance=self.cash_balance,
            news=news
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        self.step_count += 1
        reward = 0.0
        info = {}

        # 1. Process Action
        if action.action_type == ActionType.ORDER_PARTS:
            reward += self._handle_order_parts(action, info)
        elif action.action_type == ActionType.PRODUCE:
            reward += self._handle_production(action, info)
        elif action.action_type == ActionType.OFFSET:
            reward += self._handle_offset(action, info)
        elif action.action_type == ActionType.REROUTE:
            reward += self._handle_reroute(action, info)

        # 2. Advance Shipments
        for ship in self.active_shipments:
            ship.eta -= 1
            if ship.eta <= 0:
                self._receive_shipment(ship)
        
        self.active_shipments = [s for s in self.active_shipments if s.eta > 0]

        # 3. Check Order Deadlines
        for order in self.pending_orders:
            if self.step_count > order.due_date:
                reward -= 50.0  # Late penalty
        
        # 4. Check Termination
        if self.step_count >= 50 or not self.pending_orders:
            self.done = True
            info["final_score"] = self._calculate_final_score()

        return self._get_obs(), reward, self.done, info

    def _handle_order_parts(self, action: Action, info: dict) -> float:
        if not action.part_type or not action.mode or not action.quantity:
            return -5.0
        
        base_cost = 10.0 * action.quantity
        mult, eta, carbon = self.TRANSPORT_SPECS[action.mode]
        total_cost = base_cost * mult
        
        if self.cash_balance < total_cost:
            info["error"] = "Insufficient funds"
            return -10.0
        
        self.cash_balance -= total_cost
        self.carbon_total += carbon * action.quantity
        
        new_ship = Shipment(
            id=f"SHP_{len(self.active_shipments) + 1}",
            part=action.part_type,
            quantity=action.quantity,
            mode=action.mode,
            eta=eta,
            carbon_impact=carbon * action.quantity,
            cost=total_cost
        )
        self.active_shipments.append(new_ship)
        return 2.0 

    def _receive_shipment(self, ship: Shipment):
        current_val = getattr(self.inventory, ship.part.value)
        setattr(self.inventory, ship.part.value, current_val + ship.quantity)

    def _handle_production(self, action: Action, info: dict) -> float:
        if not action.product: return -5.0
        if action.product == "EcoPhone":
            if self.inventory.chips >= 1 and self.inventory.sensors >= 1:
                self.inventory.chips -= 1
                self.inventory.sensors -= 1
                for i, o in enumerate(self.pending_orders):
                    if o.product == "EcoPhone":
                        o.quantity -= 1
                        if o.quantity <= 0:
                            self.pending_orders.pop(i)
                            self.cash_balance += o.reward
                            return 100.0
                        return 10.0
            else:
                info["error"] = "Missing parts"
                return -10.0
        return 0.0

    def _handle_offset(self, action: Action, info: dict) -> float:
        if not action.offset_amount: return -5.0
        cost = action.offset_amount * 2.0
        if self.cash_balance >= cost:
            self.cash_balance -= cost
            self.carbon_total = max(0.0, self.carbon_total - action.offset_amount)
            return 5.0
        return -10.0

    def _handle_reroute(self, action: Action, info: dict) -> float:
        return 0.0

    def _calculate_final_score(self) -> float:
        carbon_perf = max(0.01, min(0.99, 1.0 - (self.carbon_total / self.carbon_quota)))
        return carbon_perf
