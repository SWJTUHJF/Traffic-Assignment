from typing import Literal, get_args

CostType = Literal["tt", "mtt"]

def cost_type_list() -> list[str]:
    return list(get_args(CostType))