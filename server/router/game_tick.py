import logging

from fastapi import APIRouter
from fastapi_utils.tasks import repeat_every

from server.typecheck_fighter import RedisType

from server.router.game_def_handlers import get_game_def, update_game_def


logger = logging.getLogger()

router = APIRouter()

async def game_tick(game_uuid:str,redis: RedisType) -> None:
    """Updating agents behaviour"""

    game_def = await get_game_def(game_uuid, redis)
    game_def.current_stage.minor += 1
    logger.info(f"Incremented current stage: {game_def.current_stage}")
    await update_game_def(game_uuid,game_def)


# @router.on_event("startup")
async def start_game_tick_task(game_uuid:str, redis: RedisType) -> None:    
    @repeat_every(seconds=1)  # 1 sec
    async def game_tick_task():
        await game_tick(game_uuid,redis)
    await game_tick_task()
