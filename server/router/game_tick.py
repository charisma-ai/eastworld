import logging

from fastapi import APIRouter
from fastapi_utils.tasks import repeat_every

from server.typecheck_fighter import RedisType

from server.router.game_def_handlers import update_game_def
from game.session import Session

logger = logging.getLogger()

router = APIRouter()

async def game_tick(game_uuid:str, session:Session,redis: RedisType) -> None:
    """Updating agents behaviour"""


    session.game_def.current_stage.minor += 1
    logger.info(f"Incremented current stage: {session.game_def.current_stage}")
    await update_game_def(game_uuid,session.game_def,redis=redis)


# @router.on_event("startup")
async def start_game_tick_task(game_uuid:str,session:Session, redis: RedisType) -> None:    
    @repeat_every(seconds=1)  # 1 sec
    async def game_tick_task():
        await game_tick(game_uuid,session,redis)

    if not session.is_ticking:
        session.is_ticking = True
        await game_tick_task()
