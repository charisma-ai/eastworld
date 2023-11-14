from __future__ import annotations
from enum import Enum

from typing import List, Optional, Set

from pydantic import UUID4, BaseModel, Field
import datetime


class GameStage(BaseModel):
    """Represents the flow of time in a game."""

    stage: int = 0
    """ 
    Should be incremented when the player undergoes some sort of tranformation. Most 
    non-permanent memories from previous stages are removed (in order of importance) and
    memories from the same stage are heavily preferred during retrieval.
    """

    major: int = 0
    """ 
    Should be incremented when the player completes a major quest line. 
    completing a quest. This adds an exponentially decaying weight to the importance of 
    the memory.
    """

    minor: int = 0
    """ 
    This is used to indicate the passage of time to the LLM.
    """

    @classmethod
    def from_time(cls,time_str:str):
        date_obj = datetime.datetime.strptime(time_str,"%I:%M%p")
        start = datetime.datetime(year=1,month=1,day=1,hour=7)        
        return cls(minor=(date_obj - start).seconds)

    #TODO: make these methods more general

    def get_day(self):
        days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        return days[self.major % 7]

    def get_time(self):
        # minor == 0 counts as 7:00 am in the morning
        start = datetime.datetime(year=1,month=1,day=1,hour=7)
        return (start + datetime.timedelta(seconds=self.minor)).strftime("%H:%M:%S")

    def __eq__(self, other: GameStage):    
        return self.stage == other.stage and self.major == other.major and self.minor == other.minor

    def __lt__(self, other: GameStage):
        # return not (other.stage < self.stage or other.major < self.major or other.minor < self.minor)    
        if self.stage > other.stage:
            return False
        if self.major > other.major:
            return False
        if self.minor > other.minor:
            return False
        return True

class MemoryConfig(BaseModel):
    max_memories: int = 1024
    embedding_dims: int = Field(...)
    """The dimensions of the vector that the embedding models return.
    i.e. OpenAI ada-002 is 1536."""

    memories_returned: int = 5
    """How many memories to return."""


class MemoryType(str, Enum):
    observation = 'observation'
    reflection = 'reflection'
    plan = 'plan'

class Memory(BaseModel):
    importance: int = 0
    """ 
    A number from 1-10 indicating the importance of the memory. An importance of 0 
    means importance will be determined by the LLM.
    """

    description: str
    type: MemoryType = MemoryType.observation
    embedding: Optional[List[float]] = None
    timestamp: Optional[GameStage] = None


class Lore(BaseModel):
    known_by: Set[UUID4] = Field(default_factory=set)
    memory: Memory


class PlanTree(BaseModel):
    value: Optional[Memory] = None
    parent: Optional[PlanTree] = None
    children: List[PlanTree] = Field(default_factory=list)