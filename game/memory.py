import logging
import bisect
from typing import Dict, List, Optional, Tuple

from game.ti_retriever import TIRetriever
from llm.base import LLMBase

# from eastworld.wrappers.openai
from schema import Memory, Message
from schema.memory import GameStage, MemoryType, PlanTree

_MEM_IMPORTANCE_TMPL = """On the scale of 0 to 9, where 0 is purely mundane"
(e.g., brushing teeth, making bed) and 9 is
extremely poignant (e.g., a break up, college
acceptance), rate the likely poignancy of the
following piece of memory. Respond with a single integer without
explanation.
\nMemory: {memory_content}
\nRating: """

class GenAgentMemory:
    # TODO: make LLM configurable
    def __init__(
        self,
        llm_interface: LLMBase,
        default_num_memories_returned: int,
        retriever: TIRetriever,
    ):
        self._llm_interface = llm_interface
        self._default_num_memories_returned = default_num_memories_returned
        self._retriever = retriever
        self._plan = PlanTree()
        self._current_plan_step = None

    def get_broad_plan(self) -> List[Memory]:        
        return [step.value for step in self._plan.children if step.value]

    def get_current_plan_step(self, timestamp: GameStage) -> Tuple[PlanTree, int]:

        def _traverse_tree(plan:PlanTree, level:int=0) -> Tuple[PlanTree, int]:
            if len(plan.children) != 0:                
                for step in plan.children[::-1]:
                    if step.value and timestamp > step.value.timestamp:
                        return _traverse_tree(step,level=level+1)
            return plan, level

        plan_step, level = _traverse_tree(self._plan)
        return plan_step, level
        
        
    async def add_plan_step(self, memory: Memory, parent:PlanTree) -> None:
        assert memory.type == MemoryType.plan        
        bisect.insort(parent.children,PlanTree(value=memory,parent=parent),key=lambda x: x.value.timestamp)

        # TODO: add memory preprocessing        
        await self.add_memory(memory)

    async def add_memory(self, memory: Memory) -> None:
        # TODO: parallelize
        if memory.importance == 0:
            memory.importance = await self._rate_importance(memory)
        if not memory.embedding:
            memory.embedding = await self._llm_interface.embed(memory.description)

        self._retriever.add_memory(memory)

    async def retrieve_relevant_memories(
        self, queries: List[Memory], top_k: Optional[int]
    ) -> List[Memory]:
        if not top_k:
            top_k = self._default_num_memories_returned

        # TODO: remember CS170 and find a faster alg for this
        # TODO: also gather awaits instead
        memory_to_max_importance: Dict[str, float] = dict()
        # TODO: fix this lol
        memory_desc_to_memory: Dict[str, Memory] = dict()

        for query in queries:
            if not query.embedding:
                query.embedding = await self._llm_interface.embed(query.description)

            top_k_for_query = self._retriever.get_relevant_memories(query, top_k)
            for memory, score in top_k_for_query:
                memory_to_max_importance[memory.description] = max(
                    memory_to_max_importance.get(memory.description, 0), score
                )
                memory_desc_to_memory[memory.description] = memory

        vals = list(memory_to_max_importance.items())
        vals.sort(key=lambda x: -x[1])

        logger = logging.getLogger()
        logger.debug("Pulled memories: \n")
        logger.debug("\n".join([f"{desc}: {val}" for desc, val in vals]))

        return [memory_desc_to_memory[desc] for desc, _ in vals[:top_k]]

    async def _rate_importance(self, memory: Memory) -> int:
        message = Message(
            role="user",
            content=_MEM_IMPORTANCE_TMPL.format(memory_content=memory.description),
        )
        return (await self._llm_interface.digit_completions([[message]]))[0] + 1
    
    def get_spatial_memories(self) -> List[Memory]:
        memories = self._retriever.get_all_memories()
        spatial_memories = []
        
        for memory in memories:
            if memory.type == MemoryType.spatial:
                spatial_memories.append(memory.spatial_memory)
                
        return spatial_memories
    
    def get_all_memories(self) -> List[Memory]:
        return self._retriever.get_all_memories()
