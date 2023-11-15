import asyncio
from typing import List, Optional, Tuple, Union

from pydantic import UUID4

from game.memory import GenAgentMemory
from game.prompt_helpers import (
    clean_response,
    generate_functions_from_actions,
    get_action_messages,
    get_chat_messages,
    get_decompose_plan_message,
    get_guardrail_query,
    get_interact_messages,
    get_query_messages,
    get_rate_function,
    rating_to_int,
    get_broad_plan_message,
    format_plan
)
from llm.base import LLMBase
from schema import ActionCompletion, Conversation, Knowledge, Memory, Message
from schema.memory import GameStage, MemoryType


class GenAgent:
    def __init__(
        self,
        knowledge: Knowledge,
        llm_interface: LLMBase,
        memory: GenAgentMemory,
    ):
        """Should never be called directly. Use create() instead."""
        self._llm_interface = llm_interface
        self._memory = memory
        self._conversation_history: List[Message] = []
        self._knowledge = knowledge
        self._conversation_context = Conversation()

    @classmethod
    async def create(
        cls, knowledge: Knowledge, llm_interface: LLMBase, memory: GenAgentMemory
    ):
        agent = cls(knowledge, llm_interface, memory)
        await agent._fill_memories()
        await agent.plan(timestamp=GameStage()) #call with initial timestamp
        return agent

    async def _fill_memories(self):
        initial_memories = [
            lore.memory
            for lore in self._knowledge.shared_lore
            if self._knowledge.agent_def.uuid in lore.known_by
        ]

        initial_memories += self._knowledge.agent_def.personal_lore

        awaitables = [self._memory.add_memory(memory) for memory in initial_memories]
        await asyncio.gather(*awaitables)

    @property
    def uuid(self) -> UUID4:
        return self._knowledge.agent_def.uuid

    @property
    def name(self) -> str:
        return self._knowledge.agent_def.name

    async def add_memory(self, memory: Memory) -> None:
        await self._memory.add_memory(memory)

    async def interact(
        self, message: Optional[str], current_stage: Optional[GameStage]
    ) -> Tuple[Union[Message, ActionCompletion], List[Message]]:
        if message:
            self._conversation_history.append(Message(role="user", content=message))

        memories = await self._queryMemories(message,current_stage)

        messages = get_interact_messages(
            self._knowledge,
            self._conversation_context,
            memories,
            self._conversation_history,
        )
        functions = generate_functions_from_actions(self._knowledge.agent_def.actions)
        completion = await self._llm_interface.completion(messages, functions)

        if isinstance(completion, Message):
            self._conversation_history.append(clean_response(self.name, completion))

        return completion, messages

    async def chat(self, message: str) -> Tuple[Message, List[Message]]:
        self._conversation_history.append(Message(role="user", content=message))

        memories = await self._queryMemories(message)

        messages = get_chat_messages(
            self._knowledge,
            self._conversation_context,
            memories,
            self._conversation_history,
        )

        completion = await self._llm_interface.chat_completion(messages)

        self._conversation_history.append(clean_response(self.name, completion))
        return completion, messages

    async def act(
        self, message: Optional[str]
    ) -> Tuple[Optional[ActionCompletion], List[Message]]:
        if message:
            self._conversation_history.append(Message(role="user", content=message))

        memories = await self._queryMemories(message)

        messages = get_action_messages(
            self._knowledge,
            self._conversation_context,
            memories,
            self._conversation_history,
        )
        functions = generate_functions_from_actions(self._knowledge.agent_def.actions)

        return (
            await self._llm_interface.action_completion(messages, functions),
            messages,
        )

    async def plan(self, timestamp:GameStage, max_depth:int=1):
        memories = await self._queryMemories(message=None,timestamp=timestamp)


        message = None
        if timestamp.minor == 0: # do broad plan after major is changed and minor set to zero
            message = get_broad_plan_message(self._knowledge,memories,timestamp)
        else:
            current_step, level = self._memory.get_current_plan_step(timestamp)   

            if level < max_depth:         
                broad_plan = ""
                for i, step in enumerate(self._memory.get_broad_plan()):
                    broad_plan += f"{i}) {step.description}"
                message = get_decompose_plan_message(broad_plan,current_step.value.description,self._knowledge,memories,timestamp)

        if message is not None:
            completion = await self._llm_interface.completion([message], [])
            plan_steps = format_plan(completion.content)

            for step in plan_steps:
                print(step)
                ts = GameStage.from_time(step["time"])
                desc = step["step"]
                
                await self.add_memory(Memory(description=desc,type=MemoryType.plan,timestamp=ts))


    async def query(self, queries: List[str]) -> List[int]:
        """Returns a numerical answer to queries into the Agent's
        thoughts and emotions. 1 = not at all, 5 = extremely
        Ex. How happy are you given this conversation? -> 3 (moderately)"""
        memories = await asyncio.gather(
            *[self._queryMemories(query) for query in queries]
        )

        query_messages = get_query_messages(
            self._knowledge,
            self._conversation_context,
            memories,
            self._conversation_history,
            queries,
        )

        functions = [get_rate_function()]

        awaitables = [
            self._llm_interface.action_completion(msgs, functions)
            for msgs in query_messages
        ]
        ratings = await asyncio.gather(*awaitables)

        return [rating_to_int(rating) for rating in ratings]

    async def guardrail(self, message: str) -> int:
        """Is `message` something that the LLM thinks the GenAgent might say?
        Useful for playable characters and not letting players say inappropriate or
        anachronistic things.
        """
        guardrail_query = get_guardrail_query(self._knowledge, message)

        query_messages = get_query_messages(
            self._knowledge,
            self._conversation_context,
            [[]],
            self._conversation_history,
            [guardrail_query],
        )

        functions = [get_rate_function()]
        completion = await self._llm_interface.action_completion(
            query_messages[0], functions
        )

        return rating_to_int(completion)

    def startConversation(
        self,
        conversation: Conversation,
        history: List[Message],
    ):
        self._conversation_context = conversation
        self._conversation_history = history

    def resetConversation(self):
        self._conversation_history = []

    # TODO: use setter?
    def updateKnowledge(self, knowledge: Knowledge):
        # TODO: this doesn't update their memories, but we also don't really
        # want to overwrite what exists. Not sure what to do here.
        self._knowledge = knowledge

    async def _queryMemories(
        self, message: Optional[str] = None, timestamp:Optional[GameStage]=None, max_memories: Optional[int] = None
    ):
        if not max_memories:
            max_memories = self._conversation_context.memories_to_include

        queries: List[Memory] = []
        if message:
            queries.append(Memory(description=message,timestamp=timestamp))
        context_description = (self._conversation_context.scene_description or "") + (
            self._conversation_context.instructions or ""
        )
        if context_description:
            queries.append(Memory(description=context_description,timestamp=timestamp))

        return [
            memory.description
            for memory in await self._memory.retrieve_relevant_memories(
                queries, top_k=max_memories
            )
        ]
