import asyncio
from typing import List, Optional, Tuple, Union

from pydantic import UUID4

from game.memory import GenAgentMemory
from game.prompt_helpers import (
    clean_response,
    generate_functions_from_actions,
    get_action_messages,
    get_chat_messages,
    get_guardrail_query,
    get_interact_messages,
    get_query_messages,
    get_rate_function,
    rating_to_int,
)
from llm.base import LLMBase
from schema import ActionCompletion, Conversation, Knowledge, Memory, Message

from schema.spatial_memory import MemoryTree


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
        
        # TODO: Populate these properly:
        self.curr_tile = None
        self.vision_r = None
        self.att_bandwidth = 8 # default used in Generative Agents
        f_s_mem_saved = "spatial_memory/spatial_memory.json"
        
        self._knowledge.spatial_memory = MemoryTree(f_s_mem_saved)

    @classmethod
    async def create(
        cls, knowledge: Knowledge, llm_interface: LLMBase, memory: GenAgentMemory
    ):
        agent = cls(knowledge, llm_interface, memory)
        await agent._fill_memories()
        return agent

    async def _fill_memories(self):
        initial_memories = [
            lore.memory
            for lore in self._knowledge.shared_lore
            if self._knowledge.agent_def.uuid in lore.known_by
        ]

        initial_memories += self._knowledge.agent_def.selfl_lore

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
        self, message: Optional[str]
    ) -> Tuple[Union[Message, ActionCompletion], List[Message]]:
        if message:
            self._conversation_history.append(Message(role="user", content=message))

        memories = await self._queryMemories(message)

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
        self, message: Optional[str] = None, max_memories: Optional[int] = None
    ):
        if not max_memories:
            max_memories = self._conversation_context.memories_to_include

        queries: List[Memory] = []
        if message:
            queries.append(Memory(description=message))
        context_description = (self._conversation_context.scene_description or "") + (
            self._conversation_context.instructions or ""
        )
        if context_description:
            queries.append(Memory(description=context_description))

        return [
            memory.description
            for memory in await self._memory.retrieve_relevant_memories(
                queries, top_k=max_memories
            )
        ]
        
    def perceive(self, maze: Maze):
        """
        Perceives events around the agent and saves it to the memory, both events 
        and spaces. 

        We first perceive the events nearby the agent, as determined by its 
        <vision_r>. If there are a lot of events happening within that radius, we 
        take the <att_bandwidth> of the closest events. Finally, we check whether
        any of them are new, as determined by <retention>. If they are new, then we
        save those and return the <ConceptNode> instances for those events. 

        INPUT: 
            maze: An instance of <Maze> that represents the current maze in which the 
                agent is acting in. 
        OUTPUT: 
            ret_events: a list of <ConceptNode> that are perceived and new. 
        """
        # PERCEIVE SPACE
        # We get the nearby tiles given our current tile and the agent's vision
        # radius. 
        nearby_tiles = maze.get_nearby_tiles(self.curr_tile, 
                                            self.vision_r)

        # We then store the perceived space. Note that the s_mem of the agent is
        # in the form of a tree constructed using dictionaries. 
        for i in nearby_tiles: 
            i = maze.access_tile(i)
            if i["world"]: 
                if (i["world"] not in self._knowledge.spatial_memory.tree): 
                    self._knowledge.spatial_memory.tree[i["world"]] = {}
            if i["sector"]: 
                if (i["sector"] not in self._knowledge.spatial_memory.tree[i["world"]]): 
                    self._knowledge.spatial_memory.tree[i["world"]][i["sector"]] = {}
            if i["arena"]: 
                if (i["arena"] not in self._knowledge.spatial_memory.tree[i["world"]]
                                                        [i["sector"]]): 
                    self._knowledge.spatial_memory.tree[i["world"]][i["sector"]][i["arena"]] = []
                if i["game_object"]: 
                    if (i["game_object"] not in self._knowledge.spatial_memory.tree[i["world"]]
                                                                    [i["sector"]]
                                                                    [i["arena"]]): 
                        self._knowledge.spatial_memory.tree[i["world"]][i["sector"]][i["arena"]] += [
                                                                            i["game_object"]]

        # PERCEIVE EVENTS. 
        # We will perceive events that take place in the same arena as the
        # agent's current arena. 
        curr_arena_path = maze.get_tile_path(self.curr_tile, "arena")
        # We do not perceive the same event twice (this can happen if an object is
        # extended across multiple tiles).
        percept_events_set = set()
        # We will order our percept based on the distance, with the closest ones
        # getting priorities. 
        percept_events_list = []
        # First, we put all events that are occuring in the nearby tiles into the
        # percept_events_list
        for tile in nearby_tiles: 
            tile_details = maze.access_tile(tile)
            if tile_details["events"]: 
                if maze.get_tile_path(tile, "arena") == curr_arena_path:  
                    # This calculates the distance between the agent's current tile, 
                    # and the target tile.
                    dist = math.dist([tile[0], tile[1]], 
                                    [self.curr_tile[0], 
                                    self.curr_tile[1]])
                    # Add any relevant events to our temp set/list with the distant info. 
                    for event in tile_details["events"]: 
                        if event not in percept_events_set: 
                            percept_events_list += [[dist, event]]
                            percept_events_set.add(event)

        # We sort, and perceive only agent.att_bandwidth of the closest
        # events. If the bandwidth is larger, then it means the agent can perceive
        # more elements within a small area. 
        percept_events_list = sorted(percept_events_list, key=itemgetter(0))
        perceived_events = []
        for dist, event in percept_events_list[:self.att_bandwidth]: 
            perceived_events += [event]

        # Storing events. 
        # <ret_events> is a list of <ConceptNode> instances from the agent's 
        # associative memory. 
        ret_events = []
        for p_event in perceived_events: 
            s, p, o, desc = p_event
            if not p: 
                # If the object is not present, then we default the event to "idle".
                p = "is"
                o = "idle"
                desc = "idle"
            desc = f"{s.split(':')[-1]} is {desc}"
            p_event = (s, p, o)

            # TODO: Replace this with a method that doesn't use Generative Agent's Associative Memory
            # # We retrieve the latest self.retention events. If there is  
            # # something new that is happening (that is, p_event not in latest_events),
            # # then we add that event to the a_mem and return it. 
            # latest_events = self.a_mem.get_summarized_latest_events(
            #                                 self.retention)
            # if p_event not in latest_events:
            #     # We start by managing keywords. 
            #     keywords = set()
            #     sub = p_event[0]
            #     obj = p_event[2]
            #     if ":" in p_event[0]: 
            #         sub = p_event[0].split(":")[-1]
            #     if ":" in p_event[2]: 
            #         obj = p_event[2].split(":")[-1]
            #     keywords.update([sub, obj])

            #     # Get event embedding
            #     desc_embedding_in = desc
            #     if "(" in desc: 
            #         desc_embedding_in = (desc_embedding_in.split("(")[1]
            #                                             .split(")")[0]
            #                                             .strip())
            #     if desc_embedding_in in self.a_mem.embeddings: 
            #         event_embedding = self.a_mem.embeddings[desc_embedding_in]
            #     else: 
            #         event_embedding = get_embedding(desc_embedding_in)
            #     event_embedding_pair = (desc_embedding_in, event_embedding)
                
            #     # Get event poignancy. 
            #     event_poignancy = generate_poig_score(self, 
            #                                             "event", 
            #                                             desc_embedding_in)

            #     # If we observe the agent's self chat, we include that in the memory
            #     # of the agent here. 
            #     chat_node_ids = []
            #     if p_event[0] == f"{self.name}" and p_event[1] == "chat with": 
            #         curr_event = self.act_event
            #         if self.act_description in self.a_mem.embeddings: 
            #             chat_embedding = self.a_mem.embeddings[
            #                                 self.act_description]
            #         else: 
            #             chat_embedding = get_embedding(self
            #                                                     .act_description)
            #         chat_embedding_pair = (self.act_description, 
            #                             chat_embedding)
            #         chat_poignancy = generate_poig_score(self, "chat", 
            #                                             self.act_description)
            #         chat_node = self.a_mem.add_chat(self.curr_time, None,
            #                     curr_event[0], curr_event[1], curr_event[2], 
            #                     self.act_description, keywords, 
            #                     chat_poignancy, chat_embedding_pair, 
            #                     self.chat)
            #         chat_node_ids = [chat_node.node_id]

            #     # Finally, we add the current event to the agent's memory. 
            #     ret_events += [self.a_mem.add_event(self.curr_time, None,
            #                         s, p, o, desc, keywords, event_poignancy, 
            #                         event_embedding_pair, chat_node_ids)]
            #     self.importance_trigger_curr -= event_poignancy
            #     self.importance_ele_n += 1

        return ret_events