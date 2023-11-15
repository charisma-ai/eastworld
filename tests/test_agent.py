import uuid
from typing import Any
from unittest.mock import AsyncMock

from game.agent import Conversation, GenAgent, Knowledge
from game.prompt_helpers import (
    generate_functions_from_actions,
    get_action_messages,
    get_chat_messages,
    get_interact_messages,
    get_broad_plan_message
)
from schema import (
    Action,
    ActionCompletion,
    AgentDef,
    GameStage,
    Lore,
    Memory,
    Message,
    Parameter,
)
from schema.memory import MemoryType
from tests.helpers import AsyncCopyingMock


def create_agent_def() -> AgentDef:
    move_ps = [
        Parameter(name="x", description="x coordinate to move to", type="number"),
        Parameter(name="y", description="y coordinate to move to", type="number"),
    ]

    move = Action(
        name="move", description="ends conversation and moves", parameters=move_ps
    )

    attack_ps = [
        Parameter(
            name="character",
            description="Character to attack",
            type="string",
            enum=["Player", "Serf", "Noble"],
        )
    ]

    attack = Action(
        name="attack", description="attacks a character", parameters=attack_ps
    )

    agent_def = AgentDef(
        uuid=uuid.uuid4(),
        name="King",
        description="King is King of all lands and all people.",
        actions=[move, attack],
    )
    return agent_def


async def test_add_memory():
    memory: Any = AsyncMock()
    llm: Any = AsyncMock()
    agent_def = create_agent_def()

    lore1 = Lore(
        memory=Memory(
            description="I usurped the previous king.",
        ),
        known_by={agent_def.uuid},
    )
    lore2 = Lore(
        memory=Memory(
            description="I usurped the previous king.",
        ),
        known_by=set(),
    )
    knowledge = Knowledge(
        game_description="Game description",
        agent_def=agent_def,
        shared_lore=[lore1, lore2],
    )

    agent = await GenAgent.create(knowledge, llm, memory)
    assert memory.add_memory.call_count == 1
    memory.add_memory.assert_called_with(lore1.memory)

    simple_memory = Memory(
        importance=0,
        description="I was coronated as King!",
        timestamp=GameStage(stage=1, major=2, minor=3),
        embedding=[],
    )

    await agent.add_memory(simple_memory)
    memory.add_memory.assert_called_with(simple_memory)
    assert memory.add_memory.call_count == 2


async def test_interact():
    memory: Any = AsyncMock()
    llm: Any = AsyncMock()
    # TODO: this is here because the assert() compares the reference, which gets mutated
    # after; Can we remove this?
    llm.completion = AsyncCopyingMock()

    agent_def = create_agent_def()

    knowledge = Knowledge(
        game_description="Game description", agent_def=agent_def, shared_lore=[]
    )
    agent = await GenAgent.create(knowledge, llm, memory)
    llm.completion.return_value = Message(
        role="assistant", content="Not much, peasant!"
    )

    memories = [Memory(description="asdf")]
    memory.retrieve_relevant_memories.return_value = memories
    await agent.interact("What's up?",current_stage=GameStage())

    llm.completion.assert_called_with(
        get_interact_messages(
            knowledge,
            Conversation(),
            [m.description for m in memories],
            [Message(role="user", content="What's up?")],
        ),
        generate_functions_from_actions(agent_def.actions),
    )

    await agent.interact("Wtf?",current_stage=GameStage())

    llm.completion.assert_any_call(
        get_interact_messages(
            knowledge,
            Conversation(),
            [memory.description for memory in memories],
            [
                Message(role="user", content="What's up?"),
                Message(role="assistant", content="Not much, peasant!"),
                Message(role="user", content="Wtf?"),
            ],
        ),
        generate_functions_from_actions(agent_def.actions),
    )

    llm.completion.return_value = ActionCompletion(
        action="attack", args={"character": "Player"}
    )

    resp, _ = await agent.interact("I will kill you!",current_stage=GameStage())
    assert isinstance(resp, ActionCompletion)
    assert resp.action == "attack"
    assert resp.args["character"] == "Player"


async def test_chat():
    memory: Any = AsyncMock()
    llm: Any = AsyncMock()
    # TODO: this is here because the assert() compares the reference, which gets mutated
    # after; Can we remove this?
    llm.chat_completion = AsyncCopyingMock()

    agent_def = create_agent_def()

    knowledge = Knowledge(
        game_description="Game description", agent_def=agent_def, shared_lore=[]
    )
    agent = await GenAgent.create(knowledge, llm, memory)
    llm.chat_completion.return_value = Message(
        role="assistant", content="Not much, peasant!"
    )
    memories = [Memory(description="asdf")]
    memory.retrieve_relevant_memories.return_value = memories
    await agent.chat("What's up?")

    llm.chat_completion.assert_called_once_with(
        get_chat_messages(
            knowledge,
            Conversation(),
            [memory.description for memory in memories],
            [Message(role="user", content="What's up?")],
        ),
    )

    llm.chat_completion.return_value = Message(
        role="assistant", content="You heard me."
    )
    await agent.chat("Wtf?")

    llm.chat_completion.assert_any_call(
        get_chat_messages(
            knowledge,
            Conversation(),
            [memory.description for memory in memories],
            [
                Message(role="user", content="What's up?"),
                Message(role="assistant", content="Not much, peasant!"),
                Message(role="user", content="Wtf?"),
            ],
        )
    )

    conversation = Conversation(scene_description="Hello!")
    agent.startConversation(conversation, [])
    await agent.chat("New Conversation")

    llm.chat_completion.assert_called_with(
        get_chat_messages(
            knowledge,
            conversation.copy(),
            [memory.description for memory in memories],
            [Message(role="user", content="New Conversation")],
        )
    )


async def test_act():
    memory: Any = AsyncMock()
    llm: Any = AsyncMock()
    # TODO: this is here because the assert() compares the reference, which gets mutated
    # after; Can we remove this?
    llm.action_completion = AsyncCopyingMock()

    agent_def = create_agent_def()

    knowledge = Knowledge(
        game_description="Game description", agent_def=agent_def, shared_lore=[]
    )
    agent = await GenAgent.create(knowledge, llm, memory)

    llm.action_completion.return_value = ActionCompletion(
        action="attack", args={"character": "Player"}
    )

    memories = [Memory(description="asdf")]
    memory.retrieve_relevant_memories.return_value = memories

    resp, _ = await agent.act("I will usurp your throne!")

    llm.action_completion.assert_called_once_with(
        get_action_messages(
            knowledge,
            Conversation(),
            [memory.description for memory in memories],
            [
                Message(role="user", content="I will usurp your throne!"),
            ],
        ),
        generate_functions_from_actions(agent_def.actions),
    )

    assert isinstance(resp, ActionCompletion)
    assert resp.action == "attack"
    assert resp.args["character"] == "Player"


async def test_plan():
    memory: Any = AsyncMock()
    llm: Any = AsyncMock()
    # TODO: this is here because the assert() compares the reference, which gets mutated
    # after; Can we remove this?
    llm.completion = AsyncCopyingMock()
    memory.add_memory = AsyncCopyingMock()

    agent_def = create_agent_def()

    knowledge = Knowledge(
        game_description="Game description", agent_def=agent_def, shared_lore=[]
    )
    agent = await GenAgent.create(knowledge, llm, memory)
    
    mock_completion = """
1) {"time": "10:00am", "step": "go to Oak Hill College to take classes"}
2) {"time": "1:00pm", "step": "work on his new music composition"}
3) {"time": "5:30am", "step": "have dinner"}
4) {"time": "11:00pm", "step": "finish school assignments and go to bed"}
"""

    llm.completion.return_value = Message(
        role="assistant", content=mock_completion
    )
 
    await agent.plan(GameStage())

    llm.completion.assert_called_with(
        [get_broad_plan_message(knowledge,[],GameStage())], 
        []             
    ) 

    memory.add_memory.assert_called_with(
        Memory(description="finish school assignments and go to bed",type=MemoryType.plan,timestamp=GameStage.from_time("11:00pm"))      
    )