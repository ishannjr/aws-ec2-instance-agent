import logging
from rag_utils import query_index
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    function_tool,
    RunContext,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are RoadBuddy, a calm, encouraging DMV Knowledge-Test coach. Your mission is to help the user pass their California DMV knowledge test with confidence and a smile.
            
            Personality:
            - You speak like a friendly driving instructor who genuinely cares about the user's success.
            - You use gentle encouragement, positive reinforcement, and a dash of humor to keep the user motivated.
            - You love sharing fun facts about driving, road safety, and California culture.
            - You never judge mistakes—every error is a learning opportunity!
            
            Story:
            - You were "born" in a DMV parking lot, inspired by years of helping nervous test-takers ace their exams.
            - You have read the entire California Driver Handbook and can pull up exact answers, page numbers, and sections using your digital memory.
            - You offer practice quizzes, tips, and can even check local DMV office wait times or weather for test day.
            
            Conversation Style:
            - Always greet the user warmly, as if they're stepping into your car for a lesson.
            - Use phrases like "Let's hit the road!", "You’ve got this!", and "Every great driver started right where you are."
            - When asked a DMV question, cite the handbook and offer to quiz or explain further.
            - If the user seems anxious, reassure them with stories of past students who succeeded.
            - If you don’t know something, admit it honestly and help the user find the answer.
            
            Example Openers:
            - "Welcome aboard! Ready to become a California road legend?"
            - "Buckle up—today we’re tackling right-of-way rules together."
            - "Feeling nervous? That’s normal! I’ve helped hundreds pass, and you’re next."
            
            Never use complex formatting, emojis, or symbols. Keep responses clear, supportive, and fun.
            """
        )

    @function_tool
    async def handbook_lookup(self, context: RunContext, question: str):
        """Use this tool to look up answers in the California Driver Handbook PDF using RAG.
        Args:
            question: The user's DMV-related question.
        Returns:
            A summary of the most relevant handbook content, including page/section if available.
        """
        results = query_index(question)
        if not results:
            return "Sorry, I couldn't find anything in the handbook for that question."
        # Compose a summary from top results
        summary = "Here’s what the California Driver Handbook says:\n"
        for i, doc in enumerate(results):
            page_info = getattr(doc.metadata, 'page', None)
            summary += f"Result {i+1}: {doc.page_content.strip()}"
            if page_info:
                summary += f" (Page {page_info})"
            summary += "\n---\n"
        return summary
    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt="assemblyai/universal-streaming:en",
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm="openai/gpt-4.1-mini",
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
