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
       
        summary = "Here’s what the California Driver Handbook says:\n"
        for i, doc in enumerate(results):
            page_info = getattr(doc.metadata, 'page', None)
            summary += f"Result {i+1}: {doc.page_content.strip()}"
            if page_info:
                summary += f" (Page {page_info})"
            summary += "\n---\n"
        return summary



def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
   
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
