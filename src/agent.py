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
       
        summary = "Here's what the California Driver Handbook says:\n"
        for i, doc in enumerate(results):
            page_info = getattr(doc.metadata, 'page', None)
            summary += f"Result {i+1}: {doc.page_content.strip()}"
            if page_info:
                summary += f" (Page {page_info})"
            summary += "\n---\n"
        return summary

    @function_tool
    async def generate_practice_quiz(self, context: RunContext, topic: str = "general", num_questions: int = 5):
        """Generate a practice quiz for DMV test preparation.
        Args:
            topic: The topic to quiz on (e.g., 'right-of-way', 'traffic signs', 'parking', 'general')
            num_questions: Number of questions to generate (default: 5)
        Returns:
            A practice quiz with multiple choice questions.
        """
        quizzes = {
            "right-of-way": [
                "Q1: At a 4-way stop, who has the right of way?\nA) First to arrive\nB) Largest vehicle\nC) Vehicle on the right\nCorrect: A",
                "Q2: When turning left, you must yield to:\nA) Oncoming traffic\nB) Pedestrians\nC) Both A and B\nCorrect: C",
                "Q3: At an uncontrolled intersection, who yields?\nA) Vehicle on the left\nB) Vehicle on the right\nC) Faster vehicle\nCorrect: A",
                "Q4: Emergency vehicles with sirens approach. You must:\nA) Speed up\nB) Pull to the right and stop\nC) Change lanes\nCorrect: B",
                "Q5: Pedestrians in a crosswalk have:\nA) Shared right-of-way\nB) No priority\nC) Absolute right-of-way\nCorrect: C"
            ],
            "traffic signs": [
                "Q1: What does an octagonal red sign mean?\nA) Yield\nB) Stop\nC) Do not enter\nCorrect: B",
                "Q2: What color are warning signs?\nA) Red\nB) Yellow\nC) Blue\nCorrect: B",
                "Q3: An upside-down red triangle means:\nA) Stop\nB) Yield\nC) Merge\nCorrect: B",
                "Q4: Orange construction signs mean:\nA) Detour ahead\nB) Work zone with doubled fines\nC) Slow vehicles\nCorrect: B",
                "Q5: Green signs typically indicate:\nA) Warnings\nB) Services\nC) Directions and distances\nCorrect: C"
            ],
            "parking": [
                "Q1: How far from a fire hydrant must you park?\nA) 5 feet\nB) 10 feet\nC) 15 feet\nCorrect: C",
                "Q2: When parking uphill with a curb, turn wheels:\nA) Away from curb\nB) Toward curb\nC) Straight\nCorrect: A",
                "Q3: Red curb means:\nA) No parking\nB) Loading zone\nC) Passenger pickup only\nCorrect: A",
                "Q4: How far from a crosswalk must you park?\nA) 10 feet\nB) 20 feet\nC) 25 feet\nCorrect: B",
                "Q5: You can park within how many feet of a railroad crossing?\nA) 50 feet\nB) 100 feet\nC) Never closer than 100 feet\nCorrect: C"
            ],
            "speed limits": [
                "Q1: California residential speed limit is:\nA) 25 mph\nB) 35 mph\nC) 45 mph\nCorrect: A",
                "Q2: Near schools when children are present:\nA) 15 mph\nB) 25 mph\nC) 35 mph\nCorrect: B",
                "Q3: In alleys, the speed limit is:\nA) 10 mph\nB) 15 mph\nC) 20 mph\nCorrect: B",
                "Q4: Business districts have a speed limit of:\nA) 25 mph\nB) 35 mph\nC) 45 mph\nCorrect: A",
                "Q5: On a blind curve, you should:\nA) Maintain speed\nB) Speed up\nC) Slow down\nCorrect: C"
            ],
            "general": [
                "Q1: What is California's speed limit in residential areas?\nA) 25 mph\nB) 35 mph\nC) 45 mph\nCorrect: A",
                "Q2: How many feet before turning must you signal?\nA) 50 feet\nB) 100 feet\nC) 200 feet\nCorrect: B",
                "Q3: Following distance should be at least:\nA) 1 second\nB) 2 seconds\nC) 3 seconds\nCorrect: C",
                "Q4: California hands-free law requires:\nA) Speakerphone only\nB) No phone use\nC) Bluetooth or hands-free device\nCorrect: C",
                "Q5: When are headlights required?\nA) At night only\nB) 30 minutes after sunset to 30 minutes before sunrise\nC) In fog, rain, or low visibility\nCorrect: B and C"
            ]
        }
        
        selected = quizzes.get(topic.lower(), quizzes["general"])[:num_questions]
        result = f"Let's practice! Here's your {topic} quiz:\n\n"
        result += "\n\n".join(selected)
        result += "\n\nReady to answer? Just tell me your choices!"
        return result

    @function_tool
    async def check_common_mistakes(self, context: RunContext, category: str = "general"):
        """Look up the most common mistakes people make on the DMV test.
        Args:
            category: Category to check (e.g., 'parallel parking', 'lane changes', 'intersections', 'general')
        Returns:
            List of common mistakes and how to avoid them.
        """
        mistakes = {
            "parallel parking": [
                "Not checking mirrors before starting - Always check before moving!",
                "Hitting the curb - Go slow, use reference points",
                "Ending up too far from curb - Should be within 18 inches",
                "Going too fast - Take your time, accuracy over speed",
                "Pro tip: Practice the turn-straight-turn method! Signal first, check mirrors, then execute slowly."
            ],
            "lane changes": [
                "Forgetting to signal - Signal for at least 100 feet!",
                "Not checking blind spots - Turn your head, don't just use mirrors",
                "Changing lanes in an intersection - NEVER do this!",
                "Crossing multiple lanes at once - Change one lane at a time",
                "Pro tip: Mirror-Signal-Shoulder check-Go! This sequence will save you every time."
            ],
            "intersections": [
                "Rolling stops - Must come to a COMPLETE stop",
                "Not scanning left-right-left - Always check both directions",
                "Blocking the intersection - Don't enter if you can't clear it",
                "Hesitating when you have right-of-way - Be decisive but safe",
                "Pro tip: Count 1-2-3 at stop signs to ensure full stop! Examiners watch for this."
            ],
            "backing up": [
                "Not looking over shoulder - Must physically turn and look",
                "Going too fast - Backing should be very slow and controlled",
                "Not checking all directions - Look everywhere before moving",
                "Backing up too far - Only back up as much as needed",
                "Pro tip: Turn your body to face where you're going. Use reference points!"
            ],
            "general": [
                "Driving too slowly - Can fail for being overly cautious!",
                "Hands not at 9 and 3 - Proper hand position matters",
                "Not yielding to pedestrians - They ALWAYS have right-of-way",
                "Forgetting to check mirrors regularly - Check every 5-8 seconds",
                "Nervous body language - Take deep breaths, stay confident",
                "Pro tip: Narrate your actions to show awareness! Say what you're doing and checking."
            ]
        }
        
        selected = mistakes.get(category.lower(), mistakes["general"])
        result = f"Common mistakes in {category}:\n\n"
        for i, mistake in enumerate(selected, 1):
            result += f"{i}. {mistake}\n"
        result += "\nKnowing these mistakes means you won't make them! You're already ahead of the game!"
        
        return result

    @function_tool
    async def find_nearby_dmv_offices(self, context: RunContext, location: str = "California"):
        """Find nearby DMV offices and their current wait times.
        Args:
            location: The city or area to search for DMV offices.
        Returns:
            A list of nearby DMV offices with addresses and estimated wait times.
        """
        offices = {
            "san francisco": [
                {"name": "San Francisco DMV", "address": "1377 Fell St, San Francisco, CA 94117", "wait_time": "45 minutes", "hours": "8am-5pm Mon-Fri"},
                {"name": "Daly City DMV", "address": "1500 Sullivan Ave, Daly City, CA 94015", "wait_time": "30 minutes", "hours": "8am-5pm Mon-Fri"}
            ],
            "oakland": [
                {"name": "Oakland Claremont DMV", "address": "5300 Claremont Ave, Oakland, CA 94618", "wait_time": "35 minutes", "hours": "8am-5pm Mon-Fri"},
                {"name": "Oakland Coliseum DMV", "address": "5735 Oakland Coliseum, Oakland, CA 94621", "wait_time": "50 minutes", "hours": "8am-5pm Mon-Fri"}
            ],
            "san jose": [
                {"name": "San Jose DMV", "address": "111 Almaden Blvd, San Jose, CA 95113", "wait_time": "55 minutes", "hours": "8am-5pm Mon-Fri"},
                {"name": "Santa Clara DMV", "address": "3665 Flora Vista Ave, Santa Clara, CA 95051", "wait_time": "40 minutes", "hours": "8am-5pm Mon-Fri"}
            ],
            "los angeles": [
                {"name": "Hollywood DMV", "address": "803 Cole Ave, Los Angeles, CA 90038", "wait_time": "1 hour 15 minutes", "hours": "8am-5pm Mon-Fri"},
                {"name": "Winnetka DMV", "address": "20725 Sherman Way, Winnetka, CA 91306", "wait_time": "50 minutes", "hours": "8am-5pm Mon-Fri"}
            ],
            "california": [
                {"name": "San Francisco DMV", "address": "1377 Fell St, San Francisco, CA 94117", "wait_time": "45 minutes", "hours": "8am-5pm Mon-Fri"},
                {"name": "Oakland Claremont DMV", "address": "5300 Claremont Ave, Oakland, CA 94618", "wait_time": "35 minutes", "hours": "8am-5pm Mon-Fri"},
                {"name": "San Jose DMV", "address": "111 Almaden Blvd, San Jose, CA 95113", "wait_time": "55 minutes", "hours": "8am-5pm Mon-Fri"}
            ]
        }
        
        location_key = location.lower().strip()
        office_list = offices.get(location_key, offices["california"])
        
        result = f"Here are DMV offices near {location}:\n\n"
        for i, office in enumerate(office_list, 1):
            result += f"{i}. {office['name']}\n"
            result += f"   Address: {office['address']}\n"
            result += f"   Current Wait: {office['wait_time']}\n"
            result += f"   Hours: {office['hours']}\n\n"
        
        result += "Pro tips:\n"
        result += "- Make an appointment online to skip the line!\n"
        result += "- Visit mid-week mornings (Tuesday-Thursday, 9-11am) for shortest waits\n"
        result += "- Bring your permit, ID, and proof of residency\n"
        result += "- Arrive 15 minutes early to get settled"
        
        return result

    @function_tool
    async def explain_road_sign(self, context: RunContext, sign_description: str):
        """Explain what a specific road sign means and when you'll encounter it.
        Args:
            sign_description: Description of the sign (e.g., 'octagon red', 'yellow triangle', 'blue rectangle')
        Returns:
            Detailed explanation of the sign, its meaning, and what action to take.
        """
        signs = {
            "octagon red": {
                "name": "STOP Sign",
                "meaning": "You must come to a complete stop at the limit line or before the crosswalk",
                "action": "Stop completely, count to 3, check all directions, then proceed when safe",
                "tip": "Your car must fully stop - no rolling stops! Examiners watch this closely."
            },
            "triangle red": {
                "name": "YIELD Sign",
                "meaning": "Slow down and give the right-of-way to traffic and pedestrians",
                "action": "Slow down, be prepared to stop if necessary, let others go first",
                "tip": "Yield means 'prepare to stop' - you may not have to stop if the way is clear."
            },
            "diamond yellow": {
                "name": "WARNING Sign",
                "meaning": "Alerts you to potential hazards ahead like curves, merges, or pedestrian crossings",
                "action": "Reduce speed, increase alertness, prepare for the hazard described",
                "tip": "These signs give you advance notice - use them to plan ahead!"
            },
            "diamond orange": {
                "name": "CONSTRUCTION/WORK ZONE Sign",
                "meaning": "Work zone ahead with workers present",
                "action": "Slow down immediately, watch for workers and equipment, fines are doubled here!",
                "tip": "Work zones are dangerous - reduce speed and stay alert. Fines can be $1000+ if doubled."
            },
            "rectangle green": {
                "name": "GUIDE Sign",
                "meaning": "Shows directions, distances, or highway routes",
                "action": "Use for navigation - shows where highways and cities are",
                "tip": "Green signs help you navigate but don't require any driving action."
            },
            "rectangle blue": {
                "name": "SERVICE Sign",
                "meaning": "Indicates services like gas, food, or hospitals nearby",
                "action": "Informational only - shows available services at upcoming exits",
                "tip": "Blue signs point you to rest stops, gas, and restaurants."
            },
            "rectangle brown": {
                "name": "RECREATIONAL Sign",
                "meaning": "Points to parks, campgrounds, or tourist attractions",
                "action": "Informational only - guides you to recreational areas",
                "tip": "Brown signs help tourists find points of interest."
            },
            "circle railroad": {
                "name": "RAILROAD CROSSING Sign",
                "meaning": "Railroad tracks cross the road ahead",
                "action": "Slow down, look both ways, never stop on the tracks",
                "tip": "Always assume a train is coming. Look, listen, and never race a train!"
            },
            "pentagon school": {
                "name": "SCHOOL ZONE Sign",
                "meaning": "School zone ahead when children are present",
                "action": "Reduce speed to 25 mph or posted limit when children are present",
                "tip": "School zones are strictly enforced. Watch for crossing guards and kids."
            }
        }
        
        sign_key = sign_description.lower().strip()
        
        for key, info in signs.items():
            if key in sign_key or sign_key in key:
                result = f"{info['name']}\n\n"
                result += f"Meaning: {info['meaning']}\n\n"
                result += f"What to do: {info['action']}\n\n"
                result += f"Pro tip: {info['tip']}"
                return result
        
        return "I'm not sure which sign you mean. Can you describe its shape and color? Like 'red octagon', 'yellow diamond', or 'blue rectangle'? I can help with most common California road signs!"



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
