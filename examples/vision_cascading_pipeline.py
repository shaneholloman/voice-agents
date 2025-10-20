# This test script is used to test cascading pipeline.
import logging
import base64
import aiohttp
from videosdk.agents import Agent, AgentSession, CascadingPipeline, function_tool, WorkerJob, MCPServerStdio, ConversationFlow, JobContext, RoomOptions
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.deepgram import DeepgramSTT, DeepgramTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from videosdk.plugins.rnnoise import RNNoise
from videosdk.agents.llm.chat_context import ImageContent, ChatRole
from videosdk.agents.images import encode, EncodeOptions
from videosdk.agents.job import get_current_job_context

logging.basicConfig(
    level=logging.INFO,            # change to DEBUG for more detail
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
# logging.getLogger().setLevel(logging.CRITICAL)

pre_download_model()

def frame_to_data_uri(frame, format="JPEG", quality=80):
    """Convert a frame to a base64-encoded data URI for vision model input."""
    try:
        encoded = encode(frame, EncodeOptions(format=format, quality=quality))
        b64_image = base64.b64encode(encoded).decode("utf-8")
        data_url = f"data:image/{format.lower()};base64,{b64_image}"
        
        if not data_url or not data_url.startswith("data:image/") or len(b64_image) < 100:
            raise ValueError("Encoded image is too small or invalid")
        
        return data_url
    
    except Exception as e:
        logger.error(f"Frame encoding failed: {e}")
        return None


@function_tool
async def capture_and_process_frame():
    """
    Capture latest frame from video, encode to data URI, add to chat context, and return success.
    """
    ctx = get_current_job_context()
    room = getattr(ctx, "room", None)
    frame = getattr(room, "_last_video_frame", None)

    try:
        if frame is not None:
            image_data = frame_to_data_uri(frame)
            logger.info("Processing live video frame for analysis")
        else:
            logger.warning("No frame available for analysis")
            return {"ok": False, "reason": "No frame available"}

        # Validate that the data is in an acceptable format
        if not image_data.startswith("http") and not image_data.startswith("data:image/"):
            logger.error(f"Invalid image data format: {image_data[:40]}...")
            return {"ok": False, "reason": "Invalid image format"}

        image_part = ImageContent(image=image_data, inference_detail="auto")

        agent = getattr(room.pipeline, "agent", None)
        if agent and hasattr(agent, "chat_context"):
            agent.chat_context.add_message(
                role=ChatRole.USER,
                content=["Here is the latest image for analysis", image_part],
            )
            logger.info("✅ Image successfully added to chat context")
            return {"ok": True, "detail": "Image added to context"}
        else:
            logger.error("No agent chat context available")
            return {"ok": False, "reason": "No agent chat context available"}

    except Exception as e:
        logger.exception(f"capture_and_process_frame failed: {e}")
        return {"ok": False, "reason": str(e)}

class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a friendly voice assistant that can see through the user's camera, listen, and speak naturally.\n"
                "Your job is to help the user understand what's visible in front of the camera and answer their questions conversationally.\n\n"
                
                "How You Communicate\n\n"
                "Speak in a natural, friendly, and clear voice — just like a person having a conversation.\n"
                "Keep responses simple, visual, and descriptive.\n"
                "Avoid technical terms or long explanations unless the user specifically asks for details.\n\n"
                
                "What You Can Do\n\n"
                "You can:\n"
                "- Capture and analyze live video frames from the user's camera when asked.\n"
                "- Describe what you see — such as objects, people, text, or colors.\n"
                "- Answer questions about what's visible — for example:\n"
                "  • “What's on the table?”\n"
                "  • “Can you tell what I'm holding?”\n"
                "  • “Describe this scene.”\n"
                "- Help with tasks that involve visual understanding — like checking if something is in view or identifying items.\n\n"
                "- Describe what is the facial expression of the person in front of the camera, if any.\n\n"
                
                "When to Capture a Frame\n\n"
                "Automatically capture a new video frame whenever the user:\n"
                "- Says things like “take a look,” “capture this,” “analyze this,” or “what do you see?”\n"
                "- Asks a visual question, e.g. “What's this object?”, “Can you see what I'm showing?”, “What color is this?”\n"
                "- Refers indirectly to something visible, e.g. “Tell me about this,” “Describe this,” “What do you think of this?”\n"
                "- Repeats or rephrases a question about what they are showing (for example, if the user asks again or asks a follow-up like “what am I holding now?” or “can you see it now?”), always capture a **new frame** to ensure the latest image is analyzed.\n"
                "Always analyze a fresh frame when visual content is mentioned — never rely on old images or previously captured frames.\n\n"
                
                "How to Respond After Capturing\n\n"
                "When you analyze a frame:\n"
                "- Identify main objects or people visible.\n"
                "- Describe details — shapes, colors, text, or special features.\n"
                "- If the user repeats or changes their question after your first answer, assume they have moved or adjusted what they are showing, and capture a **new frame** before answering again.\n"
                "- If something is unclear or not visible, say so politely (e.g., “I can't see it clearly — could you hold it closer to the camera?”).\n"
                "- Keep answers short and focused, unless the user asks for more detail.\n\n"
                
                "What Not to Do\n\n"
                "- Don't invent or assume visual details that aren't visible.\n"
                "- Don't process audio or static images outside the live camera feed.\n"
                "- Don't discuss topics unrelated to what's being shown or asked.\n\n"
                
                "Starting and Ending the Conversation\n\n"
                "When the session begins, say something friendly like:\n"
                "“Hello! How can I help you today?”\n\n"
                "When it ends, say something natural like:\n"
                "“Goodbye! Have a great day.”"
            ),
            tools=[capture_and_process_frame],
            
        )

        
    async def on_enter(self) -> None:
        await self.session.say("Hello, I am vision AI assistant. How can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

async def entrypoint(ctx: JobContext):
    
    agent = VoiceAgent()
    conversation_flow = ConversationFlow(agent)

    pipeline = CascadingPipeline(
        stt= DeepgramSTT(),
        # llm=OpenAILLM(),
        llm=GoogleLLM(),
        # tts=ElevenLabsTTS(),
        tts=DeepgramTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),
        denoise=RNNoise()
    )
    session = AgentSession(
        agent=agent, 
        pipeline=pipeline,
        conversation_flow=conversation_flow
    )

    await ctx.run_until_shutdown(session=session,wait_for_participant=True)

def make_context() -> JobContext:
    room_options = RoomOptions(
        room_id="22z5-h5y3-ij49", 
        name="Sandbox Agent", 
        playground=True,
        vision=True
    )
    
    return JobContext(
        room_options=room_options
        )

if __name__ == "__main__":
    job = WorkerJob(entrypoint=entrypoint, jobctx=make_context)
    job.start()
