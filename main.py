import os
import re
import base64
import binascii

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Multimodal Image Question-Answering API",
    description=(
        "API for answering questions about charts, receipts, "
        "invoices, tables, and scanned documents."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ImageQuestionRequest(BaseModel):
    image_base64: str = Field(
        ...,
        description="Base64-encoded image"
    )

    question: str = Field(
        ...,
        min_length=1,
        description="Question about the image"
    )


# ============================================================
# HELPER: CLEAN BASE64 AND DETECT IMAGE TYPE
# ============================================================

def process_base64_image(image_base64: str):
    """
    Accepts either:

    1. Raw Base64:
       iVBORw0KGgoAAA...

    2. Data URL:
       data:image/png;base64,iVBORw0KGgoAAA...

    Returns:
        image_bytes, mime_type
    """

    image_base64 = image_base64.strip()

    # Default MIME type
    mime_type = "image/png"

    # --------------------------------------------------------
    # Handle data URL format
    # --------------------------------------------------------

    if image_base64.startswith("data:"):
        match = re.match(
            r"data:([^;]+);base64,(.*)",
            image_base64,
            flags=re.DOTALL,
        )

        if not match:
            raise ValueError(
                "Invalid Base64 data URL."
            )

        mime_type = match.group(1)
        image_base64 = match.group(2)

    # --------------------------------------------------------
    # Remove spaces and newlines from Base64
    # --------------------------------------------------------

    image_base64 = re.sub(
        r"\s+",
        "",
        image_base64
    )

    # --------------------------------------------------------
    # Decode Base64
    # --------------------------------------------------------

    try:
        image_bytes = base64.b64decode(
            image_base64,
            validate=True
        )

    except (binascii.Error, ValueError) as error:
        raise ValueError(
            "image_base64 is not valid Base64."
        ) from error

    if not image_bytes:
        raise ValueError(
            "Decoded image is empty."
        )

    # --------------------------------------------------------
    # Detect common image formats
    # --------------------------------------------------------

    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        mime_type = "image/png"

    elif image_bytes.startswith(b"\xff\xd8\xff"):
        mime_type = "image/jpeg"

    elif (
        image_bytes.startswith(b"GIF87a")
        or image_bytes.startswith(b"GIF89a")
    ):
        mime_type = "image/gif"

    elif (
        image_bytes.startswith(b"RIFF")
        and b"WEBP" in image_bytes[:16]
    ):
        mime_type = "image/webp"

    return image_bytes, mime_type


# ============================================================
# HELPER: CLEAN MODEL ANSWER
# ============================================================

def clean_answer(answer: str) -> str:
    """
    Cleans accidental formatting from the model response.
    """

    answer = str(answer).strip()

    # Remove Markdown code fences
    answer = answer.replace("```json", "")
    answer = answer.replace("```text", "")
    answer = answer.replace("```", "")
    answer = answer.strip()

    # Remove common prefixes
    prefixes = [
        "final answer:",
        "answer:",
        "the answer is:",
    ]

    for prefix in prefixes:
        if answer.lower().startswith(prefix):
            answer = answer[len(prefix):].strip()
            break

    # Remove surrounding quotation marks
    if (
        len(answer) >= 2
        and answer[0] == answer[-1]
        and answer[0] in ["'", '"']
    ):
        answer = answer[1:-1].strip()

    return answer


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Multimodal Image QA API",
        "endpoint": "/answer-image",
        "model": MODEL_NAME,
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# MAIN IMAGE QA ENDPOINT
# ============================================================

@app.post("/answer-image")
def answer_image(request: ImageQuestionRequest):

    # --------------------------------------------------------
    # STEP 1: CHECK API KEY
    # --------------------------------------------------------

    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail=(
                "GEMINI_API_KEY environment variable "
                "is not configured."
            ),
        )

    # --------------------------------------------------------
    # STEP 2: DECODE IMAGE
    # --------------------------------------------------------

    try:
        image_bytes, mime_type = process_base64_image(
            request.image_base64
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    # --------------------------------------------------------
    # STEP 3: CREATE PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a highly accurate visual document question-answering system.

Carefully inspect the supplied image.

The image may contain:
- bar charts
- line charts
- pie charts
- tables
- receipts
- invoices
- scanned academic documents
- administrative documents
- labels
- numbers
- totals
- percentages

QUESTION:
{request.question}

OUTPUT RULES:

1. Answer using only information visible in the image.
2. Return ONLY the final answer.
3. Do not explain your reasoning.
4. Do not use Markdown.
5. Do not write "Answer:" or "The answer is".
6. Keep the answer concise.
7. If the requested answer is numeric, return only the number.
8. For numeric answers, do not include currency symbols.
9. For numeric answers, do not include units unless the question
   specifically requires the unit as part of the answer.
10. Carefully calculate totals, differences, averages, percentages,
    or other requested values when necessary.

Examples:

Question: What is the total?
Output:
4089.35

Question: How many students passed?
Output:
87

Question: Which month had the highest sales?
Output:
March

Now answer the question about the supplied image.
"""

    # --------------------------------------------------------
    # STEP 4: CALL GEMINI
    # --------------------------------------------------------

    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=mime_type,
                        ),
                        types.Part.from_text(
                            text=prompt
                        ),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=100,
            ),
        )

        # ----------------------------------------------------
        # STEP 5: GET RESPONSE TEXT
        # ----------------------------------------------------

        answer = response.text

        if not answer:
            raise HTTPException(
                status_code=502,
                detail=(
                    "The multimodal model returned "
                    "an empty answer."
                ),
            )

        # ----------------------------------------------------
        # STEP 6: CLEAN ANSWER
        # ----------------------------------------------------

        answer = clean_answer(answer)

        if not answer:
            raise HTTPException(
                status_code=502,
                detail=(
                    "The multimodal model returned "
                    "an empty answer after cleaning."
                ),
            )

        # ----------------------------------------------------
        # STEP 7: RETURN REQUIRED JSON
        # ----------------------------------------------------

        return {
            "answer": str(answer)
        }

    # --------------------------------------------------------
    # KEEP OUR OWN HTTP ERRORS
    # --------------------------------------------------------

    except HTTPException:
        raise

    # --------------------------------------------------------
    # CATCH GEMINI / NETWORK / MODEL ERRORS
    # --------------------------------------------------------

    except Exception as error:
        error_message = (
            f"{type(error).__name__}: {str(error)}"
        )

        # Show the real error in Render logs
        print(
            f"Gemini API error: {error_message}",
            flush=True
        )

        # TEMPORARILY return the real error for debugging.
        # After everything works, this can be replaced with
        # a generic error message.
        raise HTTPException(
            status_code=502,
            detail=error_message
        ) from error
