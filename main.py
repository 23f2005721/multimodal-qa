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

# You can change this through a Render environment variable
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
# Required so external graders / Cloudflare Workers can call
# the API.
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
# HELPER: CLEAN BASE64 AND DETECT MIME TYPE
# ============================================================

def process_base64_image(image_base64: str):

    image_base64 = image_base64.strip()

    mime_type = "image/png"

    # Support data URLs such as:
    # data:image/jpeg;base64,/9j/4AAQ...
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


    # Remove accidental whitespace/newlines
    image_base64 = re.sub(
        r"\s+",
        "",
        image_base64
    )


    # Decode Base64 into raw image bytes
    try:
        image_bytes = base64.b64decode(
            image_base64,
            validate=True
        )

    except (binascii.Error, ValueError):
        raise ValueError(
            "image_base64 is not valid Base64."
        )


    if not image_bytes:
        raise ValueError(
            "Decoded image is empty."
        )


    # Detect common image formats when no data URL was supplied
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        mime_type = "image/png"

    elif image_bytes.startswith(b"\xff\xd8\xff"):
        mime_type = "image/jpeg"

    elif image_bytes.startswith(b"GIF87a") or \
            image_bytes.startswith(b"GIF89a"):
        mime_type = "image/gif"

    elif image_bytes.startswith(b"RIFF") and \
            b"WEBP" in image_bytes[:16]:
        mime_type = "image/webp"


    return image_bytes, mime_type


# ============================================================
# HELPER: CLEAN MODEL ANSWER
# ============================================================

def clean_answer(answer: str) -> str:

    answer = str(answer).strip()

    # Remove markdown code fences
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

    lower_answer = answer.lower()

    for prefix in prefixes:

        if lower_answer.startswith(prefix):

            answer = answer[len(prefix):].strip()
            break


    # Remove surrounding quotes
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
    # Check API key
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
    # Decode image
    # --------------------------------------------------------

    try:

        image_bytes, mime_type = process_base64_image(
            request.image_base64
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a highly accurate visual document question-answering system.

Carefully inspect the supplied image. It may contain:
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
6. The answer must be concise.
7. If the requested answer is numeric, return only the number.
8. For a numeric answer, do not include currency symbols.
9. For a numeric answer, do not include units unless the question
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
    # Call Gemini
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


        answer = response.text


        if not answer:

            raise HTTPException(
                status_code=502,
                detail=(
                    "The multimodal model returned "
                    "an empty answer."
                ),
            )


        answer = clean_answer(answer)


        # The assignment requires a string
        return {
            "answer": str(answer)
        }


    except HTTPException:
        raise


    except Exception as error:

        print(
            f"Gemini API error: "
            f"{type(error).__name__}: {error}"
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Failed to process the image "
                "with the multimodal model."
            ),
        )
