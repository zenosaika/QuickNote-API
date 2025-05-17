from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_users import FastAPIUsers
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from fastapi_users.manager import BaseUserManager, UUIDIDMixin

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

import os
import logging
import json
import uuid
from typing import List, Optional, Dict, Any

import pydub
import math
import tempfile

from urllib.parse import quote

from fastapi.concurrency import run_in_threadpool
import asyncio

from database import create_db_and_tables, get_async_session
from models import User, Transcription
import schemas
from security import auth_backend

import io
import re
from markdown_pdf import MarkdownPdf, Section
from md2docx_python.src.md2docx_python import markdown_to_word

from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()


model = "gemini-2.5-flash-preview-04-17"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# Maximum duration for audio chunks
MAX_API_CHUNK_DURATION_MS = 30 * 60 * 1000 # 30 minutes


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Lifespan event: Initializing database...")
    os.makedirs("temp_audio", exist_ok=True)
    os.makedirs("temp_document", exist_ok=True)
    logger.info("Lifespan event: Temp directory checked/created.")
    await create_db_and_tables()
    logger.info("Lifespan event: Database setup complete.")
    yield
    logger.info("Lifespan event: Shutting down application...")


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    loaded_secret = os.getenv("SECRET_KEY")

    if not loaded_secret:
        raise ValueError("SECRET from env is missing for UserManager!")
    
    reset_password_token_secret = loaded_secret
    verification_token_secret = loaded_secret

    async def on_after_register(self, user: User, request: Optional[Request] = None): 
        print(f"User {user.id} has registered.")

    async def on_after_forgot_password(self, user: User, token: str, request: Optional[Request] = None): 
        print(f"User {user.id} forgot password. Token: {token}")

    async def on_after_request_verify(self, user: User, token: str, request: Optional[Request] = None): 
        print(f"Verification requested for {user.id}. Token: {token}")


async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)


async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)


fastapi_users = FastAPIUsers[User, uuid.UUID](
    get_user_manager,
    [auth_backend],
)


if 'logger' not in locals():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


app = FastAPI(
    title="QuickNote API",
    description="Audio Transcription and Summarization API",
    version="1.0.0",
    lifespan=lifespan
)


origins = ["http://localhost:3000", 
           "http://127.0.0.1:3000",
           "https://quicknote-g6ic.onrender.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_mm_ss_s(ts_string):
    ts_string = ts_string.rstrip('s')
    parts = ts_string.split(':')
    minutes = float(parts[0])
    seconds = float(parts[1])
    return minutes * 60 + seconds


def format_seconds_to_mm_ss(seconds):
    if seconds < 0:
        seconds = 0
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def get_fixed_duration_segment_boundaries(
    total_duration_ms,
    max_segment_duration_ms
):
    if total_duration_ms == 0 or max_segment_duration_ms <= 0:
        return []

    num_chunks = math.ceil(total_duration_ms / max_segment_duration_ms)
    segments_ms = []
    for i in range(num_chunks):
        start_ms = i * max_segment_duration_ms
        end_ms = min((i + 1) * max_segment_duration_ms, total_duration_ms)
        if end_ms > start_ms + 50:
             segments_ms.append([start_ms, end_ms])
        else:
            logger.debug(f"Skipping tiny fixed segment: {start_ms/1000:.2f}s - {end_ms/1000:.2f}s")

    logger.info(f"Generated {len(segments_ms)} fixed-duration segments.")
    return segments_ms


async def generate_transcription(path_to_audio_file):
    uploaded_file = await run_in_threadpool(client.files.upload, file=path_to_audio_file)
    files = [uploaded_file]

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=files[0].uri,
                    mime_type=files[0].mime_type,
                ),
                types.Part.from_text(text="""Please transcribe the provided audio file. Your task is to generate a detailed transcription that includes the spoken text, timestamps for the start and end of each speech segment (using timestamps of the form MM:SS), and speaker diarization (identifying different speakers; speaker IDs start from 1)."""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["transcriptions"],
            properties={
                "transcriptions": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        required=["start_timestamp", "end_timestamp",
                                  "speaker_id", "text_transcript"],
                        properties={
                            "start_timestamp": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                            "end_timestamp": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                            "speaker_id": genai.types.Schema(
                                type=genai.types.Type.INTEGER,
                            ),
                            "text_transcript": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                        },
                    ),
                ),
            },
        ),
    )

    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config
    )

    try:
        await run_in_threadpool(client.files.delete, name=files[0].name)
        logger.info(f"Deleted uploaded file {files[0].name} from Gemini Files API.")
    except Exception as delete_e:
        logger.warning(f"Failed to delete file {files[0].name} from Gemini Files API: {delete_e}")

    return json.loads("".join(response.text))


async def summarize_transcription(transcriptions):
    transcription = '\n'.join(
        [f'SPEAKER {t["speaker_id"]}: {t["text_transcript"]}' for t in transcriptions])

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"""Analyze the following transcription. First, determine if it strongly aligns with the characteristics of a \"Meeting\", \"Podcast\", or \"Lecture\".

- Meeting characteristics: Multiple speakers discussing topics, making decisions, assigning action items, potential for agenda points.
- Podcast characteristics: Conversational flow, often with host(s) and guest(s), discussion of specific themes or stories, less formal structure than a lecture.
- Lecture characteristics: Primary speaker presenting structured information on a specific subject, often with a clear progression of topics, educational in tone.
If the transcription clearly fits one of these types, proceed with the specific summarization instructions for that type below.

If the transcription does NOT strongly align with Meeting, Podcast, or Lecture, analyze the content to infer its purpose and what a user would likely need a summary of. For example, is it an interview (focus on questions and answers, key insights from the interviewee), a presentation (focus on main points and supporting details), a narrative (focus on the story arc and key events), etc.?

Transcription:
{transcription}

Summarization Instructions:

Based on the inferred Transcription Type:
- If inferred as 'Meeting': Summarize the key discussions, decisions made, assigned action items, and the different viewpoints or contributions from participants. Emphasize the outcomes and any agreed-upon next steps. Utilize speaker IDs to indicate who was associated with key decisions or actions.
- If inferred as 'Podcast': Summarize the main topics covered, the participants (host(s) and guest(s)), the core arguments or insights presented, and any significant stories or examples. The summary should capture the essence and flow of the conversation. Mention speakers when attributing specific opinions or key pieces of information.
- If inferred as 'Lecture': Summarize the primary concepts, theories, and essential information delivered by the main speaker. Structure the summary logically, following the lecture's progression. Include definitions of important terms and relevant examples. Focus on the core educational content, attributing information to the lecturer (or relevant speaker ID).
- If inferred as 'Other' (not Meeting, Podcast, or Lecture): Based on your analysis of the content's purpose, provide a general summary that captures the most important information, key themes, and overall flow. Focus on what a user reading this transcription would likely want to take away from it. Use speaker IDs where they help clarify who said what significant piece of information or contribution.
                                     
Language Output Condition:
- If the provided transcription contains a significant amount of Thai language, provide the entire summary primarily in Thai.
- Crucially, when summarizing in Thai, retain English words or technical terms from the original transcription if they are commonly used or provide better clarity than a direct Thai translation. Mix English terms naturally within the Thai sentences where appropriate.
- If the transcription is primarily in English or another language (with little or no Thai), provide the summary in English.

General Instructions:
- Use the provided speaker IDs within the summary where it adds clarity, context, or highlights a specific contribution.
- Maintain a neutral and objective tone throughout the summary.
- Ensure the summary is coherent, easy to read, and focuses on the most important information relevant to the inferred type or purpose.
- Avoid unnecessary conversational filler or repetitive points unless they are central to the content.
- The summary should be a concise representation of the provided transcription.

Desired Output Format:
- Present the summary in GitHub Markdown format
- Structured sections with headings based on the content flow.
- A simple bulleted list of key takeaways."""),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.5,
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["inferred_type", "summarize_markdown"],
            properties={
                "inferred_type": genai.types.Schema(
                    type=genai.types.Type.STRING,
                ),
                "summarize_markdown": genai.types.Schema(
                    type=genai.types.Type.STRING,
                ),
            },
        ),
    )

    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config
    )

    return json.loads("".join(response.text))


async def process_single_audio_chunk(
    full_audio_segment,
    start_ms,
    end_ms,
    chunk_idx,
    total_chunks
):
    chunk_temp_file_path = None
    processed_segments = []
    logger.info(f"  Starting chunk {chunk_idx}/{total_chunks} ({start_ms/1000:.2f}s - {end_ms/1000:.2f}s)")

    try:
        chunk_audio = full_audio_segment[start_ms:end_ms]

        fd, chunk_temp_file_path = tempfile.mkstemp(suffix=".mp3", dir="temp_audio")
        os.close(fd)

        logger.debug(f"  Exporting chunk {chunk_idx} to MP3: {chunk_temp_file_path}")
        await run_in_threadpool(chunk_audio.export, chunk_temp_file_path, format="mp3")
        logger.debug(f"  Exported chunk {chunk_idx}.")

        logger.debug(f"  Calling transcription API for chunk {chunk_idx}")
        chunk_result_obj = await generate_transcription(chunk_temp_file_path)
        chunk_transcription_segments = chunk_result_obj.get("transcriptions", [])
        logger.debug(f"  API returned {len(chunk_transcription_segments)} segments for chunk {chunk_idx}")

        if 1 <= chunk_idx <= 26:
            chunk_letter_prefix = chr(ord('A') + chunk_idx - 1)
        else:
            logger.warning(
                f"Chunk index {chunk_idx} is outside the A-Z range (1-26). "
                f"Using 'Ck{chunk_idx}_' as speaker prefix."
            )
            chunk_letter_prefix = f"Ck{chunk_idx}_"

        chunk_start_offset_sec = start_ms / 1000.0
        for segment in chunk_transcription_segments:
            try:
                raw_start_ts = segment.get('start_timestamp', '00:00')
                raw_end_ts = segment.get('end_timestamp', '00:00')
                relative_start_sec = parse_mm_ss_s(raw_start_ts)
                relative_end_sec = parse_mm_ss_s(raw_end_ts)
                absolute_start_sec = chunk_start_offset_sec + relative_start_sec
                absolute_end_sec = chunk_start_offset_sec + relative_end_sec

                original_speaker_id_from_api = int(segment.get('speaker_id', 0))
                new_speaker_id = f"{chunk_letter_prefix}{original_speaker_id_from_api}"

                adjusted_segment = {
                    'start_timestamp': format_seconds_to_mm_ss(absolute_start_sec),
                    'end_timestamp': format_seconds_to_mm_ss(absolute_end_sec),
                    'speaker_id': new_speaker_id,
                    'text_transcript': str(segment.get('text_transcript', ''))
                }
                processed_segments.append(adjusted_segment)
            except Exception as e:
                logger.warning(f"  Failed to process individual segment result from API for chunk {chunk_idx}: {segment}. Error: {e}", exc_info=True)
        
        logger.info(f"  Finished processing chunk {chunk_idx}/{total_chunks}. Segments found: {len(processed_segments)}")
        return processed_segments

    except Exception as e:
        logger.error(f"  Error processing chunk {chunk_idx} ({start_ms/1000:.2f}s - {end_ms/1000:.2f}s): {e}", exc_info=True)
        raise
    finally:
        if chunk_temp_file_path and os.path.exists(chunk_temp_file_path):
            try:
                await run_in_threadpool(os.remove, chunk_temp_file_path)
                logger.debug(f"  Cleaned up chunk temp file for chunk {chunk_idx}: {chunk_temp_file_path}")
            except OSError as remove_error:
                logger.error(f"  Error removing chunk temp file {chunk_temp_file_path} for chunk {chunk_idx}: {remove_error}")



current_active_user = fastapi_users.current_user(active=True)

# API Endpoints
app.include_router(fastapi_users.get_auth_router(
    auth_backend), prefix="/api/auth", tags=["auth"])
app.include_router(fastapi_users.get_register_router(
    schemas.UserRead, schemas.UserCreate), prefix="/api/auth", tags=["auth"])


@app.get("/api/auth/session", response_model=schemas.UserRead, tags=["auth"])
async def get_session(user: User = Depends(current_active_user)): return user


@app.post("/api/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file.")
    if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
        logger.warning(f"Non-audio type: {audio_file.content_type}")

    temp_dir = "temp_audio"
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{audio_file.filename}")
    db_transcription = None
    transcription_segments = None
    summarized_text = None
    all_transcription_results = []
    full_audio = None

    try:
        try:
            content = await audio_file.read()
            await run_in_threadpool(lambda p, c: open(p, "wb").write(c), temp_file_path, content)
            logger.info(f"User '{user.email}' uploaded file, saved to: {temp_file_path}")
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Could not save file.")

        try:
            logger.info(f"Loading audio from temporary file: {temp_file_path}")
            full_audio = await run_in_threadpool(pydub.AudioSegment.from_file, temp_file_path)
            logger.info(f"Audio loaded successfully. Duration: {len(full_audio) / 1000:.2f} seconds.")
        except pydub.exceptions.CouldntDecodeError as e:
             logger.error(f"pydub could not decode audio file {temp_file_path}: {e}", exc_info=True)
             raise HTTPException(status_code=400, detail="Could not decode audio file format.")
        except FileNotFoundError:
             logger.error(f"Temporary audio file not found after saving: {temp_file_path}", exc_info=True)
             raise HTTPException(status_code=500, detail="Internal error: Temporary file not found.")
        except Exception as e:
            logger.error(f"Error during audio loading: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error processing audio file.")

        total_duration_ms = len(full_audio)
        logger.info("Using fixed-duration splitting.")
        segment_intervals_ms = get_fixed_duration_segment_boundaries(
            total_duration_ms,
            MAX_API_CHUNK_DURATION_MS
        )

        if not segment_intervals_ms:
             logger.warning("No audio segments determined for transcription.")
             all_transcription_results = []
        else:
            logger.info(f"Creating {len(segment_intervals_ms)} parallel tasks for audio chunks.")
            tasks = []
            for i, (start_ms, end_ms) in enumerate(segment_intervals_ms):
                tasks.append(process_single_audio_chunk(full_audio, start_ms, end_ms, i + 1, len(segment_intervals_ms)))
            
            list_of_chunk_results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)
            
            processed_chunks_count = 0
            failed_chunks_count = 0
            for i, result_or_exc in enumerate(list_of_chunk_results_or_exceptions):
                if isinstance(result_or_exc, Exception):
                    logger.error(f"Chunk {i+1}/{len(tasks)} processing failed with exception: {result_or_exc}")
                    failed_chunks_count += 1
                elif result_or_exc:
                    all_transcription_results.extend(result_or_exc)
                    processed_chunks_count +=1
                else:
                    logger.warning(f"Chunk {i+1}/{len(tasks)} returned an unexpected empty or None result.")


            logger.info(f"Finished processing all {len(tasks)} chunk tasks. "
                        f"Successful: {processed_chunks_count}, Failed: {failed_chunks_count}. "
                        f"Total combined segments: {len(all_transcription_results)}")

            if not all_transcription_results and failed_chunks_count > 0 and processed_chunks_count == 0:
                raise HTTPException(status_code=500, detail="All audio chunk processing tasks failed.")

            all_transcription_results.sort(key=lambda x: parse_mm_ss_s(x.get('start_timestamp', '00:00')))

        summarization_result_obj = await summarize_transcription(all_transcription_results)
        summarized_text = summarization_result_obj.get("summarize_markdown")

        # Save Metadata and Result to Database
        logger.info(f"Saving transcription metadata for user {user.id}...")
        try:
            transcription_result_json = json.dumps(all_transcription_results)

            db_transcription = Transcription(
                user_id=user.id,
                filename=audio_file.filename,
                status="completed",
                result_json=transcription_result_json,
                summarized_text=summarized_text
            )

            session.add(db_transcription)
            await session.commit()
            await session.refresh(db_transcription)
            logger.info(f"Transcription metadata saved with ID: {db_transcription.id}")

        except Exception as db_e:
            await session.rollback()
            logger.error(f"Error saving transcription to DB: {db_e}", exc_info=True)

            raise HTTPException(
                status_code=500, detail="Transcription successful, but failed to save result to database.")

        return {
            'transcriptions': transcription_segments,
            'summarized_text': summarized_text
        }

    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        logger.error(f"Unexpected error in /transcribe for {audio_file.filename}, user {user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription process failed unexpectedly: {e}")
    finally:
        # Cleanup
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up local temporary file: {temp_file_path}")
            except OSError as remove_error:
                logger.error(f"Error removing local temp file {temp_file_path}: {remove_error}")


@app.get("/api/history", response_model=List[schemas.TranscriptionHistoryItem], tags=["transcription"])
async def get_transcription_history(
    user: User = Depends(current_active_user), session: AsyncSession = Depends(get_async_session),
    skip: int = 0, limit: int = 100
):
    logger.info(f"Fetching history for user {user.id} (skip={skip}, limit={limit})")
    query = (
        select(Transcription)
        .where(Transcription.user_id == user.id)
        .order_by(Transcription.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await session.execute(query)
    history_items = result.scalars().all()
    logger.info(f"Found {len(history_items)} history items for user {user.id}")

    return history_items


@app.get("/api/transcription/{transcription_id}", response_model=schemas.TranscriptionDetail, tags=["transcription"])
async def get_transcription_detail(
    transcription_id: uuid.UUID,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    logger.info(f"Fetching details for transcription {transcription_id}, user {user.id}")

    query = select(Transcription).where(Transcription.id == transcription_id)
    result = await session.execute(query)
    db_transcription = result.scalar_one_or_none()

    if db_transcription is None:
        logger.warning(f"Transcription {transcription_id} not found.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Transcription not found")

    if db_transcription.user_id != user.id:
        logger.warning(f"User {user.id} attempted access to transcription {transcription_id} owned by {db_transcription.user_id}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Not authorized to view this transcription")
    
    parsed_segments_list: Optional[List[Dict[str, Any]]] = None
    try:
        parsed_segments_list = json.loads(db_transcription.result_json)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON for transcription {transcription_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error decoding transcription result.")

    result_object_for_schema: Optional[Dict[str, List]] = None
    if parsed_segments_list is not None:
        result_object_for_schema = schemas.TranscriptionSegments(transcriptions=parsed_segments_list).model_dump()

    response_data = {
        "id": db_transcription.id,
        "filename": db_transcription.filename,
        "created_at": db_transcription.created_at,
        "status": db_transcription.status,
        "summarized_text": db_transcription.summarized_text,
        "result": result_object_for_schema
    }

    return response_data


@app.get("/api/transcription/{transcription_id}/export/pdf", tags=["transcription"], summary="Export summary as PDF")
async def export_summary_pdf(
    transcription_id: uuid.UUID,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    logger.info(f"Attempting PDF export for transcription {transcription_id} by user {user.id}")

    query = select(Transcription).where(Transcription.id == transcription_id)
    result = await session.execute(query)
    db_transcription = result.scalar_one_or_none()

    if db_transcription is None:
        logger.warning(f"PDF export failed: Transcription {transcription_id} not found.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Transcription not found")

    if db_transcription.user_id != user.id:
        logger.warning(f"PDF export unauthorized: User {user.id} attempted access to transcription {transcription_id}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Not authorized to export this transcription")

    summary_text = db_transcription.summarized_text

    if not summary_text or not summary_text.strip():
        logger.info(f"PDF export requested for transcription {transcription_id} but no summary available.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="No summary available for this transcription.")

    try:
        pdf = MarkdownPdf(toc_level=2, optimize=True)
        pdf.add_section(Section(summary_text, toc=False))

        pdf_buffer = io.BytesIO()
        pdf.save(pdf_buffer)
        pdf_buffer.seek(0)

        filename = f"{db_transcription.filename.rsplit('.', 1)[0] if '.' in db_transcription.filename else db_transcription.filename}_summary.pdf"

        logger.info(f"PDF generated successfully for transcription {transcription_id}.")

        encoded_filename = quote(filename)
        content_disposition_header = f"attachment; filename*=UTF-8''{encoded_filename}"

        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                'Content-Disposition': content_disposition_header
            }
        )

    except Exception as e:
        logger.error(f"Error generating PDF for transcription {transcription_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate PDF.")


@app.get("/api/transcription/{transcription_id}/export/docx", tags=["transcription"], summary="Export summary as DOCX")
async def export_summary_docx(
    transcription_id: uuid.UUID,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    logger.info(f"Attempting DOCX export for transcription {transcription_id} by user {user.id}")

    query = select(Transcription).where(Transcription.id == transcription_id)
    result = await session.execute(query)
    db_transcription = result.scalar_one_or_none()

    if db_transcription is None:
        logger.warning(f"DOCX export failed: Transcription {transcription_id} not found.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Transcription not found")

    if db_transcription.user_id != user.id:
        logger.warning(f"DOCX export unauthorized: User {user.id} attempted access to transcription {transcription_id}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Not authorized to export this transcription")

    summary_text = db_transcription.summarized_text

    if not summary_text or not summary_text.strip():
        logger.info(f"DOCX export requested for transcription {transcription_id} but no summary available.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="No summary available for this transcription.")

    try:
        temp_id = str(db_transcription.id)
        temp_md_file = f'temp_document/{temp_id}.md'
        temp_docx_file = f'temp_document/{temp_id}.docx'

        summary_text = re.sub(r"(^\s*)\*\s", r"\1- ", summary_text, flags=re.MULTILINE)

        with open(temp_md_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)

        markdown_to_word(temp_md_file, temp_docx_file)
        
        docx_buffer = io.BytesIO()
        with open(temp_docx_file, 'rb') as temp_file_handle:
             docx_buffer.write(temp_file_handle.read())
        docx_buffer.seek(0)

        if os.path.exists(temp_md_file):
            os.remove(temp_md_file)
        if os.path.exists(temp_docx_file):
            os.remove(temp_docx_file)

        filename = f"{db_transcription.filename.rsplit('.', 1)[0] if '.' in db_transcription.filename else db_transcription.filename}_summary.docx"

        logger.info(f"DOCX generated successfully for transcription {transcription_id}.")

        encoded_filename = quote(filename)
        content_disposition_header = f"attachment; filename*=UTF-8''{encoded_filename}"

        return StreamingResponse(
            docx_buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                'Content-Disposition': content_disposition_header
            }
        )

    except Exception as e:
        logger.error(f"Error generating DOCX for transcription {transcription_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate DOCX.")


@app.delete("/api/transcription/{transcription_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["transcription"])
async def delete_transcription(
    transcription_id: uuid.UUID, user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    logger.info(f"Attempting delete transcription {transcription_id} by user {user.id}")
    query = select(Transcription).where(Transcription.id == transcription_id)
    result = await session.execute(query)
    db_transcription = result.scalar_one_or_none()

    if db_transcription is None:
        raise HTTPException(status_code=404, detail="Transcription not found")
    if db_transcription.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this transcription")

    try:
        await session.delete(db_transcription)
        await session.commit()
        logger.info(f"Deleted transcription {transcription_id}")
        return
    except Exception as e:
        await session.rollback()
        logger.error(f"Error deleting transcription {transcription_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not delete transcription.")


@app.get("/")
async def read_root(): return {"message": "Welcome to the QuickNote Transcription and Summarization API"}

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000,
#                 reload=True, timeout_keep_alive=120)