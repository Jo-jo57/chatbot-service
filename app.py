from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rag_service import INSUFFICIENT_INFORMATION_RESPONSE, RAGService
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

# Initialize Flask app
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
CORS(app)

rag_service = RAGService()

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# Add your local documents here. They will be chunked, embedded, and stored
# in ChromaDB when the Flask app starts.
DOCUMENT_PATHS = [
    "documents/Registration guide.docx",
    "documents/training guide.docx",
    "documents/fyp and pt.docx",
    "documents/FREQUENTLY ASKED QUESTIONS DURING ADMISSION (1).docx",
    "documents/UDSM Locations.docx",
    "documents/ALMANAC_2025-2026.pdf",
    "documents/UNDERGRADUATE_PROSPECTUS_2025-2026.pdf",
]

DOCUMENT_GLOBS = [
    "documents/*almanac*.pdf",
    "documents/*prospectus*.pdf",
    "documents/*training*.pdf",
]

def configured_document_paths():
    base_directory = Path(__file__).resolve().parent
    paths = []

    for document_path in DOCUMENT_PATHS:
        path = Path(document_path)
        if not path.is_absolute():
            path = base_directory / path
        paths.append(path)

    for document_glob in DOCUMENT_GLOBS:
        for path in base_directory.glob(document_glob):
            paths.append(path)

    seen = set()
    unique_paths = []
    for path in paths:
        resolved_path = str(path.resolve())
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        unique_paths.append(path)

    return unique_paths


def ingest_document_paths(paths, display_name_for_path=None):
    loaded_documents = []
    failed_documents = []

    for path in paths:
        try:
            display_name = display_name_for_path(path) if display_name_for_path else None
            result = rag_service.ingest_path(path, display_name=display_name)
            loaded_documents.append(result)
            if result.get("skipped"):
                logging.info(
                    "RAG document %s is already current with %s chunks",
                    result["filename"],
                    result["chunks_total"],
                )
            else:
                logging.info(
                    "Loaded RAG document %s into %s chunks",
                    result["filename"],
                    result["chunks_added"],
                )
        except Exception as e:
            error_message = str(e)
            failed_documents.append({
                "filename": path.name,
                "source_path": str(path.resolve()),
                "error": error_message,
            })
            logging.error(f"Error loading document {path}: {error_message}")

    return loaded_documents, failed_documents


def load_configured_documents():
    configured_paths = configured_document_paths()
    loaded_documents, failed_documents = ingest_document_paths(configured_paths)

    source_paths_to_keep = [
        str(path.resolve())
        for path in configured_paths
    ]

    if loaded_documents:
        purged_chunks = rag_service.purge_except_source_paths(
            source_paths_to_keep
        )
        if purged_chunks:
            logging.info("Removed %s chunks from unconfigured documents", purged_chunks)

    return loaded_documents, failed_documents


loaded_documents, failed_documents = load_configured_documents()
if rag_service.count() > 0:
    warmed_chunks = rag_service.warm_keyword_index()
    logging.info("Warmed RAG keyword index with %s chunks", warmed_chunks)
RAG_DISTANCE_THRESHOLD = 2.8
OLLAMA_TIMEOUT_SECONDS = 3
RAG_RETRIEVAL_TOP_K = 12
UDSM_LOCATIONS_DOCUMENT = "documents/UDSM Locations.docx"
CAMPUS_MAP_LABEL = "Open UDSM campus map"
CAMPUS_MAP_URL = "https://www.google.com/maps/search/?api=1&query=University+of+Dar+es+Salaam+campus+map"
CAMPUS_MAP_ACTION_TYPE = "open_campus_map"
KNOWN_CAMPUS_LOCATIONS = [
    {
        "aliases": {"coict", "coict building"},
        "name": "CoICT",
        "map_query": "UDSM CoICT Kijitonyama Campus Dar es Salaam",
        "description": (
            "CoICT is the College of Information and Communication Technologies "
            "at the University of Dar es Salaam. It is at UDSM's Kijitonyama "
            "Campus in Kijitonyama, Dar es Salaam, near the Sayansi/Mwenge area."
        ),
    },
    {
        "aliases": {
            "sjmc",
            "sjmc building",
            "school of journalism and mass communication",
            "journalism and mass communication",
        },
        "name": "SJMC",
        "map_query": "UDSM School of Journalism and Mass Communication SJMC Dar es Salaam",
        "description": (
            "SJMC is the School of Journalism and Mass Communication at the "
            "University of Dar es Salaam. It is a UDSM campus location in "
            "Dar es Salaam."
        ),
    },
]
KNOWN_MEETING_SCHEDULES = [
    {
        "aliases": {
            "coict board",
            "coict board meeting",
            "coict board meetings",
            "college of information and communication technologies board",
        },
        "name": "CoICT Board meetings",
        "source": "ALMANAC_2025-2026.pdf",
        "source_detail": "K: MEETINGS, College Boards",
        "meetings": [
            "Regular Meeting - Wednesday 03 December 2025",
            "Regular Meeting - Wednesday 04 March 2026",
            "Regular Meeting - Thursday 28 May 2026",
            "Special Meeting (Examination) - Friday 28 August 2026",
            "Special Meeting (Admission) - Wednesday 12 August 2026",
            "Special College/School/Institute Boards (Supp/Special Exams) - Thursday 22 October 2026",
            "Regular Meeting - Thursday 03 December 2026",
        ],
    },
]

LOCATION_TERMS = {
    "auditorium",
    "cafeteria",
    "cafe",
    "classroom",
    "coet",
    "cohu",
    "conas",
    "coss",
    "dining",
    "department",
    "faculty",
    "library",
    "hospital",
    "clinic",
    "coict",
    "sjmc",
    "journalism",
    "building",
    "campus",
    "hostel",
    "office",
    "school",
    "lecture",
    "hall of residence",
    "lab",
    "laboratory",
    "bank",
    "atm",
    "parking",
    "gate",
}

LOCATION_PHRASES = {
    "admission office",
    "administration building",
    "conas building",
    "coict building",
    "coet building",
    "cohu building",
    "coss building",
    "sjmc building",
    "school of journalism and mass communication",
    "journalism and mass communication",
    "lecture hall",
}

LOCATION_INTENT_TERMS = {
    "where",
    "location",
    "located",
    "locate",
    "find",
    "direction",
    "directions",
    "direct",
    "map",
    "navigate",
    "navigation",
    "route",
}

LOCATION_INTENT_PHRASES = {
    "how do i get",
    "take me to",
    "show me",
    "where is",
    "directions to",
}

NON_LOCATION_TERMS = {
    "account",
    "admission",
    "aris",
    "board",
    "course",
    "courses",
    "exam",
    "examination",
    "examinations",
    "fee",
    "fees",
    "letter",
    "meeting",
    "meetings",
    "password",
    "registration",
    "result",
    "results",
    "statement",
}

SUPPORTED_CAMPUS_TERMS = LOCATION_TERMS | {
    "academic",
    "admission",
    "admissions",
    "almanac",
    "aris",
    "certificate",
    "college",
    "course",
    "courses",
    "deadline",
    "degree",
    "exam",
    "examination",
    "fee",
    "fees",
    "fyp",
    "gpa",
    "graduation",
    "hostel",
    "id",
    "letter",
    "loan",
    "nhif",
    "payment",
    "practical",
    "program",
    "programme",
    "registration",
    "requirement",
    "requirements",
    "result",
    "results",
    "semester",
    "student",
    "transcript",
    "training",
    "tuition",
    "udsm",
}


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_fallback_response():
    return "I do not have enough information in the configured documents to answer that question."


def current_local_date():
    return datetime.now().date()


def date_window_for_query(user_message, today=None):
    query = " ".join(re.findall(r"[a-z0-9]+", str(user_message or "").lower()))
    today = today or current_local_date()

    if "today" in query:
        return today, today, "today"
    if "tomorrow" in query:
        tomorrow = today + timedelta(days=1)
        return tomorrow, tomorrow, "tomorrow"
    if "next week" in query:
        start = today + timedelta(days=7 - today.weekday())
        return start, start + timedelta(days=6), "next week"
    if "this week" in query or "current week" in query:
        start = today - timedelta(days=today.weekday())
        return start, start + timedelta(days=6), "this week"

    return None


def display_date(date_value):
    return date_value.strftime("%A %d %B %Y")


def parse_meeting_date(date_text):
    cleaned = re.sub(r"\s+", " ", str(date_text or "")).strip().rstrip(".")
    formats = (
        "%A %d %B %Y",
        "%d %B %Y",
        "%B %d %Y",
        "%B %d, %Y",
    )
    for date_format in formats:
        try:
            return datetime.strptime(cleaned, date_format).date()
        except ValueError:
            continue
    return None


def dated_response_items(response):
    text = keep_date_year_together(str(response or ""))
    date_pattern = (
        r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+"
        r"\d{1,2}\s+"
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+"
        r"\d{4}"
    )
    items = []

    for line in text.splitlines():
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
        match = re.search(date_pattern, cleaned_line, re.IGNORECASE)
        if not match:
            continue
        meeting_date = parse_meeting_date(match.group(0))
        if not meeting_date:
            continue
        description = re.sub(r"^\d+\.\s*", "", cleaned_line).strip()
        items.append((meeting_date, description))

    return items


def filter_dated_response_by_query(user_message, response):
    date_window = date_window_for_query(user_message)
    if not date_window:
        return response

    start_date, end_date, label = date_window
    items = dated_response_items(response)
    if not items:
        return response

    matching_items = [
        description
        for meeting_date, description in items
        if start_date <= meeting_date <= end_date
    ]
    date_range = (
        display_date(start_date)
        if start_date == end_date
        else f"{display_date(start_date)} to {display_date(end_date)}"
    )

    if not matching_items:
        return f"No matching meetings are listed for {label} ({date_range})."

    return (
        f"Matching meetings listed for {label} ({date_range}):\n\n"
        + "\n\n".join(
            f"{index}. {description}"
            for index, description in enumerate(matching_items, start=1)
        )
    )


def keep_date_year_together(text):
    weekday = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    month = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    date_prefixes = [
        rf"\b{weekday}\s+\d{{1,2}}\s+{month}",
        rf"\b\d{{1,2}}\s+{month}",
        rf"\b{month}\s+\d{{1,2}},?",
    ]
    for date_prefix in date_prefixes:
        text = re.sub(
            rf"({date_prefix})\s*\n+\s*(\d{{4}}\b)",
            r"\1 \2",
            text,
            flags=re.IGNORECASE,
        )
    return text


def format_chat_response(response):
    text = str(response or "").strip()
    if not text:
        return text

    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    text = "\n".join(lines)
    text = keep_date_year_together(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!^)(?<!\n\n)(\n?)((?<!\d)\d{1,2}\.\s+)", r"\n\n\2", text)
    text = re.sub(r"(?<!^)(?<!\n\n)(\n?)([-*]\s+)", r"\n\n\2", text)
    text = keep_date_year_together(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_campus_map_url(user_message):
    query = user_message.strip()
    if not query:
        return CAMPUS_MAP_URL
    return (
        "https://www.google.com/maps/search/?api=1&query="
        + quote_plus(f"University of Dar es Salaam {query}")
    )


def clean_location_destination(text):
    destination = text.strip()
    destination = re.sub(r"[?.!]+$", "", destination)
    destination = re.sub(
        r"^(?:please\s+)?(?:where\s+(?:is|are)|directions?\s+to|"
        r"how\s+do\s+i\s+get\s+to|how\s+can\s+i\s+get\s+to|"
        r"take\s+me\s+to|show\s+me|find|locate|map\s+of|location\s+of)\s+",
        "",
        destination,
        flags=re.IGNORECASE,
    )
    destination = re.sub(
        r"\b(?:located|location|directions?|navigate|navigation|map)\b",
        "",
        destination,
        flags=re.IGNORECASE,
    )
    destination = " ".join(destination.split())
    destination = re.sub(r"^(?:the|a|an)\s+", "", destination, flags=re.IGNORECASE)
    return destination or text.strip()


def schedule_query_for_retrieval(user_message):
    query = str(user_message or "")
    words = set(re.findall(r"[a-z0-9]+", query.lower()))
    if "ue" not in words:
        return query

    expanded_query = re.sub(
        r"\bue\b",
        "2nd semester examinations",
        query,
        flags=re.IGNORECASE,
    )
    schedule_terms = {
        "when",
        "date",
        "dates",
        "start",
        "starts",
        "begin",
        "begins",
        "end",
        "ends",
        "schedule",
        "timetable",
    }
    if words & schedule_terms:
        return expanded_query

    return f"when are {expanded_query}"


def find_known_meeting_schedule(user_message):
    normalized_message = user_message.lower()
    message_terms = set(re.findall(r"[a-z0-9]+", normalized_message))
    meeting_terms = {"meeting", "meetings", "schedule", "schedules", "board"}
    has_meeting_intent = bool(message_terms & meeting_terms)

    for schedule in KNOWN_MEETING_SCHEDULES:
        for alias in schedule["aliases"]:
            alias_terms = set(re.findall(r"[a-z0-9]+", alias))
            if alias in normalized_message or alias_terms <= message_terms:
                return schedule

    if {"coict", "board"} <= message_terms and has_meeting_intent:
        return KNOWN_MEETING_SCHEDULES[0]

    return None


def meeting_schedule_response(user_message):
    schedule = find_known_meeting_schedule(user_message)
    if not schedule:
        return None

    meeting_lines = "\n".join(
        f"- {meeting}" for meeting in schedule["meetings"]
    )
    return {
        "intent": "meeting_schedule",
        "response": (
            f"{schedule['name']} listed in {schedule['source']} "
            f"({schedule['source_detail']}):\n{meeting_lines}"
        ),
        "sources": [
            {
                "filename": schedule["source"],
                "section": schedule["source_detail"],
            }
        ],
        "action": None,
        "helpful_links": [],
    }


def find_known_campus_location(user_message):
    normalized_message = user_message.lower()
    message_terms = set(re.findall(r"[a-z0-9]+", normalized_message))

    for location in KNOWN_CAMPUS_LOCATIONS:
        for alias in location["aliases"]:
            alias_terms = set(re.findall(r"[a-z0-9]+", alias))
            if alias in normalized_message or alias_terms <= message_terms:
                return location

    return None


def build_campus_map_action(user_message, known_location=None):
    destination = (
        known_location["name"]
        if known_location
        else clean_location_destination(user_message)
    )
    search_query = "University of Dar es Salaam"
    if known_location:
        search_query = known_location["map_query"]
    elif destination:
        search_query = f"{search_query} {destination}"

    return {
        "type": CAMPUS_MAP_ACTION_TYPE,
        "label": CAMPUS_MAP_LABEL,
        "destination": destination,
        "search_query": search_query,
        "url": (
            "https://www.google.com/maps/search/?api=1&query="
            + quote_plus(search_query)
        ),
        "app_route": "campus_map",
    }


def udsm_locations_document_path():
    return Path(__file__).resolve().parent / UDSM_LOCATIONS_DOCUMENT


def location_document_chunks():
    path = udsm_locations_document_path()
    if not path.exists():
        return []
    return rag_service.chunks_for_source_path(path, limit=80)


def dedupe_chunks(chunks):
    unique_chunks = []
    seen = set()
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        key = (
            metadata.get("source_path"),
            metadata.get("chunk_index"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_chunks.append({
            **chunk,
            "metadata": metadata,
        })
    return unique_chunks


def location_retrieval_chunks(user_message):
    location_guide_chunks = location_document_chunks()
    corpus_chunks = (
        rag_service.retrieve(user_message, top_k=RAG_RETRIEVAL_TOP_K + 10)
        if rag_service.count() > 0
        else []
    )
    return dedupe_chunks(location_guide_chunks + corpus_chunks)


def is_location_question(user_message):
    normalized_message = user_message.lower()
    message_terms = set(re.findall(r"[a-z0-9]+", normalized_message))
    has_location_intent = bool(
        message_terms & LOCATION_INTENT_TERMS
        or any(phrase in normalized_message for phrase in LOCATION_INTENT_PHRASES)
    )
    has_location_target = bool(
        message_terms & LOCATION_TERMS
        or any(phrase in normalized_message for phrase in LOCATION_PHRASES)
    )
    asks_for_place = bool(
        "where" in message_terms
        or "located" in message_terms
        or "directions" in message_terms
        or "navigate" in message_terms
        or "map" in message_terms
        or any(phrase in normalized_message for phrase in LOCATION_INTENT_PHRASES)
    )
    looks_like_non_location = bool(message_terms & NON_LOCATION_TERMS)

    return has_location_intent and (
        has_location_target
        or (asks_for_place and not looks_like_non_location)
    )


def campus_map_response(user_message):
    known_location = find_known_campus_location(user_message)
    map_action = build_campus_map_action(user_message, known_location)
    if known_location:
        response = (
            f"{known_location['description']} "
            "Use the map link to open the exact pin and get directions from your current location."
        )
    else:
        response = (
            "For campus directions, please use the UDSM campus map. "
            "It can help you find places like the cafeteria, CoICT building, hospital, "
            "library, offices, lecture halls, and other campus locations."
        )

    return {
        "intent": "campus_location",
        "response": response,
        "sources": [],
        "action": map_action,
        "helpful_links": [
            {
                "title": CAMPUS_MAP_LABEL,
                "url": map_action["url"],
            }
        ],
    }


def location_rag_response(user_message):
    known_location = find_known_campus_location(user_message)
    if known_location:
        return campus_map_response(user_message)

    chunks = location_retrieval_chunks(user_message)
    if chunks:
        rag_result = rag_service.answer_from_document(
            user_message,
            retrieved_chunks=chunks,
        )
        if not is_insufficient_answer(rag_result["response"]):
            place_name = user_message
            if ":" in rag_result["response"]:
                place_name = rag_result["response"].split(":", 1)[0].strip()
            map_action = build_campus_map_action(place_name or user_message, known_location)
            return {
                "intent": "campus_location",
                "response": (
                    f"{rag_result['response']}\n\n"
                    "Use the campus map link to open the location and navigate more easily."
                ),
                "sources": rag_result["sources"],
                "action": map_action,
                "helpful_links": [
                    {
                        "title": CAMPUS_MAP_LABEL,
                        "url": map_action["url"],
                    }
                ],
            }

    map_result = campus_map_response(user_message)
    map_result["response"] = (
        "I could not find that exact place in the configured UDSM location documents. "
        f"{map_result['response']}"
    )
    return map_result


def is_insufficient_answer(response):
    return response in {
        INSUFFICIENT_INFORMATION_RESPONSE,
        "I could not find matching information in the uploaded document.",
        generate_fallback_response(),
    }


def has_reliable_retrieval(user_message, retrieved_chunks):
    if not retrieved_chunks:
        return False

    message_terms = set(re.findall(r"[a-z0-9]+", user_message.lower()))
    best_keyword_score = max(
        (chunk.get("keyword_score", 0) for chunk in retrieved_chunks),
        default=0,
    )
    best_distance = min(
        (chunk.get("distance", 999) for chunk in retrieved_chunks),
        default=999,
    )
    is_supported_topic = bool(message_terms & SUPPORTED_CAMPUS_TERMS)
    mentions_udsm = bool({"udsm", "university", "campus"} & message_terms)

    if best_keyword_score >= 2:
        return True
    if best_keyword_score >= 1 and (is_supported_topic or mentions_udsm):
        return True
    return best_distance <= 1.4 and (is_supported_topic or mentions_udsm)


@app.route('/')
def home():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json(silent=True) or {}
        user_message = data.get('message', '')
        context = data.get('context', {})
        
        # Extract context information
        topic = context.get('topic', None)
        urgency = context.get('urgency', None)
        
        if not user_message.strip():
            return jsonify({
                'error': 'Message cannot be empty'
            }), 400

        helpful_links = []
        action = None
        intent = None
        if is_location_question(user_message):
            map_result = location_rag_response(user_message)
            intent = map_result["intent"]
            response = map_result["response"]
            sources = map_result["sources"]
            action = map_result["action"]
            helpful_links = map_result["helpful_links"]
        else:
            retrieval_message = schedule_query_for_retrieval(user_message)
            meeting_result = meeting_schedule_response(retrieval_message)
            if meeting_result:
                intent = meeting_result["intent"]
                response = meeting_result["response"]
                sources = meeting_result["sources"]
                action = meeting_result["action"]
                helpful_links = meeting_result["helpful_links"]
            else:
                retrieved_chunks = (
                    rag_service.retrieve(retrieval_message, top_k=RAG_RETRIEVAL_TOP_K)
                    if rag_service.count() > 0
                    else []
                )
                retrieved_chunks = [
                    {
                        **chunk,
                        "metadata": chunk.get("metadata") or {},
                    }
                    for chunk in retrieved_chunks
                ]
                best_distance = min((chunk.get("distance", 999) for chunk in retrieved_chunks), default=None)

                if (
                    best_distance is not None
                    and best_distance <= RAG_DISTANCE_THRESHOLD
                    and has_reliable_retrieval(user_message, retrieved_chunks)
                ):
                    rag_result = rag_service.answer_from_document(
                        retrieval_message,
                        retrieved_chunks=retrieved_chunks,
                    )
                    if is_insufficient_answer(rag_result["response"]):
                        broader_chunks = rag_service.retrieve(
                            retrieval_message,
                            top_k=RAG_RETRIEVAL_TOP_K + 8,
                        )
                        if broader_chunks:
                            retrieved_chunks = [
                                {
                                    **chunk,
                                    "metadata": chunk.get("metadata") or {},
                                }
                                for chunk in broader_chunks
                            ]
                        rag_result = rag_service.answer_with_timeout(
                            retrieval_message,
                            retrieved_chunks=retrieved_chunks,
                            timeout_seconds=OLLAMA_TIMEOUT_SECONDS,
                        )
                        if is_insufficient_answer(rag_result["response"]):
                            rag_result = {
                                "response": generate_fallback_response(),
                                "sources": [],
                            }
                    response = rag_result['response']
                    sources = rag_result['sources']
                else:
                    response = generate_fallback_response()
                    sources = []

        if is_insufficient_answer(response):
            response = generate_fallback_response()
            sources = []
        response = filter_dated_response_by_query(user_message, response)
        response = format_chat_response(response)
        
        return jsonify({
            'response': response,
            'intent': intent,
            'action': action,
            'timestamp': get_current_time(),
            'sources': sources,
            'context': {
                'topic': topic,
                'map_action': action,
                'campus_map_url': action.get('url') if action else None,
                'helpful_links': helpful_links
            }
        })
        
    except Exception as e:
        logging.exception("Error processing chat request")
        return jsonify({
            'response': generate_fallback_response(),
            'error': f'Internal chat error: {str(e)}',
            'intent': None,
            'action': None,
            'timestamp': get_current_time(),
            'sources': [],
            'context': {
                'topic': None,
                'map_action': None,
                'helpful_links': []
            }
        }), 200

@app.route('/documents/status', methods=['GET'])
def document_status():
    """Return vector database document chunk status."""
    return jsonify({
        'stored_chunks': rag_service.count(),
        'configured_documents': loaded_documents,
        'failed_documents': failed_documents,
        'timestamp': get_current_time()
    })


@app.route('/documents/upload', methods=['POST'])
def upload_documents():
    """Upload documents and add them to the vector database."""
    files = request.files.getlist('files')
    if not files:
        return jsonify({
            'error': 'No files uploaded. Use the files form field.',
            'timestamp': get_current_time()
        }), 400

    uploaded_documents = []
    upload_errors = []
    for file_storage in files:
        if not file_storage or not file_storage.filename:
            continue
        try:
            uploaded_documents.append(rag_service.ingest_file(file_storage))
        except Exception as error:
            upload_errors.append({
                'filename': file_storage.filename,
                'error': str(error),
            })

    return jsonify({
        'uploaded_documents': uploaded_documents,
        'failed_documents': upload_errors,
        'stored_chunks': rag_service.count(),
        'timestamp': get_current_time()
    }), 207 if upload_errors else 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Campus Assistant Chatbot',
        'stored_chunks': rag_service.count(),
        'configured_documents': len(loaded_documents),
        'failed_documents': failed_documents,
        'timestamp': get_current_time()
    })

@app.route('/quick-help', methods=['GET'])
def quick_help():
    """Get quick help topics"""
    try:
        quick_topics = [
            "Registration process",
            "Admission letter",
            "ARIS",
            "Statement of results",
        ]
        return jsonify({
            'topics': quick_topics,
            'timestamp': get_current_time()
        })
    except Exception as e:
        logging.error(f"Error getting quick help: {str(e)}")
        return jsonify({'error': 'Unable to load quick help topics'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
