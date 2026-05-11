"""
quipux_api — main.py  (compatible con quipux_extractor.py v4)
====================
API REST para el extractor NER de documentos Quipux/SERCOP.

Endpoints
---------
  GET  /health          → estado de la API y modelos cargados
  POST /extraer         → procesa un PDF y devuelve JSON enriquecido
  POST /extraer/batch   → procesa múltiples PDFs en paralelo
  GET  /docs            → Swagger UI (automático con FastAPI)

Uso local
---------
  uvicorn main:app --reload --port 8000

Variables de entorno
--------------------
  QUIPUX_DEVICE       cpu | cuda          (default: cpu)
  QUIPUX_LOAD_BETO    true | false        (default: true)
  QUIPUX_MAX_FILE_MB  tamaño máximo PDF   (default: 20)
  API_KEY             clave Bearer        (default: desactivado)
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from quipux_extractor import QuipuxDocument, QuipuxPipeline

log = logging.getLogger("quipux.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ===========================================================================
# CONFIGURACIÓN
# ===========================================================================

DEVICE       = os.getenv("QUIPUX_DEVICE", "cpu")
LOAD_BETO    = os.getenv("QUIPUX_LOAD_BETO", "true").lower() == "true"
MAX_FILE_MB  = int(os.getenv("QUIPUX_MAX_FILE_MB", "20"))
API_KEY      = os.getenv("API_KEY", "")
MAX_BATCH    = 10

# ===========================================================================
# ESTADO GLOBAL
# ===========================================================================

class AppState:
    pipeline: QuipuxPipeline | None = None
    started_at: float = 0.0
    models_ready: bool = False

app_state = AppState()


# ===========================================================================
# LIFESPAN
# ===========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Iniciando QuipuxPipeline (device=%s, beto=%s)…", DEVICE, LOAD_BETO)
    t0 = time.time()
    app_state.pipeline = QuipuxPipeline(device=DEVICE, load_beto=LOAD_BETO)
    app_state.started_at = time.time()
    app_state.models_ready = True
    log.info("Pipeline listo en %.1f s", time.time() - t0)
    yield
    log.info("Cerrando API.")


# ===========================================================================
# APP
# ===========================================================================

app = FastAPI(
    title="Quipux NER API",
    description="Extracción de entidades nombradas en documentos PDF Quipux/SERCOP.",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# MODELOS DE RESPUESTA
# ===========================================================================

class HealthResponse(BaseModel):
    status: str
    models_ready: bool
    device: str
    beto_enabled: bool
    uptime_seconds: float


class ExtractionResponse(BaseModel):
    request_id: str = Field(description="UUID único de la request")
    procesado_en_ms: float
    resultado: dict = Field(description="JSON completo del QuipuxDocument")


class BatchExtractionResponse(BaseModel):
    total: int
    exitosos: int
    fallidos: int
    resultados: list[dict]


# ===========================================================================
# DEPENDENCIAS
# ===========================================================================

def require_api_key(authorization: Annotated[str | None, Header()] = None):
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida o ausente.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_pipeline() -> QuipuxPipeline:
    if not app_state.models_ready or app_state.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelos aún cargando. Reintenta en unos segundos.",
        )
    return app_state.pipeline


def _validate_pdf(content: bytes, filename: str) -> None:
    """Valida tamaño y header mágico del PDF."""
    if len(content) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"'{filename}' supera el límite de {MAX_FILE_MB} MB.",
        )
    if not content.startswith(b"%PDF"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"'{filename}' no parece un PDF válido.",
        )


def _process_bytes(content: bytes, filename: str, pipeline: QuipuxPipeline) -> tuple[dict, float]:
    """
    Escribe el contenido en un temporal, llama a pipeline.process(Path)
    y devuelve (resultado_dict, elapsed_ms).
    Compatible con QuipuxPipeline.process(pdf_path: Path) de v4.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        t0 = time.time()
        doc: QuipuxDocument = pipeline.process(tmp_path)
        elapsed_ms = (time.time() - t0) * 1000
        resultado = doc.to_dict()
        # Reemplazar ruta temporal con el nombre real del archivo subido
        resultado["fuente"] = filename
        return resultado, elapsed_ms
    finally:
        tmp_path.unlink(missing_ok=True)


# ===========================================================================
# ENDPOINTS
# ===========================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado de la API y modelos",
    tags=["Sistema"],
)
def health():
    uptime = time.time() - app_state.started_at if app_state.started_at else 0.0
    return HealthResponse(
        status="ok" if app_state.models_ready else "iniciando",
        models_ready=app_state.models_ready,
        device=DEVICE,
        beto_enabled=LOAD_BETO,
        uptime_seconds=round(uptime, 1),
    )


@app.post(
    "/extraer",
    response_model=ExtractionResponse,
    summary="Extrae entidades de un PDF Quipux",
    tags=["Extracción"],
    dependencies=[Depends(require_api_key)],
)
async def extraer(
    file: Annotated[UploadFile, File(description="PDF Quipux/SERCOP")],
    pipeline: QuipuxPipeline = Depends(require_pipeline),
):
    """
    Recibe un PDF Quipux/SERCOP y devuelve el JSON con entidades extraídas:
    personas, organizaciones, cargos, lugares, fechas, referencias normativas,
    estado del trámite, acción requerida y anexos.
    """
    request_id = str(uuid.uuid4())
    log.info("[%s] Procesando: %s", request_id, file.filename)

    content = await file.read()
    _validate_pdf(content, file.filename)

    try:
        resultado, elapsed_ms = _process_bytes(content, file.filename, pipeline)

        log.info(
            "[%s] OK en %.0f ms | tipo=%s procesable=%s",
            request_id,
            elapsed_ms,
            resultado.get("metadata", {}).get("tipo_documento"),
            resultado.get("metadata", {}).get("procesable"),
        )

        return ExtractionResponse(
            request_id=request_id,
            procesado_en_ms=round(elapsed_ms, 1),
            resultado=resultado,
        )

    except HTTPException:
        raise
    except Exception as exc:
        log.exception("[%s] Error procesando PDF: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al procesar el PDF: {exc}",
        )


@app.post(
    "/extraer/batch",
    response_model=BatchExtractionResponse,
    summary=f"Procesa hasta {MAX_BATCH} PDFs Quipux",
    tags=["Extracción"],
    dependencies=[Depends(require_api_key)],
)
async def extraer_batch(
    files: Annotated[list[UploadFile], File(description=f"Hasta {MAX_BATCH} PDFs")],
    pipeline: QuipuxPipeline = Depends(require_pipeline),
):
    """
    Procesa varios PDFs en una sola llamada (máximo 10).
    Los errores individuales no detienen el batch — se registran por archivo.
    """
    if len(files) > MAX_BATCH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Máximo {MAX_BATCH} PDFs por llamada. Recibidos: {len(files)}.",
        )

    resultados = []
    exitosos = 0
    fallidos = 0

    for file in files:
        item: dict = {"archivo": file.filename}
        content = await file.read()

        if len(content) > MAX_FILE_MB * 1024 * 1024:
            item["error"] = f"Supera {MAX_FILE_MB} MB"
            fallidos += 1
            resultados.append(item)
            continue

        if not content.startswith(b"%PDF"):
            item["error"] = "No es un PDF válido"
            fallidos += 1
            resultados.append(item)
            continue

        try:
            resultado, elapsed_ms = _process_bytes(content, file.filename, pipeline)
            item["procesado_en_ms"] = round(elapsed_ms, 1)
            item["resultado"] = resultado
            exitosos += 1
        except Exception as exc:
            log.warning("Batch error en '%s': %s", file.filename, exc)
            item["error"] = str(exc)
            fallidos += 1

        resultados.append(item)

    return BatchExtractionResponse(
        total=len(files),
        exitosos=exitosos,
        fallidos=fallidos,
        resultados=resultados,
    )


# ===========================================================================
# ARRANQUE LOCAL
# ===========================================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)