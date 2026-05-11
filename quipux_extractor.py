"""
quipux_extractor.py  —  v4
==========================
Extractor NER para documentos Quipux (PDF → JSON enriquecido).
Implementa la Matriz de Selección Tecnológica NER v2.

Cambios v4 respecto a v3
-------------------------
  - Módulo ASUNTO eliminado completamente (campo, embedding y extracción)
  - PERSONA: capa 1.5 de búsqueda contextual post-tratamiento sin ML
  - CARGO:   catálogo ampliado + detección de cargo_firmante por línea adyacente
  - ORG_PUBLICA: patrones Ruler más robustos, siglas institucionales añadidas
  - FECHA: segunda pasada sobre el cuerpo del documento como fallback
  - BETO: filtro mejorado (threshold score, limpieza de subwords, exclusión de tokens
          de ruido tipo "Quito", "Ecuador" que BETO etiqueta como PER erróneamente)
  - FallbackParser: extrae también ORG y LUGAR de cartas BID
  - QuipuxMetadata: campos asunto/asunto_embedding eliminados
  - Constantes centralizadas en _CONFIG para fácil mantenimiento

Tipos de documento soportados
------------------------------
  MEMORANDO   → cabecera PARA / DE / firmante estructurada
  OFICIO      → cabecera PARA / DE / firmante estructurada
  RESOLUCIÓN  → sin PARA/DE; firmante al inicio; artículos en cuerpo
  CIRCULAR    → PARA/DE similar a Memorando
  INFORME     → firmante al pie

Arquitectura
------------
  PDFProcessor       extrae texto plano (pdfplumber)
  DocumentClassifier detecta tipo de documento (RegEx, sin ML)
  StructuralParser   extrae bloques posicionales (PARA, DE, firmante)
  FallbackParser     extrae campos clave-valor de documentos externos
  EntityExtractor    orquesta los extractores de entidad (Singleton de modelos)
  MetadataEncoder    serializa a JSON

Uso
---
  python quipux_extractor.py oficio.pdf
  python quipux_extractor.py oficio.pdf --output resultado.json
  python quipux_extractor.py oficio.pdf --no-beto
"""

from __future__ import annotations

import json
import re
import sys
import argparse
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import dateparser
import pdfplumber
import spacy
import torch
from rapidfuzz import process as fuzz_process
from spacy.language import Language
from spacy.matcher import Matcher
from transformers import pipeline as hf_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("quipux.extractor")


# ===========================================================================
# CONSTANTES CENTRALIZADAS
# ===========================================================================

class _CONFIG:
    """Parámetros ajustables sin tocar lógica de extracción."""

    # Longitud mínima de token para aceptar como entidad
    MIN_PERSONA_LEN   = 7    # chars — filtra iniciales sueltas y ruido corto
    MIN_ORG_LEN       = 6
    MIN_CARGO_LEN     = 5

    # Score mínimo para coincidencia fuzzy de estado Quipux
    ESTADO_SCORE_MIN  = 82

    # Ventana de texto que BETO procesa (chars desde el inicio)
    BETO_HEAD_CHARS   = 800

    # Score mínimo de BETO para aceptar entidad (0–1)
    BETO_SCORE_MIN    = 0.80

    # Palabras que BETO suele etiquetar erróneamente como PER en docs EC
    BETO_PER_STOPLIST = frozenset({
        "quito", "guayaquil", "cuenca", "ecuador", "quipux", "sercop",
        "señor", "señora", "señorita", "estimado", "presente",
        "consideración", "adjunto", "cordialmente",
    })


# ===========================================================================
# SECCIÓN 1 — MODELOS DE DATOS
# ===========================================================================

@dataclass
class QuipuxMetadata:
    procesable: bool = True
    motivo_rechazo: str | None = None
    tipo_documento: str | None = None
    codigo_documental: str | None = None
    fecha_iso: str | None = None
    lugar: list[str] = field(default_factory=list)
    personas: list[str] = field(default_factory=list)
    organizaciones: list[str] = field(default_factory=list)
    cargos: list[str] = field(default_factory=list)
    referencias_normativas: list[str] = field(default_factory=list)
    estado: str | None = None
    accion_requerida: str | None = None
    accion_clase: str | None = None
    anexos: list[dict] = field(default_factory=list)


@dataclass
class QuipuxDocument:
    fuente: str
    metadata: QuipuxMetadata

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ===========================================================================
# SECCIÓN 2 — PDFProcessor
# ===========================================================================

class PDFProcessor:
    """
    Extrae texto plano limpio de PDF usando pdfplumber.
    Elimina artefactos de página: números de página, pie de firma Quipux,
    marcas de agua "COPIA NO CONTROLADA".
    """

    _NOISE = re.compile(
        r"^\s*\*\s*Documento firmado electrónicamente.*$"
        r"|^\s*\d+/\d+\s*$"
        r"|^\s*Página\s+\d+\s+de\s+\d+\s*$"
        r"|^\s*COPIA\s+NO\s+CONTROLADA\s*$",
        re.MULTILINE | re.IGNORECASE,
    )

    def extract_text(self, pdf_path: Path) -> str:
        log.info("Extrayendo texto de %s", pdf_path.name)
        pages: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=2, y_tolerance=3)
                if text:
                    pages.append(text)
        raw = "\n".join(pages)
        return self._NOISE.sub("", raw).strip()


# ===========================================================================
# SECCIÓN 3 — DocumentClassifier
# ===========================================================================

class DocumentClassifier:
    """
    Detecta el tipo de documento por las primeras líneas.
    100% RegEx — sin modelo ML.
    """

    _RE_NATIVO = re.compile(
        r"^\s*(Memorando|Oficio|Resolución|Circular|Informe)\s+Nro?\.",
        re.IGNORECASE | re.MULTILINE,
    )
    _RE_CARTA = re.compile(
        r"CARTA\s+DE\s+(?:INVITACI[OÓ]N|PRESENTACI[OÓ]N|SOLICITUD)"
        r"|CONVOCATORIA\s+(?:P[UÚ]BLICA|ABIERTA)"
        r"|SELECCI[OÓ]N\s+DE\s+CONSULTOR",
        re.IGNORECASE,
    )
    _RE_CONTRATO = re.compile(
        r"^\s*CONTRATO\s+(?:DE|N[°º]|Nro\.)|CONVENIO\s+(?:MARCO|DE\s+COOPERACI[OÓ]N)",
        re.IGNORECASE | re.MULTILINE,
    )

    TIPOS_NATIVOS = {"MEMORANDO", "OFICIO", "RESOLUCIÓN", "CIRCULAR", "INFORME"}

    def classify(self, text: str) -> tuple[str, bool, str | None]:
        """Retorna (tipo, procesable, motivo_rechazo)."""
        head = text[:500]

        m = self._RE_NATIVO.search(head)
        if m:
            return m.group(1).upper(), True, None

        if self._RE_CARTA.search(head):
            return "CARTA", False, (
                "Documento tipo CARTA/CONVOCATORIA: no es un documento Quipux nativo. "
                "Se extraen campos disponibles (fecha, lugar, organizaciones). "
                "Revisar manualmente o indexar como documento de referencia."
            )

        if self._RE_CONTRATO.search(head):
            return "CONTRATO", False, (
                "Documento tipo CONTRATO/CONVENIO: estructura no compatible con "
                "el parser Quipux. Indexar como documento adjunto de referencia."
            )

        return "DESCONOCIDO", False, (
            "Tipo de documento no reconocido. No comienza con "
            "'Memorando/Oficio/Resolución/Circular/Informe Nro.' "
            "Verificar que el PDF sea un documento Quipux válido."
        )


# ===========================================================================
# SECCIÓN 4 — FallbackParser
# ===========================================================================

class FallbackParser:
    """
    Parser de campos clave-valor para documentos externos (Cartas BID, Contratos).
    Extrae lo que puede sin asumir estructura Quipux.
    """

    _RE_KV = re.compile(
        r"^(?P<clave>Fecha de publicaci[oó]n|Instituci[oó]n|Organismo\s+\w+|"
        r"Pa[ií]s|Programa|N[uú]mero de operaci[oó]n|N[uú]mero de pr[eé]stamo|"
        r"T[ií]tulo[^:]*|N[uú]mero de los Documentos[^:]*)"
        r"\s*:\s*(?P<valor>.+?)$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Captura líneas con patrones de organismo ("Banco Interamericano de ...", etc.)
    _RE_ORG_EXTERNA = re.compile(
        r"\b(Banco\s+(?:Interamericano|Mundial|de\s+Desarrollo|Centroamericano)"
        r"|BID|BIRF|CAF|CEPAL|OEA|OPS|UNICEF|UNESCO|PNUD|FAO"
        r"|Fondo\s+Monetario\s+Internacional|Naciones\s+Unidas)\b",
        re.IGNORECASE,
    )

    def parse(self, text: str) -> dict:
        kv: dict[str, str] = {}
        for m in self._RE_KV.finditer(text):
            clave = " ".join(m.group("clave").split()).lower()
            valor = " ".join(m.group("valor").split()).strip()
            kv[clave] = valor

        # Organizaciones externas presentes en el texto
        orgs_externas = list({
            m.group(0) for m in self._RE_ORG_EXTERNA.finditer(text[:2000])
        })

        return {
            "para_nombres": [],
            "para_cargos": [],
            "de_nombres": [],
            "firmante": None,
            "firmante_cargo": None,
            "orgs_externas": orgs_externas,
            "kv_extra": kv,
        }


# ===========================================================================
# SECCIÓN 4b — StructuralParser
# ===========================================================================

class StructuralParser:
    """
    Extrae bloques posicionales de la cabecera Quipux.

    Estructura Memorando/Oficio/Circular/Informe:
        <Tipo> Nro. <CODIGO>
        <Ciudad>, [D.M.,] DD de mes de YYYY
        PARA:  [múltiples destinatarios con cargo]
        [DE:   remitente]
        <cuerpo>
        Atentamente,
        <firmante>  <cargo>

    Estructura Resolución:
        RESOLUCIÓN Nro. <CODIGO>
        <nombre firmante>
        <CARGO FIRMANTE>
        CONSIDERANDO: ...
        RESUELVE: ...
    """

    # Tratamientos profesionales Ecuador (más completo que v3)
    _TRAT = (
        r"(?:Srta?\.|Sra?\.|Dr[a]?\.|Ing\.(?:\s+Civil|\.)?|Abg?a?\.|"
        r"Lcda?\.|Mgs\.|MSc\.|PhD\.|Arq\.|CPA\.|Eco\.|Psic\.|Psicól\.|"
        r"Tlgo\.|Econ\.|Espc?\.|Psc\.|MVZ\.|Biól\.|Lic\.)"
    )
    # Nombre propio: 2-5 tokens en Title Case (acepta partículas de, del, de la)
    _NOMBRE = (
        r"[A-ZÁÉÍÓÚÑ][a-záéíóúñA-ZÁÉÍÓÚÑ]+"
        r"(?:\s+(?:de(?:\s+la?)?|del|y)\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñA-ZÁÉÍÓÚÑ]+)?"
        r"(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñA-ZÁÉÍÓÚÑ]+){1,3}"
    )

    # Código documental: cubre todos los formatos vistos en los PDFs
    RE_CODIGO = re.compile(
        r"\b("
        r"(?:[A-Z0-9]{2,15}-){2,6}\d{4}-\d{1,6}-[A-Z]{1,3}"   # SERCOP-CGAJ-2023-0489-M
        r"|(?:[A-Z0-9]{2,15}-){2,6}\d{4}"                        # DINARP-DINARP-2025-0079
        r"|\d{3}-[A-Z]{2}-[A-Z0-9]+-\d{4}"                       # 001-CP-DINARP-2025
        r")\b"
    )

    # Fecha en cabecera (primeras 12 líneas)
    RE_FECHA_HEAD = re.compile(
        r"(?:"
        r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ\s\.]+,\s*(?:D\.?M\.?,\s*)?"
        r"(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})"   # ciudad + fecha larga
        r"|(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})"  # solo fecha larga
        r"|(\d{1,2}/\d{1,2}/\d{4})"             # DD/MM/YYYY
        r")",
        re.IGNORECASE,
    )

    # Fecha en el cuerpo (fallback — más permisiva)
    RE_FECHA_BODY = re.compile(
        r"\b(\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
        r"septiembre|octubre|noviembre|diciembre)\s+de\s+\d{4})\b",
        re.IGNORECASE,
    )

    RE_PARA = re.compile(
        r"PARA\s*:\s*\n?(.*?)(?=\n\s*(?:DE|ASUNTO|REFERENCIA|REF\.?)\s*:"
        r"|\n\s*De mi consideración"
        r"|\n\s*Al respecto"
        r"|\n\s*Señor[a]?\s+[A-ZÁÉÍÓÚÑ])",
        re.IGNORECASE | re.DOTALL,
    )
    RE_DE = re.compile(
        r"\bDE\s*:\s*\n?(.*?)(?=\n\s*(?:ASUNTO|REFERENCIA|REF\.?)\s*:"
        r"|\n\s*De mi consideración"
        r"|\n\n)",
        re.IGNORECASE | re.DOTALL,
    )

    # Firmante al pie — acepta variantes de cierre
    RE_FIRMANTE = re.compile(
        r"(?:Atentamente|Comuníquese\s+y\s+Publíquese|Con\s+sentimientos"
        r"|Cordialmente|Saludos\s+cordiales)[^.\n]*[.,]?\s*\n+"
        r"(?:Documento firmado electrónicamente\s*\n+)?"
        r"(" + _TRAT + r"\s+" + _NOMBRE + r")",
        re.IGNORECASE,
    )
    # Firmante Resolución: justo después del número
    RE_FIRMANTE_RES = re.compile(
        r"RESOLUCIÓN[^\n]*\n+(" + _TRAT + r"\s+" + _NOMBRE + r")",
        re.IGNORECASE,
    )

    RE_NOMBRE_TRAT = re.compile(_TRAT + r"\s+" + _NOMBRE, re.IGNORECASE)

    # Cargo en sector público Ecuador — ampliado v4
    RE_CARGO = re.compile(
        r"(?:"
        r"Director[a]?\s+(?:Nacional|General|Administrativ[oa]|Financier[oa]|"
        r"Técnic[oa]|Ejecutiv[oa]|de\s+\w+(?:\s+\w+)?(?:\s+\w+)?)"
        r"|Coordinador[a]?\s+(?:General|de\s+\w+(?:\s+\w+)?|Zonal|Nacional|Regional)"
        r"|Subdirector[a]?\s+(?:Nacional|General|de\s+\w+(?:\s+\w+)?)"
        r"|Viceministro?[a]?(?:\s+de\s+\w+)?"
        r"|Ministro?[a]?(?:\s+de\s+\w+)?"
        r"|Secretario?[a]?\s+(?:Nacional|General|de\s+\w+(?:\s+\w+)?)"
        r"|Subsecretario?[a]?\s+(?:de\s+\w+(?:\s+\w+)?)?"
        r"|Registrador[a]\s+Mercantil(?:\s+de\s+\w+)?"
        r"|(?:Inscriptor[a]?|Certificador[a]?|Contador[a]|Tesorero?)"
        r"(?:\s+de\s+Registro\s+Mercantil)?"
        r"|Especialista\s+Administrativo\s+Financiero"
        r"|(?:Analista|Especialista|Asistente|Técnico[a]?)\s+"
        r"(?:Jurídic[oa]|Informátic[oa]|Administrativ[oa]|Senior|Junior)?"
        r"|Encargad[oa](?:\s+de\s+\w+(?:\s+\w+)?)?"
        r"|Subrogante"
        r"|Jefe[a]?\s+(?:Departamental|de\s+\w+(?:\s+\w+)?)"
        r"|Asesor[a]?\s+(?:Jurídic[oa]|Técnic[oa])?"
        r"|Gerente\s+(?:General|de\s+\w+(?:\s+\w+)?)"
        r")",
        re.IGNORECASE,
    )

    def parse(self, text: str, tipo: str) -> dict:
        result = {
            "para_nombres": [], "para_cargos": [],
            "de_nombres": [], "firmante": None,
            "firmante_cargo": None,
            "orgs_externas": [],
        }

        # Firmante
        if tipo == "RESOLUCIÓN":
            m = self.RE_FIRMANTE_RES.search(text)
        else:
            m = self.RE_FIRMANTE.search(text)

        if m:
            result["firmante"] = m.group(1).strip()
            # Cargo: siguiente línea no vacía (más tolerante v4)
            rest = text[m.end():m.end() + 250].strip().splitlines()
            for line in rest[:6]:
                line = line.strip()
                if line and self.RE_CARGO.search(line) and 4 < len(line) < 100:
                    result["firmante_cargo"] = line
                    break

        # PARA y DE (solo no-Resolución)
        if tipo != "RESOLUCIÓN":
            m = self.RE_PARA.search(text)
            if m:
                result["para_nombres"], result["para_cargos"] = \
                    self._parse_nombres_cargos(m.group(1))
            m = self.RE_DE.search(text)
            if m:
                result["de_nombres"], _ = self._parse_nombres_cargos(m.group(1))

        return result

    def _parse_nombres_cargos(self, block: str) -> tuple[list[str], list[str]]:
        nombres: list[str] = []
        cargos: list[str] = []
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        i = 0
        while i < len(lines):
            line = lines[i]
            m = self.RE_NOMBRE_TRAT.search(line)
            if m:
                nombres.append(m.group(0).strip())
                if i + 1 < len(lines):
                    nxt = lines[i + 1]
                    if (self.RE_CARGO.search(nxt)
                            and not self.RE_NOMBRE_TRAT.search(nxt)
                            and len(nxt) < 100):
                        cargos.append(nxt)
                        i += 2
                        continue
            i += 1
        return nombres, cargos


# ===========================================================================
# SECCIÓN 5 — Catálogos controlados
# ===========================================================================

class _Catalogs:
    """Catálogos ecuatorianos centralizados para EntityRuler y fuzzy matching."""

    ORGS = [
        # Ministerios
        "Ministerio de Relaciones Exteriores y Movilidad Humana",
        "Ministerio de Educación", "Ministerio de Salud Pública",
        "Ministerio de Economía y Finanzas", "Ministerio del Interior",
        "Ministerio de Defensa Nacional",
        "Ministerio de Telecomunicaciones y de la Sociedad de la Información",
        "Ministerio de Ambiente, Agua y Transición Ecológica",
        "Ministerio de Agricultura y Ganadería",
        "Ministerio de Producción, Comercio Exterior, Inversiones y Pesca",
        "Ministerio de Transporte y Obras Públicas",
        "Ministerio de Desarrollo Urbano y Vivienda",
        "Ministerio de Trabajo", "Ministerio de Cultura y Patrimonio",
        "Ministerio de Turismo",
        # Secretarías
        "Secretaría Nacional de Planificación",
        "Secretaría Nacional de Gestión Pública",
        "Secretaría General de la Presidencia",
        "Secretaría de Educación Superior, Ciencia, Tecnología e Innovación",
        # Entidades de control
        "Contraloría General del Estado",
        "Procuraduría General del Estado",
        "Defensoría del Pueblo",
        "Consejo Nacional Electoral",
        "Tribunal Contencioso Electoral",
        "Corte Nacional de Justicia",
        "Corte Constitucional",
        "Consejo de la Judicatura",
        "Fiscalía General del Estado",
        "Superintendencia de Compañías, Valores y Seguros",
        "Superintendencia de Bancos",
        "Superintendencia de Control del Poder de Mercado",
        "Superintendencia de Economía Popular y Solidaria",
        # Entidades especializadas
        "Servicio de Rentas Internas",
        "Instituto Ecuatoriano de Seguridad Social",
        "Banco Central del Ecuador",
        "Servicio Nacional de Contratación Pública",
        "Dirección Nacional de Registros Públicos",
        "Consejo de Participación Ciudadana y Control Social",
        "Agencia de Regulación y Control Hidrocarburífero",
        "Agencia de Regulación y Control Minero",
        "Servicio Nacional de Aduana del Ecuador",
        # Registros Mercantiles
        "Registro Mercantil Quito", "Registro Mercantil Ambato",
        "Registro Mercantil Guayaquil", "Registro Mercantil Cuenca",
    ]

    # Siglas institucionales reconocidas
    SIGLAS_ORGS = [
        "SERCOP", "SRI", "IESS", "BCE", "CGE", "PGE", "CNE", "SENESCYT",
        "SENAGUA", "MAATE", "MIES", "MINEDUC", "MSP", "MDT", "MEF",
        "SBS", "SCVS", "SUPERCIAS", "MEER", "MREMH",
    ]

    CARGOS = [
        "Director General", "Director Nacional", "Director Administrativo",
        "Director Financiero", "Director de Planificación",
        "Director de Talento Humano", "Director de Comunicación Social",
        "Director de Tecnologías de la Información",
        "Directora Nacional", "Directora Administrativa", "Directora Financiera",
        "Directora de Comunicación Social", "Directora de Asesoría Jurídica",
        "Subdirector Nacional", "Subdirectora Nacional",
        "Coordinador General", "Coordinadora General",
        "Coordinador Zonal", "Coordinadora Zonal",
        "Coordinador de Desarrollo Organizacional",
        "Secretario Nacional", "Secretaria Nacional",
        "Registrador Mercantil", "Registradora Mercantil",
        "Inscriptor de Registro Mercantil", "Certificador de Registro Mercantil",
        "Contadora de Registro Mercantil", "Analista Jurídico",
        "Especialista Administrativo Financiero",
        "Director de Asesoría Jurídica", "Directora de Asesoría Jurídica",
        "Director de Protección de la Información",
        "Tesorero", "Tesorera", "Contadora", "Contador",
        "Asistente Administrativo", "Técnico Informático",
        "Jefe Departamental", "Jefa Departamental",
        "Asesor Jurídico", "Asesora Jurídica",
        "Gerente General", "Gerente de Operaciones",
        "Viceministro", "Viceministra", "Ministro", "Ministra",
    ]

    LUGARES = [
        # Capitales provinciales
        "Quito", "Guayaquil", "Cuenca", "Ambato", "Riobamba", "Latacunga",
        "Ibarra", "Loja", "Machala", "Esmeraldas", "Portoviejo", "Manta",
        "Santo Domingo", "Babahoyo", "Milagro", "Azogues", "Guaranda",
        "Tulcán", "Macas", "Tena", "Puyo", "Nueva Loja", "Zamora",
        "Santa Elena", "Santa Rosa",
        # Provincias
        "Pichincha", "Guayas", "Azuay", "Tungurahua", "Chimborazo",
        "Cotopaxi", "Imbabura", "Manabí", "Los Ríos", "Bolívar",
        "Carchi", "Cañar", "El Oro", "Morona Santiago", "Napo",
        "Orellana", "Pastaza", "Sucumbíos", "Zamora Chinchipe", "Galápagos",
        "Santa Elena", "Santo Domingo de los Tsáchilas",
    ]

    ESTADOS = [
        "pendiente", "atendido", "archivado", "en trámite",
        "en proceso", "resuelto", "derivado", "anulado", "recibido", "leído",
    ]


# ===========================================================================
# SECCIÓN 6 — EntityExtractor
# ===========================================================================

class EntityExtractor:
    """Orquesta los extractores de entidad (10 en v4, sin ASUNTO)."""

    def __init__(self, device: str = "cpu", load_beto: bool = True):
        self.device = device
        self.sp = StructuralParser()

        log.info("Cargando spaCy es_core_news_sm…")
        try:
            self.nlp: Language = spacy.load("es_core_news_sm")
        except OSError:
            log.warning("es_core_news_sm no encontrado; usando modelo en blanco.")
            self.nlp = spacy.blank("es")

        self._build_rulers()
        self._build_matchers()

        self.beto = None
        if load_beto:
            try:
                log.info("Cargando BETO NER…")
                self.beto = hf_pipeline(
                    "ner",
                    model="dccuchile/bert-base-spanish-wwm-cased",
                    aggregation_strategy="simple",
                    device=0 if device == "cuda" and torch.cuda.is_available() else -1,
                )
            except Exception as e:
                log.warning("BETO no disponible (%s). Usando solo parser estructural.", e)

        log.info("EntityExtractor listo.")

    # ------------------------------------------------------------------
    # Construcción de componentes spaCy
    # ------------------------------------------------------------------

    def _build_rulers(self):
        """EntityRuler con catálogos controlados — score 1.0 sobre NER."""
        if "entity_ruler" not in self.nlp.pipe_names:
            kw = {"before": "ner"} if "ner" in self.nlp.pipe_names else {}
            ruler = self.nlp.add_pipe("entity_ruler", **kw)
        else:
            ruler = self.nlp.get_pipe("entity_ruler")

        patterns = []

        # Organizaciones exactas del catálogo
        for org in _Catalogs.ORGS:
            patterns.append({"label": "ORG_PUBLICA", "pattern": org})

        # Siglas institucionales
        for sigla in _Catalogs.SIGLAS_ORGS:
            patterns.append({"label": "ORG_PUBLICA", "pattern": sigla})

        # Patrones de prefijo institucional (captura orgs no listadas)
        _ORG_PREFIXES = [
            "Ministerio de", "Dirección Nacional de", "Dirección de",
            "Secretaría de", "Secretaría Nacional de",
            "Subsecretaría de", "Coordinación General de",
            "Coordinación de", "Registro Mercantil de",
            "Instituto Nacional de", "Agencia Nacional de",
            "Agencia de Regulación",
        ]
        for prefix in _ORG_PREFIXES:
            patterns.append({
                "label": "ORG_PUBLICA",
                "pattern": [{"LOWER": t.lower()} for t in prefix.split()]
                           + [{"IS_ALPHA": True, "OP": "+"}],
            })

        # Cargos del catálogo
        for cargo in _Catalogs.CARGOS:
            patterns.append({"label": "CARGO", "pattern": cargo})

        # Lugares
        for lugar in _Catalogs.LUGARES:
            patterns.append({"label": "LUGAR_EC", "pattern": lugar})

        ruler.add_patterns(patterns)

    def _build_matchers(self):
        self.matcher = Matcher(self.nlp.vocab)

        verbos = [
            "suministrar", "informar", "autorizar", "disponer", "remitir",
            "adjuntar", "comunicar", "notificar", "solicitar", "requerir",
            "gestionar", "proceder", "atender", "resolver", "archivar",
            "derivar", "designar", "aprobar", "publicar", "conformar",
            "coordinar", "ejecutar", "cumplir", "verificar", "revisar",
        ]
        self.matcher.add("ACCION", [[{"LEMMA": v, "POS": "VERB"}] for v in verbos])
        self.matcher.add("ANEXO_REF", [
            [{"LOWER": {"IN": ["anexo", "adjunto", "apéndice", "tabla"]}},
             {"IS_DIGIT": True}],
            [{"LOWER": {"IN": ["anexo", "adjunto"]}},
             {"IS_ALPHA": True, "LENGTH": 1}],
        ])

    # ------------------------------------------------------------------
    # Extractores individuales
    # ------------------------------------------------------------------

    def extract_codigo(self, text: str) -> str | None:
        """E9: Código documental — busca primero en cabecera, luego en todo el texto."""
        head = "\n".join(text.splitlines()[:6])
        m = StructuralParser.RE_CODIGO.search(head)
        if m:
            return m.group(1)
        m = StructuralParser.RE_CODIGO.search(text)
        return m.group(1) if m else None

    def extract_fecha(self, text: str) -> str | None:
        """E3: Fecha — cabecera primero, luego fallback al cuerpo."""
        head = "\n".join(text.splitlines()[:12])
        m = StructuralParser.RE_FECHA_HEAD.search(head)
        raw = None
        if m:
            raw = next((g for g in m.groups() if g), None)
        # Fallback: primera fecha larga en el cuerpo
        if not raw:
            m2 = StructuralParser.RE_FECHA_BODY.search(text)
            if m2:
                raw = m2.group(1)
        if not raw:
            return None
        parsed = dateparser.parse(
            raw, languages=["es"],
            settings={"PREFER_DAY_OF_MONTH": "first", "RETURN_AS_TIMEZONE_AWARE": False},
        )
        return parsed.date().isoformat() if parsed else raw

    _RE_NORM = re.compile(
        r"(?:"
        r"[Aa]rt(?:ículo)?\.?\s*\d+(?:\.\d+)*(?:\s+(?:literal|numeral|inciso)\s+\w+)?"
        r"|Resolución\s+(?:Nro?\.N°\.?|No\.?)\s*[\w\-]+"
        r"|Acuerdo\s+Ministerial\s+(?:Nro?\.N°\.?)?\s*[\w\-]+"
        r"|Decreto\s+Ejecutivo\s+(?:No\.?|Nro?\.)?\s*\d+"
        r"|Ley\s+Orgánica\s+[\w\s]{5,60}?(?=\s*[-,;\n])"
        r"|Código\s+Orgánico\s+[\w\s]{3,50}?(?=\s*[-,;\n])"
        r"|Código\s+del\s+Trabajo"
        r"|Oficio\s+(?:Nro?\.N°\.?|No\.?)\s*[\w\-]+"
        r"|Memorando\s+(?:Nro?\.N°\.?|No\.?)\s*[\w\-]+"
        r"|Reglamento\s+General\s+[\w\s]{3,50}?(?=\s*[-,;\n])"
        r"|Reglamento\s+a\s+la\s+Ley\s+[\w\s]{3,50}?(?=\s*[-,;\n])"
        r")",
        re.IGNORECASE,
    )

    def extract_referencias(self, text: str) -> list[str]:
        """E4: Referencias normativas y documentales."""
        seen: set[str] = set()
        refs: list[str] = []
        for m in self._RE_NORM.finditer(text):
            val = " ".join(m.group(0).split())
            if val not in seen and len(val) > 5:
                seen.add(val)
                refs.append(val)
        return refs

    def extract_estado(self, text: str) -> str | None:
        """E5: Estado Quipux por fuzzy matching."""
        r = fuzz_process.extractOne(
            text.lower(), _Catalogs.ESTADOS,
            score_cutoff=_CONFIG.ESTADO_SCORE_MIN,
        )
        return r[0] if r else None

    _RE_ANEXOS_PIE = re.compile(
        r"^ANEXOS?\s*:?\s*\n((?:\s*-\s*.+\n?)+)",
        re.MULTILINE | re.IGNORECASE,
    )

    def extract_anexos(self, text: str, doc) -> list[dict]:
        """E11: Anexos referenciados en cuerpo y pie de documento."""
        anexos: list[dict] = []
        seen: set[str] = set()
        for match_id, start, end in self.matcher(doc):
            if self.nlp.vocab.strings[match_id] == "ANEXO_REF":
                label = doc[start:end].text
                if label not in seen:
                    seen.add(label)
                    anexos.append({"tipo": "cuerpo", "etiqueta": label, "pagina_ref": None})
        m = self._RE_ANEXOS_PIE.search(text)
        if m:
            for item in re.findall(r"-\s*(.+)", m.group(1)):
                anexos.append({"tipo": "pie", "etiqueta": item.strip(), "pagina_ref": None})
        return anexos

    def extract_accion(self, doc) -> tuple[str | None, str | None]:
        """E10: Primera acción requerida detectada por Matcher."""
        for match_id, start, end in self.matcher(doc):
            if self.nlp.vocab.strings[match_id] == "ACCION":
                return doc[start:end].text, None
        return None, None

    # ------------------------------------------------------------------
    # Extractor principal de personas, orgs, cargos, lugares (E1,E2,E7,E8)
    # ------------------------------------------------------------------

    def extract_from_structure(
        self, parsed: dict, text: str, doc
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        personas: set[str] = set()
        orgs: set[str] = set()
        cargos: set[str] = set()
        lugares: set[str] = set()

        # — Capa 1: parser estructural (posición fija en el documento) —
        for n in parsed.get("para_nombres", []) + parsed.get("de_nombres", []):
            personas.add(n.strip())
        if parsed.get("firmante"):
            personas.add(parsed["firmante"].strip())
        for c in parsed.get("para_cargos", []):
            cargos.add(c.strip())
        if parsed.get("firmante_cargo"):
            cargos.add(parsed["firmante_cargo"].strip())

        # Orgs externas de FallbackParser
        for org in parsed.get("orgs_externas", []):
            orgs.add(org.strip())

        # — Capa 1.5: búsqueda contextual de nombres con tratamiento —
        # Busca en todo el texto (no solo cabecera) pares tratamiento+nombre
        # que el StructuralParser pudo haber omitido (p.ej. delegados en cuerpo)
        for m in StructuralParser.RE_NOMBRE_TRAT.finditer(text):
            candidato = m.group(0).strip()
            # Evitar fragmentos de cargos etiquetados erróneamente
            if len(candidato) >= _CONFIG.MIN_PERSONA_LEN:
                personas.add(candidato)

        # — Capa 2: EntityRuler (catálogo controlado, score ~1.0) —
        for ent in doc.ents:
            if ent.label_ == "ORG_PUBLICA":
                orgs.add(ent.text.strip())
            elif ent.label_ == "CARGO":
                cargos.add(ent.text.strip())
            elif ent.label_ == "LUGAR_EC":
                lugares.add(ent.text.strip())
            # spaCy PER nativo — solo si supera longitud mínima
            elif ent.label_ == "PER" and len(ent.text) >= _CONFIG.MIN_PERSONA_LEN:
                if not ent.text.isupper():
                    personas.add(ent.text.strip())

        # — Capa 3: BETO sobre cabecera ampliada (≤800 chars) —
        if self.beto:
            try:
                head = text[:_CONFIG.BETO_HEAD_CHARS]
                for ent in self.beto(head):
                    if ent["score"] < _CONFIG.BETO_SCORE_MIN:
                        continue
                    word = ent["word"].strip()
                    # Limpiar subwords de BERT
                    if "##" in word or not word:
                        continue
                    # Filtrar tokens cortos y todo-mayúsculas (siglas/ruido)
                    if len(word) < 5 or word.isupper():
                        continue
                    if ent["entity_group"] in ("PER", "PERSON"):
                        # Filtrar stoplist de falsos positivos geográficos/protocolares
                        if word.lower() not in _CONFIG.BETO_PER_STOPLIST:
                            if any(c.islower() for c in word):
                                personas.add(word)
                    elif ent["entity_group"] == "ORG" and len(word) > _CONFIG.MIN_ORG_LEN:
                        orgs.add(word)
            except Exception as e:
                log.warning("BETO error: %s", e)

        # — Ciudad posicional (línea de fecha) —
        m_ciudad = re.search(
            r"^([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ\.]+)*)"
            r"(?:,\s*D\.?M\.?)?,\s*\d{1,2}\s+de",
            text, re.MULTILINE,
        )
        if m_ciudad:
            lugares.add(m_ciudad.group(1).strip())

        # — Limpieza y deduplicación —
        personas = {
            p for p in personas
            if len(p) >= _CONFIG.MIN_PERSONA_LEN and not p.isupper()
        }
        # Eliminar personas que sean substrings de otra persona más completa
        personas = self._dedup_substrings(personas)

        orgs = {o for o in orgs if len(o) >= _CONFIG.MIN_ORG_LEN}
        cargos = {c for c in cargos if len(c) >= _CONFIG.MIN_CARGO_LEN}

        return sorted(personas), sorted(orgs), sorted(cargos), sorted(lugares)

    @staticmethod
    def _dedup_substrings(names: set[str]) -> set[str]:
        """Elimina nombres que son substring de otro nombre en el set."""
        result = set()
        sorted_names = sorted(names, key=len, reverse=True)
        for name in sorted_names:
            if not any(name != other and name in other for other in sorted_names):
                result.add(name)
        return result

    def extract_all(self, text: str, tipo: str, parsed: dict) -> QuipuxMetadata:
        log.info("Extrayendo entidades (tipo=%s, %d chars)…", tipo, len(text))
        # spaCy sobre los primeros 100k chars para no saturar memoria
        doc = self.nlp(text[:100_000])

        personas, orgs, cargos, lugares = self.extract_from_structure(parsed, text, doc)
        accion, accion_clase = self.extract_accion(doc)
        anexos = self.extract_anexos(text, doc)

        return QuipuxMetadata(
            tipo_documento=tipo,
            codigo_documental=self.extract_codigo(text),
            fecha_iso=self.extract_fecha(text),
            lugar=lugares,
            personas=personas,
            organizaciones=orgs,
            cargos=cargos,
            referencias_normativas=self.extract_referencias(text),
            estado=self.extract_estado(text),
            accion_requerida=accion,
            accion_clase=accion_clase,
            anexos=anexos,
        )


# ===========================================================================
# SECCIÓN 7 — MetadataEncoder
# ===========================================================================

class MetadataEncoder:
    def save(self, doc: QuipuxDocument, output_path: Path):
        output_path.write_text(doc.to_json(), encoding="utf-8")
        log.info("JSON guardado en %s", output_path)


# ===========================================================================
# SECCIÓN 8 — QuipuxPipeline (Singleton de modelos pesados)
# ===========================================================================

class QuipuxPipeline:
    """
    PDFProcessor → Classifier → StructuralParser|FallbackParser
    → EntityExtractor → Encoder

    Singleton implícito: instanciar una vez y reutilizar para lotes.
    """

    def __init__(self, device: str = "cpu", load_beto: bool = True):
        self.pdf = PDFProcessor()
        self.clf = DocumentClassifier()
        self.sp = StructuralParser()
        self.fp = FallbackParser()
        self.extractor = EntityExtractor(device=device, load_beto=load_beto)
        self.encoder = MetadataEncoder()

    def process(self, pdf_path: Path) -> QuipuxDocument:
        text = self.pdf.extract_text(pdf_path)
        tipo, procesable, motivo = self.clf.classify(text)

        if procesable:
            parsed = self.sp.parse(text, tipo)
        else:
            log.warning(
                "Documento no Quipux nativo (tipo=%s). Extracción parcial. %s",
                tipo, motivo,
            )
            parsed = self.fp.parse(text)

        metadata = self.extractor.extract_all(text, tipo, parsed)
        metadata.procesable = procesable
        metadata.motivo_rechazo = motivo

        return QuipuxDocument(fuente=str(pdf_path), metadata=metadata)

    def process_and_save(
        self,
        pdf_path: Path,
        output_path: Path | None = None,
    ) -> QuipuxDocument:
        doc = self.process(pdf_path)
        if output_path is None:
            output_path = pdf_path.with_suffix(".json")
        self.encoder.save(doc, output_path)
        return doc


# ===========================================================================
# SECCIÓN 9 — CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Quipux NER Extractor v4")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-beto", action="store_true")
    args = parser.parse_args()

    if not args.pdf.exists():
        log.error("No encontrado: %s", args.pdf)
        sys.exit(1)

    pipeline = QuipuxPipeline(device=args.device, load_beto=not args.no_beto)
    doc = pipeline.process_and_save(args.pdf, args.output)
    print(doc.to_json())


if __name__ == "__main__":
    main()