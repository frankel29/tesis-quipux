"""
quipux_extractor.py  —  v6.1
============================
Extractor NER para documentos Quipux (PDF → JSON enriquecido).

Schema de salida v6.1 (4 bloques estructurados para RAG):
  METADATOS:           tipo_documento, numero_documento, fecha_documento, asunto,
                       institucion_emisora, institucion_destino,
                       firmante, cargo_firmante
  CONTENIDO_FUNCIONAL: objeto_tramite, accion_principal, solicitudes,
                       acciones_requeridas, problemas_juridicos,
                       hechos_relevantes, documentos_mencionados
  NORMATIVA:           normativa_principal, referencias_normativas,
                       articulos_mencionados, fundamentos_juridicos
  ENTIDADES:           personas, instituciones, cargos, fechas_relevantes,
                       referencias_documentales

Cambios v6.1 respecto a v6
---------------------------
  - NormativaBlock: nuevo campo fundamentos_juridicos[FundamentoJuridico]
                    con (cita, articulo, norma, sigla, tema) para indexación
                    en RAG por par artículo+documento (ej. "Art. 71 LOSNCP").
  - _Catalogs.SIGLAS_NORMAS: catálogo sigla→nombre de leyes/códigos EC.
  - EntityExtractor.extract_fundamentos_juridicos: regex compacto
                    "Art. N SIGLA" sobre ASUNTO + cuerpo.
  - SemanticExtractor: LLM enriquece fundamentos con `tema` y selecciona
                       los centrales; valida tipos y deduplica.

Cambios v5 respecto a v4
-------------------------
  - QuipuxMetadata refactorizado a 4 sub-dataclasses anidados
  - StructuralParser: re-añadido ASUNTO + extracción de institucion_emisora
                       e institucion_destino vía fuzzy match contra catálogos
  - EntityExtractor: extract_referencias_split separa normativas/artículos/
                      documentales + identifica normativa_principal
  - EntityExtractor: extract_todas_fechas extrae todas las fechas relevantes
  - SemanticExtractor (NUEVO): LLM API (OpenAI/Azure/Anthropic) para campos
                                 semánticos sin fine-tuning. Activado con
                                 QUIPUX_LOAD_LLM=true.

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
  StructuralParser   extrae bloques posicionales (PARA, DE, firmante, asunto)
  FallbackParser     extrae campos clave-valor de documentos externos
  EntityExtractor    orquesta los extractores de entidad (spaCy + BETO)
  SemanticExtractor  LLM API para campos semánticos (opcional)
  MetadataEncoder    serializa a JSON

Variables de entorno LLM
-------------------------
  QUIPUX_LOAD_LLM        true|false (default false)
  QUIPUX_LLM_PROVIDER    openai|anthropic (default openai)
  QUIPUX_LLM_MODEL       nombre de modelo o deployment Azure (default gpt-4o)
  QUIPUX_LLM_USE_AZURE   true|false — usar AzureOpenAI (default false)
  OPENAI_API_KEY         API key directa OpenAI
  AZURE_OPENAI_ENDPOINT  endpoint Azure
  AZURE_OPENAI_KEY       API key Azure
  AZURE_OPENAI_API_VERSION  versión API Azure (default 2024-02-01)
  ANTHROPIC_API_KEY      API key Anthropic

Uso
---
  python quipux_extractor.py oficio.pdf
  python quipux_extractor.py oficio.pdf --output resultado.json
  python quipux_extractor.py oficio.pdf --no-beto
  python quipux_extractor.py oficio.pdf --llm   (activa SemanticExtractor)
"""

from __future__ import annotations

import json
import os
import re
import sys
import argparse
import logging
import unicodedata
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

    # Regex para filtrar falsos positivos de spaCy PER (títulos, capítulos, términos jurídicos)
    SPACY_PER_NOISE = re.compile(
        r"^(?:Capítulo|Código|Resolución|Oficio|Memorando|Licitación|Contrato"
        r"|Convenio|Reglamento|Artículo|Catálogo|Tercera|Segunda|Primera"
        r"|Edición|Buenas\s+Prácticas|Así\s+también|Señor\s*$|Señora\s*$"
        r"|Señorita\s*$)",
        re.IGNORECASE,
    )

    # Palabras que BETO suele etiquetar erróneamente como PER en docs EC
    BETO_PER_STOPLIST = frozenset({
        "quito", "guayaquil", "cuenca", "ecuador", "quipux", "sercop",
        "señor", "señora", "señorita", "estimado", "presente",
        "consideración", "adjunto", "cordialmente",
        "oficio", "memorando", "resolución", "circular", "informe",
        "vivienda", "decisión", "ministerio", "secretaría", "dirección",
        "anexo", "artículo", "numeral", "literal", "reglamento",
    })


# ===========================================================================
# SECCIÓN 1 — MODELOS DE DATOS
# ===========================================================================

@dataclass
class MetadatosBlock:
    tipo_documento: str | None = None
    numero_documento: str | None = None
    fecha_documento: str | None = None
    asunto: str | None = None
    institucion_emisora: str | None = None
    institucion_destino: str | None = None
    firmante: str | None = None
    cargo_firmante: str | None = None


@dataclass
class ContenidoFuncionalBlock:
    objeto_tramite: str | None = None
    accion_principal: str | None = None
    solicitudes: list[str] = field(default_factory=list)
    acciones_requeridas: list[str] = field(default_factory=list)
    problemas_juridicos: list[str] = field(default_factory=list)
    hechos_relevantes: list[str] = field(default_factory=list)
    documentos_mencionados: list[str] = field(default_factory=list)


@dataclass
class FundamentoJuridico:
    """
    Vínculo artículo + norma para indexación en RAG.

    Campo `cita` es la forma compacta lista para vectorizar
    ("Art. 71 LOSNCP"). El resto son metadatos para filtros y
    explicación humana.
    """
    cita: str                       # "Art. 71 LOSNCP"
    articulo: str                   # "Art. 71"
    norma: str | None = None        # "Ley Orgánica del Sistema Nacional de Contratación Pública"
    sigla: str | None = None        # "LOSNCP"
    tema: str | None = None         # "Cálculo de multas contractuales"
    prioridad: str | None = None    # "ALTA" | "MEDIA" | "BAJA" — relevancia jurídica frente al problema central; null si solo lo halló el regex


@dataclass
class NormativaBlock:
    normativa_principal: str | None = None
    referencias_normativas: list[str] = field(default_factory=list)
    articulos_mencionados: list[str] = field(default_factory=list)
    fundamentos_juridicos: list[FundamentoJuridico] = field(default_factory=list)


@dataclass
class EntidadesBlock:
    personas: list[str] = field(default_factory=list)
    instituciones: list[str] = field(default_factory=list)
    cargos: list[str] = field(default_factory=list)
    fechas_relevantes: list[str] = field(default_factory=list)
    referencias_documentales: list[str] = field(default_factory=list)


@dataclass
class QuipuxMetadata:
    procesable: bool = True
    motivo_rechazo: str | None = None
    METADATOS: MetadatosBlock = field(default_factory=MetadatosBlock)
    CONTENIDO_FUNCIONAL: ContenidoFuncionalBlock = field(default_factory=ContenidoFuncionalBlock)
    NORMATIVA: NormativaBlock = field(default_factory=NormativaBlock)
    ENTIDADES: EntidadesBlock = field(default_factory=EntidadesBlock)


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
            "asunto": None,
            "institucion_emisora": None,
            "institucion_destino": None,
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

    # Asunto del documento (campo Quipux estructurado)
    RE_ASUNTO = re.compile(
        r"ASUNTO\s*:\s*(.+?)(?=\n\s*\n|\nDe\s+mi\s+consideraci[oó]n"
        r"|\nAl\s+respecto|\nEn\s+atenci[oó]n)",
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
            "asunto": None,
            "institucion_emisora": None,
            "institucion_destino": None,
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

        # Fallback: firma electrónica en página separada (texto de pie rompe RE_FIRMANTE)
        if not result["firmante"]:
            m_df = re.search(
                r"Documento\s+firmado\s+electr[oó]nicamente\s*\n+",
                text, re.IGNORECASE,
            )
            if m_df:
                rest_df = text[m_df.end():m_df.end() + 200]
                m_name = self.RE_NOMBRE_TRAT.search(rest_df[:120])
                if m_name:
                    result["firmante"] = m_name.group(0).strip()
                    for line in rest_df[m_name.end():].splitlines():
                        line = line.strip()
                        if line and 4 < len(line) < 120:
                            result["firmante_cargo"] = line
                            break

        # Asunto (campo Quipux estructurado)
        m = self.RE_ASUNTO.search(text)
        if m:
            raw = " ".join(m.group(1).split())
            # Cortar en el primer indicio del bloque PARA (destinatario)
            # o en un punto que cierre oración seguido de mayúscula.
            # Los lookbehinds excluyen cadenas de abreviaturas (FF.AA., D.M.,
            # EE.UU., U.S.A.) que tienen mayúsculas antes del punto pero NO
            # son fin de oración.
            cut = re.search(
                # Punto + minúscula = fin de oración real.
                # NO cortar si el punto sigue a una mayúscula: descarta
                # cadenas de abreviaturas (FF.AA., D.M., EE.UU.) y acrónimos
                # de leyes (LOSNCP., RGLOSNCP., LOSEP., COA.).
                r'(?<![A-ZÁÉÍÓÚÑ])\.\s+(?=[A-ZÁÉÍÓÚÑ][a-záéíóúñ])'
                r'|[\)\.]\s+(?=Señor[a]?\s+(?:\w+\.?\s+)?[A-ZÁÉÍÓÚÑ])',
                raw
            )
            if cut and cut.start() > 10:
                raw = raw[:cut.start() + 1]
            result["asunto"] = raw

        # PARA y DE (solo no-Resolución)
        if tipo != "RESOLUCIÓN":
            m = self.RE_PARA.search(text)
            if m:
                para_block = m.group(1)
                result["para_nombres"], result["para_cargos"] = \
                    self._parse_nombres_cargos(para_block)
                result["institucion_destino"] = self._extract_institucion(para_block)
            m = self.RE_DE.search(text)
            if m:
                de_block = m.group(1)
                result["de_nombres"], _ = self._parse_nombres_cargos(de_block)
                result["institucion_emisora"] = self._extract_institucion(de_block)

        return result

    def _extract_institucion(self, block: str) -> str | None:
        """
        Identifica la institución dentro de un bloque PARA/DE.
        Estrategia: línea ALL-CAPS no-nombre + fallback fuzzy match contra catálogos.
        """
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        # Estrategia 1: línea ALL-CAPS larga sin tratamiento (típico patrón Quipux)
        for line in lines:
            if (line.isupper() and len(line) > 6
                    and not self.RE_NOMBRE_TRAT.search(line)
                    and not self.RE_CARGO.search(line)):
                return line
        # Estrategia 2: fuzzy match contra catálogos institucionales
        catalog = _Catalogs.ORGS + _Catalogs.SIGLAS_ORGS
        for line in lines:
            if self.RE_NOMBRE_TRAT.search(line) or self.RE_CARGO.search(line):
                continue
            if len(line) < 4:
                continue
            best = fuzz_process.extractOne(line, catalog, score_cutoff=75)
            if best:
                return best[0]
        return None

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

    # Siglas oficiales de instituciones ecuatorianas → nombre completo
    # Usado para inferir institucion_emisora desde el numero_documento
    # cuando el parser estructural no la encontró (ej.: memos internos
    # sin bloque DE explícito).
    SIGLAS_INSTITUCIONES = {
        "SERCOP":      "Servicio Nacional de Contratación Pública",
        "SRI":         "Servicio de Rentas Internas",
        "IESS":        "Instituto Ecuatoriano de Seguridad Social",
        "BCE":         "Banco Central del Ecuador",
        "CGE":         "Contraloría General del Estado",
        "PGE":         "Procuraduría General del Estado",
        "CNE":         "Consejo Nacional Electoral",
        "SENESCYT":    "Secretaría de Educación Superior, Ciencia, Tecnología e Innovación",
        "SENAGUA":     "Secretaría Nacional del Agua",
        "MAATE":       "Ministerio de Ambiente, Agua y Transición Ecológica",
        "MIES":        "Ministerio de Inclusión Económica y Social",
        "MINEDUC":     "Ministerio de Educación",
        "MSP":         "Ministerio de Salud Pública",
        "MDT":         "Ministerio del Trabajo",
        "MEF":         "Ministerio de Economía y Finanzas",
        "SBS":         "Superintendencia de Bancos",
        "SCVS":        "Superintendencia de Compañías, Valores y Seguros",
        "SUPERCIAS":   "Superintendencia de Compañías, Valores y Seguros",
        "MEER":        "Ministerio de Energía y Recursos Naturales No Renovables",
        "MREMH":       "Ministerio de Relaciones Exteriores y Movilidad Humana",
        "DINARP":      "Dirección Nacional de Registros Públicos",
        "MTOP":        "Ministerio de Transporte y Obras Públicas",
        "MIDUVI":      "Ministerio de Desarrollo Urbano y Vivienda",
        "MIPRO":       "Ministerio de Producción, Comercio Exterior, Inversiones y Pesca",
        "MAG":         "Ministerio de Agricultura y Ganadería",
        "MINTUR":      "Ministerio de Turismo",
        "MINDEF":      "Ministerio de Defensa Nacional",
        "MINTEL":      "Ministerio de Telecomunicaciones y de la Sociedad de la Información",
        "SENAE":       "Servicio Nacional de Aduana del Ecuador",
        "ARCSA":       "Agencia Nacional de Regulación, Control y Vigilancia Sanitaria",
        "ANT":         "Agencia Nacional de Tránsito",
        "INEC":        "Instituto Nacional de Estadística y Censos",
        "CFN":         "Corporación Financiera Nacional",
        "CNT":         "Corporación Nacional de Telecomunicaciones",
        "ECU911":      "Servicio Integrado de Seguridad ECU 911",
        "CCFFAA":      "Comando Conjunto de las Fuerzas Armadas",
    }

    # Siglas oficiales de normativa ecuatoriana → nombre completo
    # Usado para construir FundamentoJuridico (cita, sigla, norma)
    SIGLAS_NORMAS = {
        # Constitución
        "CRE":     "Constitución de la República del Ecuador",
        "CPRE":    "Constitución de la República del Ecuador",
        # Códigos orgánicos
        "COA":     "Código Orgánico Administrativo",
        "COIP":    "Código Orgánico Integral Penal",
        "COGEP":   "Código Orgánico General de Procesos",
        "COFJ":    "Código Orgánico de la Función Judicial",
        "COOTAD":  "Código Orgánico de Organización Territorial, Autonomía y Descentralización",
        "COPLAFIP": "Código Orgánico de Planificación y Finanzas Públicas",
        "COPYFP":  "Código Orgánico de Planificación y Finanzas Públicas",
        "COESCCI": "Código Orgánico de la Economía Social de los Conocimientos, Creatividad e Innovación",
        "COMYF":   "Código Orgánico Monetario y Financiero",
        # Códigos
        "CC":      "Código Civil",
        "CT":      "Código del Trabajo",
        "CTRIB":   "Código Tributario",
        # Leyes orgánicas
        "LOSNCP":  "Ley Orgánica del Sistema Nacional de Contratación Pública",
        "LOSEP":   "Ley Orgánica del Servicio Público",
        "LOES":    "Ley Orgánica de Educación Superior",
        "LOEI":    "Ley Orgánica de Educación Intercultural",
        "LOGJCC":  "Ley Orgánica de Garantías Jurisdiccionales y Control Constitucional",
        "LOTAIP":  "Ley Orgánica de Transparencia y Acceso a la Información Pública",
        "LOEPS":   "Ley Orgánica de la Economía Popular y Solidaria",
        "LOPC":    "Ley Orgánica de Participación Ciudadana",
        "LOMH":    "Ley Orgánica de Movilidad Humana",
        "LORTI":   "Ley Orgánica de Régimen Tributario Interno",
        "LOC":     "Ley Orgánica de Comunicación",
        "LOSCCA":  "Ley Orgánica del Consejo de Participación Ciudadana y Control Social",
        "LORHUyA": "Ley Orgánica de Recursos Hídricos, Usos y Aprovechamiento del Agua",
        "LOSNS":   "Ley Orgánica del Sistema Nacional de Salud",
        "LODP":    "Ley Orgánica de Datos Personales",
        # Reglamentos generales (con frecuencia citados como RGLO… o RLO…)
        "RGLOSNCP": "Reglamento General a la Ley Orgánica del Sistema Nacional de Contratación Pública",
        "RLOSNCP":  "Reglamento General a la Ley Orgánica del Sistema Nacional de Contratación Pública",
        "RGLOSEP":  "Reglamento General a la Ley Orgánica del Servicio Público",
        "RLOSEP":   "Reglamento General a la Ley Orgánica del Servicio Público",
        # Leyes ordinarias frecuentes
        "LC":      "Ley de Compañías",
        "LDC":     "Ley Orgánica de Defensa del Consumidor",
        "LODC":    "Ley Orgánica de Defensa del Consumidor",
    }


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
        r"|Resolución\s+(?:Nro?\.N°\.?|No\.?)\s*[\w\-]*\d[\w\-]*"
        r"|Acuerdo\s+Ministerial\s+(?:Nro?\.N°\.?)?\s*[\w\-]*\d[\w\-]*"
        r"|Decreto\s+Ejecutivo\s+(?:No\.?|Nro?\.)?\s*\d+"
        r"|Ley\s+Orgánica\s+[\w\s]{5,60}?(?=\s*[-,;\n])"
        r"|Código\s+Orgánico\s+[\w\s]{3,50}?(?=\s*[-,;\n])"
        r"|Código\s+del\s+Trabajo"
        r"|Oficio\s+(?:Nro?\.N°\.?|No\.?)\s*[\w\-]*\d[\w\-]*"
        r"|Memorando\s+(?:Nro?\.N°\.?|No\.?)\s*[\w\-]*\d[\w\-]*"
        r"|Reglamento\s+General\s+[\w\s]{3,50}?(?=\s*[-,;\n])"
        r"|Reglamento\s+a\s+la\s+Ley\s+[\w\s]{3,50}?(?=\s*[-,;\n])"
        r")",
        re.IGNORECASE,
    )

    # Sub-clasificadores de referencias (aplicados sobre cada hit de _RE_NORM)
    _RE_REF_ART = re.compile(r"^[Aa]rt(?:[ií]culo)?\.?\s*\d+", re.IGNORECASE)
    _RE_REF_DOC = re.compile(
        r"^(?:Oficio|Memorando|Informe|Circular)\s+(?:Nro?\.?|No\.?|N°)",
        re.IGNORECASE,
    )
    _RE_REF_LEY_PRINCIPAL = re.compile(
        r"^(?:Ley\s+Org[aá]nica|C[oó]digo\s+Org[aá]nico|C[oó]digo\s+del\s+Trabajo"
        r"|C[oó]digo\s+Civil|Constituci[oó]n)",
        re.IGNORECASE,
    )

    # ----- Fundamentos jurídicos (Art. N + sigla de norma) -----
    # Pre-ordenado por longitud descendente para evitar prefix-matches:
    # "RGLOSNCP" debe intentar match antes que "LOSNCP".
    _SIGLAS_PATTERN = "|".join(
        sorted(_Catalogs.SIGLAS_NORMAS.keys(), key=len, reverse=True)
    )
    # Captura con prefijo "art/artículo(s)" + conector OPCIONAL.
    # Soporta plural y listas con sigla compartida:
    #   "Art. 71 LOSNCP", "art 71 de la LOSNCP", "artículo 7 LOSEP",
    #   "artículos 70 y 80 de la LOSNCP", "arts. 92, 93 y 94 LOSNCP"
    # group(1) = run de números ("70 y 80"); group(2) = sigla.
    _RE_ART_SIGLA = re.compile(
        rf"\b[Aa]rt(?:s|[ií]culos?)?\.?\s*"
        rf"(\d+(?:\.\d+)*(?:\s*(?:,|y|e)\s*\d+(?:\.\d+)*)*)"
        rf"(?:\s+(?:numeral|literal|inciso)\s+\w+)?"
        rf"\s+(?:de\s+(?:la\s+|los?\s+|el\s+)?|del\s+)?"
        rf"({_SIGLAS_PATTERN})\b",
        re.IGNORECASE,
    )
    # Continuación de lista SIN prefijo "art", conector OBLIGATORIO.
    # Rescata el segundo tramo de "…LOSNCP y 121 del RGLOSNCP".
    # El lookbehind exige que el número sea continuación de enumeración
    # (precedido por "y "/", "/"e ") + conector "de la"/"del" obligatorio.
    # Esto evita falsos positivos con años ("2020 de la LOSNCP") y
    # numerales ("numeral 4 del RGLOSNCP").
    # group(1) = run de números; group(2) = sigla.
    _RE_NUM_SIGLA = re.compile(
        rf"(?<=[,ye]\s)\b(\d+(?:\.\d+)*(?:\s*(?:,|y|e)\s*\d+(?:\.\d+)*)*)"
        rf"\s+(?:de\s+(?:la\s+|los?\s+|el\s+)|del\s+)"
        rf"({_SIGLAS_PATTERN})\b",
        re.IGNORECASE,
    )
    # Separa un run ("70 y 80", "92, 93 y 94") en números individuales.
    _RE_NUM_SPLIT = re.compile(r"\d+(?:\.\d+)*")

    @staticmethod
    def _infer_emisora_from_codigo(codigo: str | None) -> str | None:
        """
        Infiere institucion_emisora del prefijo del numero_documento.
        Itera los segmentos separados por '-' y devuelve la primera
        coincidencia en SIGLAS_INSTITUCIONES.

        Ej.: 'SERCOP-CGAJ-2023-0489-M' → 'Servicio Nacional de
        Contratación Pública'
        Soporta también códigos que empiezan con dígitos:
        '001-CP-DINARP-2025' → 'Dirección Nacional de Registros Públicos'
        """
        if not codigo:
            return None
        for segment in codigo.split("-"):
            seg = segment.strip().upper()
            if seg in _Catalogs.SIGLAS_INSTITUCIONES:
                return _Catalogs.SIGLAS_INSTITUCIONES[seg]
        return None

    @staticmethod
    def _normalize_articulo(val: str) -> str | None:
        """
        Forma canónica de un artículo: 'Art. N [calificador]'.
        Acepta: 'art 71', 'artículo 71', 'Art. 71', 'Art. 303 numeral 4'.
        Devuelve: 'Art. 71', 'Art. 303 numeral 4'.
        """
        m = re.match(
            r"[Aa]rt(?:[ií]culo)?\.?\s*(\d+(?:\.\d+)*)"
            r"(?:\s+(numeral|literal|inciso)\s+(\w+))?",
            val,
            re.IGNORECASE,
        )
        if not m:
            return None
        num = m.group(1)
        qual_kind = m.group(2)
        qual_val = m.group(3)
        if qual_kind:
            return f"Art. {num} {qual_kind.lower()} {qual_val}"
        return f"Art. {num}"

    def extract_referencias_split(
        self, text: str
    ) -> tuple[list[str], list[str], list[str], str | None]:
        """
        E4 (v5): Separa referencias en 3 categorías + identifica normativa principal.
        Retorna: (referencias_normativas, articulos_mencionados,
                   referencias_documentales, normativa_principal)
        """
        normativas: list[str] = []
        articulos: list[str] = []
        documentales: list[str] = []
        seen: set[str] = set()
        articulos_canonicos: set[str] = set()

        for m in self._RE_NORM.finditer(text):
            val = " ".join(m.group(0).split())
            if val in seen or len(val) <= 5:
                continue
            seen.add(val)
            if self._RE_REF_ART.match(val):
                # Normalizar para deduplicar variantes de formato
                # ("Art. 71" / "art 71" / "artículo 71" → "Art. 71")
                canonical = self._normalize_articulo(val) or val
                if canonical not in articulos_canonicos:
                    articulos_canonicos.add(canonical)
                    articulos.append(canonical)
            elif self._RE_REF_DOC.match(val):
                documentales.append(val)
            else:
                normativas.append(val)

        # normativa_principal = primera ley/código de jerarquía superior
        normativa_principal: str | None = None
        for ref in normativas:
            if self._RE_REF_LEY_PRINCIPAL.match(ref):
                normativa_principal = ref
                break

        return normativas, articulos, documentales, normativa_principal

    def extract_fundamentos_juridicos(
        self, text: str, asunto: str | None,
    ) -> list[FundamentoJuridico]:
        """
        Extrae pares artículo + norma en formato compacto ("Art. N SIGLA")
        listos para indexar en RAG.

        Prioridad por fuente (heurística, sin LLM):
          - ASUNTO Quipux  -> MEDIA. El redactor coloca aquí las citas que
            considera relevantes; es señal de importancia del autor.
          - Cuerpo del documento -> BAJA. Cita contextual / secundaria.
        El LLM puede luego sobreescribir con ALTA y completar `tema`; el
        merge en QuipuxPipeline conserva la entrada del LLM (dedup por
        (articulo, sigla)). Así se evitan prioridades null.

        Solo detecta el patrón compacto con sigla conocida. La forma
        extendida ("Art. 71 de la Ley Orgánica del Sistema…") y el campo
        `tema` los completa el LLM si está habilitado.

        Retorna hasta 5 fundamentos deduplicados por (artículo, sigla).
        """
        found: dict[tuple[str, str], FundamentoJuridico] = {}

        # (texto fuente, prioridad heurística). El ASUNTO se procesa primero;
        # el dedup conserva la primera ocurrencia, de modo que un artículo
        # citado en el asunto Y el cuerpo se queda en MEDIA (señal más fuerte).
        fuentes: list[tuple[str, str]] = []
        if asunto:
            fuentes.append((asunto, "MEDIA"))
        fuentes.append((text, "BAJA"))

        for fuente, prioridad_fuente in fuentes:
            for rx in (self._RE_ART_SIGLA, self._RE_NUM_SIGLA):
                for m in rx.finditer(fuente):
                    sigla = m.group(2).upper()
                    norma = _Catalogs.SIGLAS_NORMAS.get(sigla)
                    # Un match puede traer varios artículos (run "70 y 80")
                    # que comparten la misma sigla.
                    for articulo_num in self._RE_NUM_SPLIT.findall(m.group(1)):
                        articulo = f"Art. {articulo_num}"
                        cita = f"{articulo} {sigla}"
                        key = (articulo, sigla)
                        if key not in found:
                            found[key] = FundamentoJuridico(
                                cita=cita,
                                articulo=articulo,
                                norma=norma,
                                sigla=sigla,
                                tema=None,
                                prioridad=prioridad_fuente,
                            )
                        if len(found) >= 5:
                            return list(found.values())

        return list(found.values())

    # Fechas en el cuerpo (formato largo o numérico)
    _RE_FECHAS_CUERPO = re.compile(
        r"\b(\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
        r"septiembre|octubre|noviembre|diciembre)\s+(?:de\s+)?\d{4})\b"
        r"|\b(\d{1,2}/\d{1,2}/\d{4})\b"
        r"|\b(\d{4}-\d{2}-\d{2})\b",
        re.IGNORECASE,
    )

    def extract_todas_fechas(self, text: str, cap: int = 20) -> list[str]:
        """E3-bis (v5): todas las fechas relevantes del documento como ISO strings."""
        seen: set[str] = set()
        result: list[str] = []
        for m in self._RE_FECHAS_CUERPO.finditer(text):
            raw = next((g for g in m.groups() if g), None)
            if not raw:
                continue
            try:
                parsed = dateparser.parse(
                    raw, languages=["es"],
                    settings={
                        "PREFER_DAY_OF_MONTH": "first",
                        "RETURN_AS_TIMEZONE_AWARE": False,
                    },
                )
                iso = parsed.date().isoformat() if parsed else raw
            except Exception:
                iso = raw
            if iso not in seen:
                seen.add(iso)
                result.append(iso)
            if len(result) >= cap:
                break
        return result

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
            candidato = " ".join(m.group(0).split())  # normaliza saltos de línea
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
            # spaCy PER nativo — normalizar whitespace, requerir 2+ palabras, filtrar ruido
            elif ent.label_ == "PER" and len(ent.text) >= _CONFIG.MIN_PERSONA_LEN:
                clean = " ".join(ent.text.split())
                if (not clean.isupper() and " " in clean
                        and not _CONFIG.SPACY_PER_NOISE.search(clean)):
                    personas.add(clean)

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

        personas, orgs, cargos, _ = self.extract_from_structure(parsed, text, doc)
        accion, _ = self.extract_accion(doc)
        anexos = self.extract_anexos(text, doc)
        normativas, articulos, documentales, normativa_principal = \
            self.extract_referencias_split(text)
        fechas = self.extract_todas_fechas(text)
        fundamentos = self.extract_fundamentos_juridicos(text, parsed.get("asunto"))

        codigo = self.extract_codigo(text)
        # Fallback: si el parser estructural no halló institucion_emisora,
        # inferir del prefijo del código (ej. SERCOP-CGAJ-… → SERCOP).
        emisora = (
            parsed.get("institucion_emisora")
            or self._infer_emisora_from_codigo(codigo)
        )

        return QuipuxMetadata(
            METADATOS=MetadatosBlock(
                tipo_documento=tipo,
                numero_documento=codigo,
                fecha_documento=self.extract_fecha(text),
                asunto=parsed.get("asunto"),
                institucion_emisora=emisora,
                institucion_destino=parsed.get("institucion_destino"),
                firmante=parsed.get("firmante"),
                cargo_firmante=parsed.get("firmante_cargo"),
            ),
            CONTENIDO_FUNCIONAL=ContenidoFuncionalBlock(
                accion_principal=accion,
                documentos_mencionados=[a["etiqueta"] for a in anexos],
                # solicitudes/acciones_requeridas/problemas_juridicos/hechos_relevantes
                # los llena SemanticExtractor en QuipuxPipeline si está habilitado
            ),
            NORMATIVA=NormativaBlock(
                normativa_principal=normativa_principal,
                referencias_normativas=normativas,
                articulos_mencionados=articulos,
                fundamentos_juridicos=fundamentos,
            ),
            ENTIDADES=EntidadesBlock(
                personas=personas,
                instituciones=orgs,
                cargos=cargos,
                fechas_relevantes=fechas,
                referencias_documentales=documentales,
            ),
        )


# ===========================================================================
# SECCIÓN 6b — SemanticExtractor (LLM API para campos semánticos)
# ===========================================================================

class SemanticExtractor:
    """
    Extrae campos semánticos vía LLM API: objeto_tramite, solicitudes,
    acciones_requeridas, problemas_juridicos, hechos_relevantes.

    Activado con QUIPUX_LOAD_LLM=true. Soporta OpenAI directo, Azure OpenAI
    y Anthropic Claude. Si falta la API key o falla la llamada, retorna
    valores vacíos sin propagar excepción.
    """

    _PROMPT_SISTEMA = (
        "Eres un asistente jurídico especializado en documentos administrativos "
        "del sistema Quipux del Ecuador (memorandos, oficios, resoluciones). "
        "Extraes información estructurada en JSON estricto aplicando las reglas "
        "de formato indicadas. No añades texto fuera del JSON."
    )

    _PROMPT_USUARIO = """Del siguiente documento administrativo ecuatoriano extrae esta estructura JSON exacta:

{{
  "accion_principal": "<verbo en infinitivo + objeto/contexto breve>",
  "objeto_tramite": "<sintagma nominal breve centrado en un sustantivo>",
  "solicitudes": ["<solicitud explícita 1>"],
  "acciones_requeridas": ["<acción en infinitivo 1>"],
  "problemas_juridicos": ["<cuestión jurídica 1>"],
  "hechos_relevantes": ["<antecedente o hecho relevante 1>"],
  "fundamentos_juridicos": [
    {{
      "cita": "Art. 71 LOSNCP",
      "articulo": "Art. 71",
      "norma": "Ley Orgánica del Sistema Nacional de Contratación Pública",
      "sigla": "LOSNCP",
      "tema": "Cálculo de multas contractuales",
      "prioridad": "ALTA"
    }}
  ]
}}

=== REGLAS ESTRICTAS POR CAMPO ===

CAMPO: accion_principal
  Definicion: Intención documental u operativa del EMISOR del documento (no del consultante ni destinatario).
  Formato obligatorio: Verbo en INFINITIVO + objeto/contexto breve.
  PROHIBIDO: verbos conjugados sueltos (asumen un estado o decision ya tomada).

  INCORRECTO         -> CORRECTO
  "autoriza"         -> "Autorizar la compra de equipos"
  "solicita"         -> "Solicitar aprobación de informe"
  "remite"           -> "Remitir expediente al departamento jurídico"
  "aprueba"          -> "Aprobar modificación presupuestaria"
  "emite"            -> "Emitir criterio jurídico sobre viabilidad"
  "informa"          -> "Informar sobre estado del proceso de contratación"
  "absuelve consulta sobre X" -> "Absolver consulta sobre X"

CAMPO: objeto_tramite
  Definicion: Tema central o asunto del documento.
  Formato obligatorio: Sintagma nominal (frase corta centrada en un sustantivo).
  PROHIBIDO: lenguaje narrativo, descriptivo extenso o frases introductorias
             del tipo "El documento trata de...", "Es una respuesta a...",
             "Trata sobre...", "Se refiere a...".

  INCORRECTO
    "El documento es una respuesta al memorando que solicita un pronunciamiento jurídico."
  CORRECTO
    "Pronunciamiento jurídico sobre designación de Auditor Interno Bancario"

  INCORRECTO
    "Trata sobre la terminación del contrato del servidor público."
  CORRECTO
    "Terminación de contrato de servidor público"

  INCORRECTO
    "Es un oficio informativo sobre el estado del proceso de contratación."
  CORRECTO
    "Informe de estado de proceso de contratación"

CAMPO: solicitudes
  Acciones explícitas que el emisor solicita al destinatario.

CAMPO: acciones_requeridas
  Pasos concretos en infinitivo que el destinatario debe ejecutar.

CAMPO: problemas_juridicos
  Controversias, cuestiones de derecho o inconsistencias normativas planteadas.

CAMPO: hechos_relevantes
  Definicion: Antecedentes factuales o circunstancias concretas que sustentan
  el contenido del documento. Acontecimientos, no opiniones ni solicitudes.

  Criterios de calidad por hecho:
    - Especifico: incluye referencias concretas (numeros de documento, fechas,
      autoridades involucradas) cuando el texto las provea.
    - Autocontenido: legible sin necesidad de leer el documento completo.
    - Factual: describe algo que ocurrio o existe, no algo que se pide
      ni una controversia juridica.

  Que SI cuenta como hecho relevante:
    - Documentos previos referenciados con su numero y fecha
      (oficios, memorandos, informes, dictamenes, resoluciones, sentencias).
    - Actos administrativos previos que enmarcan el caso
      (decretos, acuerdos ministeriales, resoluciones).
    - Hechos concretos del expediente con fecha o datos especificos.
    - Pronunciamientos institucionales previos sobre la materia.

  Que NO cuenta (va en otro campo):
    - Lo que el emisor pide o solicita          -> solicitudes
    - Cuestiones de derecho o controversias     -> problemas_juridicos
    - Pasos que el destinatario debe ejecutar   -> acciones_requeridas
    - Citas de articulos sueltas                -> fundamentos_juridicos

  Cantidad: extrae TODOS los hechos relevantes que el documento contenga,
  sin tope artificial. No infles con detalles triviales ni omitas hechos
  centrales por brevedad. La calidad es el criterio, no la cantidad.

  INCORRECTO (pocos y genericos):
    ["Hubo una consulta previa", "Existe un dictamen anterior"]
  CORRECTO (especificos y referenciados):
    ["Consulta original presentada mediante Oficio Nro. XXX-2023-001 de 5 de
      marzo de 2023 por el Director Nacional de la entidad consultante",
     "Dictamen previo de la Procuraduria General del Estado en Oficio
      Nro. PGE-12208 de 22 de septiembre de 2017",
     "Reforma normativa introducida por Decreto Ejecutivo No. 550 publicado
      el 31 de agosto de 2022"]

CAMPO: fundamentos_juridicos
  Definicion: Articulos juridicos CENTRALES que sustentan el criterio o
              decision principal del documento. NO toda cita normativa.
  Formato: lista de objetos. MAXIMO 3 items, ordenados por importancia.

  Estructura de cada item:
    - "cita":     formato compacto "Art. N SIGLA" para indexacion RAG.
                  Ej.: "Art. 71 LOSNCP", "Art. 14 COA", "Art. 226 CRE".
    - "articulo": "Art. N" (sin sigla). Ej.: "Art. 71".
    - "norma":    nombre completo y oficial de la ley/codigo/constitucion.
                  Ej.: "Ley Organica del Sistema Nacional de Contratacion Publica".
    - "sigla":    acronimo oficial. Ej.: "LOSNCP", "COA", "COIP", "CRE",
                  "LOSEP", "RGLOSNCP", "CC", "CT".
    - "tema":     que regula ese articulo EN EL CONTEXTO del documento.
                  Frase nominal breve (max 10 palabras).
    - "prioridad": nivel de relevancia juridica del fundamento frente al
                   PROBLEMA JURIDICO PRINCIPAL del tramite.
                   Valores: "ALTA" | "MEDIA" | "BAJA".
                   REGLA DURA: como MAXIMO UN (1) fundamento puede ser
                   "ALTA" — el unico que resuelve el problema central.
                   El resto es "MEDIA" o "BAJA". (Ver PRIORIZACION).

  PRIORIZACION JURIDICA (campo "prioridad"):
    CARDINALIDAD (regla maxima, no negociable): SOLO UN fundamento de toda
    la lista puede llevar "ALTA". Es el que aborda y resuelve el problema
    juridico central. Todos los demas son "MEDIA" (aportan contexto) o
    "BAJA" (menor impacto al problema principal). Si dudas entre dos
    candidatos a ALTA, elige UNO solo (el mas especifico/operativo) y baja
    el otro a "MEDIA". Si ninguno resuelve el conflicto, NO marques ALTA.

    NO priorices por orden de aparicion ni por prominencia en el texto.
    Los articulos citados en los "vistos"/"considerandos" como marco
    institucional (competencia, atribuciones, ambito) aparecen primero y
    muchas veces, pero casi NUNCA son el fundamento central. Ignora esa
    prominencia y evalua la relacion SEMANTICA con el conflicto, ponderando:
      1. problemas_juridicos detectados (el conflicto central a resolver).
      2. tema asociado al propio fundamento.
      3. objeto_tramite.
      4. contexto general del asunto.

    REGLA DE ORO para marcar "ALTA" — aplica este test SIEMPRE:
      Preguntate: "¿Este articulo contiene la REGLA OPERATIVA CONCRETA
      (formula de calculo, procedimiento, condicion de aplicacion o regla
      de decision) que RESUELVE el problema_juridico, o solo HABILITA la
      figura juridica / dice QUIEN tiene competencia / cual es el AMBITO /
      definiciones / principios generales?"
      Si solo habilita una potestad ("habra multas", "podra sancionarse"),
      describe competencia, atribuciones, ambito, objeto, definiciones o
      principios -> NO es ALTA (va MEDIA o BAJA), aunque encabece el
      documento o se repita muchas veces.

    CRITERIO DE ESPECIFICIDAD OPERATIVA (criterio A):
      ALTA exige el COMO concreto del caso: la formula, el procedimiento,
      los plazos, la condicion de aplicacion o la regla de decision
      aplicable al problema central. Un articulo que solo establece QUE
      EXISTE la figura juridica, sin decir COMO aplicarla, es MEDIA aunque
      sea el origen normativo de la figura discutida.

    NUNCA marques ALTA (van BAJA) los articulos cuyo tema sea:
      - Atribuciones / facultades / competencias de un organo
        (p. ej. "Atribuciones del SERCOP").
      - Ambito de aplicacion u objeto/fines de la ley.
      - Definiciones o glosario.
      - Principios generales o fines institucionales.
      - Creacion o estructura organica de una entidad.

    REGLA LEY MARCO vs REGLAMENTO OPERATIVO (criterio B):
      Cuando coexisten en el documento una LEY y su REGLAMENTO regulando
      la misma materia, distingue el rol normativo de cada uno:
        - La LEY normalmente HABILITA la figura juridica (establece que
          existe, fija el marco general): por defecto es MEDIA.
        - El REGLAMENTO normalmente OPERATIVIZA esa figura (formula de
          calculo, procedimiento, plazos, porcentajes, condiciones de
          aplicacion): es el candidato natural a ALTA si contiene la regla
          concreta del caso.
      Aplica este criterio a CUALQUIER par ley + reglamento (no solo
      LOSNCP/RGLOSNCP: tambien LOSEP/RGLOSEP, LOES/RGLOES, LOSNS/RGLOSNS,
      etc.).
      EJEMPLO: si el documento discute COMO se calculan multas y aparecen
      el articulo de la LEY que dice "habra multas por mora" y el del
      REGLAMENTO que da el porcentaje/dia y la base de calculo, el de la
      LEY es MEDIA y el del REGLAMENTO es ALTA.
      Excepcion: si la LEY misma contiene la formula o el procedimiento
      operativo aplicable al caso, entonces la LEY SI es ALTA.

    Niveles:
      - "ALTA":  EXACTAMENTE UNO en toda la lista. Es el UNICO articulo que
                 contiene la regla operativa concreta (formula, procedimiento,
                 condicion o regla de decision) que RESUELVE el problema
                 juridico central. Si hay dos candidatos, gana el mas
                 especifico y el otro pasa a MEDIA. Ante la duda, usa MEDIA.
      - "MEDIA": aporta contexto normativo complementario. Puede haber varios.
      - "BAJA":  referencia secundaria, institucional, competencial, de
                 ambito o introductoria. Puede haber varios.

    EJEMPLO (problema: "inexistencia de figura para agregar dias de
    suspension al plazo total del contrato"):
      - "Art. 10 LOSNCP" (Atribuciones del SERCOP)            -> "BAJA"
        Solo dice QUIEN tiene competencia. NO resuelve el conflicto.
        Aunque encabece los considerandos, NUNCA es ALTA.
      - "Art. 9 LOSNCP" (Objeto / fines del Sistema)          -> "BAJA"
        Es declarativo/institucional, no normativo del caso.
      - "Art. 121 RGLOSNCP" (Suspension del plazo contractual) -> "ALTA"
        Contiene la regla que resuelve directamente el conflicto.
      - "Art. 80 LOSNCP" (Responsable de la administracion)   -> "MEDIA"
        Aplicable al contexto pero no resuelve el nucleo.

  PROHIBIDO:
    - Incluir TODAS las citas mencionadas en el documento.
    - Inventar siglas que no esten oficialmente reconocidas.
    - Dejar "cita" sin sigla si la sigla es conocida.
    - Repetir el mismo (articulo, sigla) en mas de un item.

  EJEMPLO (memorando sobre multas en contratacion publica):
    Un solo ALTA: la LEY habilita la figura (MEDIA) y el REGLAMENTO
    aporta la regla operativa que resuelve el caso (ALTA).
    "fundamentos_juridicos": [
      {{
        "cita": "Art. 292 RGLOSNCP",
        "articulo": "Art. 292",
        "norma": "Reglamento General a la Ley Organica del Sistema Nacional de Contratacion Publica",
        "sigla": "RGLOSNCP",
        "tema": "Formula y porcentaje de calculo de multas por mora",
        "prioridad": "ALTA"
      }},
      {{
        "cita": "Art. 71 LOSNCP",
        "articulo": "Art. 71",
        "norma": "Ley Organica del Sistema Nacional de Contratacion Publica",
        "sigla": "LOSNCP",
        "tema": "Existencia de multas por incumplimiento contractual",
        "prioridad": "MEDIA"
      }}
    ]

  Si no hay ningun articulo central identificable, devuelve [].

=== REGLAS GLOBALES ===
- Responde UNICAMENTE con el JSON. Sin markdown, sin texto adicional.
- Si un campo de lista no aplica usa [].
- Si accion_principal u objeto_tramite no son identificables usa null.
- Cada cadena: máximo 2 oraciones concisas.

TEXTO DEL DOCUMENTO:
---
{texto}
---"""

    _FALLBACK = {
        "accion_principal": None,
        "objeto_tramite": None,
        "solicitudes": [],
        "acciones_requeridas": [],
        "problemas_juridicos": [],
        "hechos_relevantes": [],
        "fundamentos_juridicos": [],
    }

    def __init__(self, provider: str = "openai", use_azure: bool = False):
        self.provider = provider
        self.use_azure = use_azure
        self.client = None
        self._model = None
        self._init_client()

    def _init_client(self):
        if self.provider == "openai":
            try:
                import openai
            except ImportError:
                log.error(
                    "openai no instalado. Agregar 'openai' a requirements.txt "
                    "o desactivar QUIPUX_LOAD_LLM."
                )
                return
            try:
                if self.use_azure:
                    self.client = openai.AzureOpenAI(
                        api_key=os.environ.get("AZURE_OPENAI_KEY"),
                        api_version=os.environ.get(
                            "AZURE_OPENAI_API_VERSION", "2024-02-01"
                        ),
                        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                    )
                    self._model = os.environ.get("QUIPUX_LLM_MODEL", "gpt-4o")
                    log.info("SemanticExtractor: AzureOpenAI listo (deployment=%s)",
                             self._model)
                else:
                    self.client = openai.OpenAI(
                        api_key=os.environ.get("OPENAI_API_KEY"),
                    )
                    self._model = os.environ.get("QUIPUX_LLM_MODEL", "gpt-4o")
                    log.info("SemanticExtractor: OpenAI listo (model=%s)", self._model)
            except Exception as e:
                log.error("SemanticExtractor: error inicializando cliente OpenAI: %s", e)
                self.client = None

        elif self.provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                log.error(
                    "anthropic no instalado. Agregar 'anthropic' a requirements.txt "
                    "o cambiar QUIPUX_LLM_PROVIDER a openai."
                )
                return
            try:
                self.client = anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY")
                )
                self._model = os.environ.get("QUIPUX_LLM_MODEL", "claude-haiku-4-5")
                log.info("SemanticExtractor: Anthropic listo (model=%s)", self._model)
            except Exception as e:
                log.error("SemanticExtractor: error inicializando cliente Anthropic: %s", e)
                self.client = None
        else:
            log.error("SemanticExtractor: provider desconocido '%s'", self.provider)

    def extract(self, text: str, max_chars: int | None = None) -> dict:
        """Llama al LLM y retorna dict con los 5 campos semánticos.

        max_chars controla cuánto texto ve el LLM. 6000 chars solo cubría
        el encabezado + considerandos (preámbulo institucional), dejando
        fuera el análisis jurídico donde se discuten los artículos que
        regulan el conflicto. Default 16000; override con QUIPUX_LLM_MAX_CHARS.
        """
        if self.client is None:
            return dict(self._FALLBACK)

        if max_chars is None:
            max_chars = int(os.getenv("QUIPUX_LLM_MAX_CHARS", "16000"))
        texto_truncado = text[:max_chars]
        prompt = self._PROMPT_USUARIO.format(texto=texto_truncado)

        try:
            if self.provider == "openai":
                resp = self.client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": self._PROMPT_SISTEMA},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    seed=7,  # reproducibilidad best-effort (OpenAI no es
                             # determinista a temp=0 sin seed fijo)
                    response_format={"type": "json_object"},
                    max_tokens=1200,
                )
                raw = resp.choices[0].message.content
            elif self.provider == "anthropic":
                resp = self.client.messages.create(
                    model=self._model,
                    max_tokens=1200,
                    system=self._PROMPT_SISTEMA,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text
            else:
                return dict(self._FALLBACK)

            data = json.loads(raw)
            return {k: data.get(k, self._FALLBACK[k]) for k in self._FALLBACK}

        except Exception as e:
            log.warning("SemanticExtractor: llamada al LLM falló: %s", e)
            return dict(self._FALLBACK)


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

    def __init__(
        self,
        device: str = "cpu",
        load_beto: bool = True,
        load_llm: bool | None = None,
    ):
        self.pdf = PDFProcessor()
        self.clf = DocumentClassifier()
        self.sp = StructuralParser()
        self.fp = FallbackParser()
        self.extractor = EntityExtractor(device=device, load_beto=load_beto)
        self.encoder = MetadataEncoder()

        # SemanticExtractor opcional (LLM API)
        if load_llm is None:
            load_llm = os.getenv("QUIPUX_LOAD_LLM", "false").lower() == "true"
        self.semantic: SemanticExtractor | None = None
        if load_llm:
            provider = os.getenv("QUIPUX_LLM_PROVIDER", "openai")
            use_azure = os.getenv("QUIPUX_LLM_USE_AZURE", "false").lower() == "true"
            try:
                self.semantic = SemanticExtractor(
                    provider=provider, use_azure=use_azure,
                )
            except Exception as e:
                log.warning("SemanticExtractor no disponible: %s", e)

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

        # Enriquecimiento semántico vía LLM (solo si está habilitado y procesable)
        if self.semantic is not None and procesable:
            sem = self.semantic.extract(text)
            metadata.CONTENIDO_FUNCIONAL.accion_principal = sem["accion_principal"]
            metadata.CONTENIDO_FUNCIONAL.objeto_tramite = sem["objeto_tramite"]
            metadata.CONTENIDO_FUNCIONAL.solicitudes = sem["solicitudes"]
            metadata.CONTENIDO_FUNCIONAL.acciones_requeridas = sem["acciones_requeridas"]
            metadata.CONTENIDO_FUNCIONAL.problemas_juridicos = sem["problemas_juridicos"]
            metadata.CONTENIDO_FUNCIONAL.hechos_relevantes = sem["hechos_relevantes"]

            # fundamentos_juridicos: MERGE regex + LLM.
            # - Regex garantiza cobertura de citas centrales (especialmente
            #   las que el redactor puso en el ASUNTO).
            # - LLM aporta `tema` enriquecido y orden por importancia.
            # Dedup por (articulo, sigla). LLM primero para preservar
            # su ranking de relevancia. Cap a 5 para RAG manejable.
            llm_funds = self._validate_fundamentos(sem.get("fundamentos_juridicos"))
            merged: list[FundamentoJuridico] = []
            seen: set[tuple[str, str | None]] = set()
            for f in llm_funds:
                key = (f.articulo, f.sigla)
                if key not in seen:
                    merged.append(f)
                    seen.add(key)
            for f in metadata.NORMATIVA.fundamentos_juridicos:
                key = (f.articulo, f.sigla)
                if key not in seen:
                    merged.append(f)
                    seen.add(key)
            # Regla de negocio (garantia dura): un unico fundamento ALTA.
            # Se aplica ANTES de ordenar para que el ALTA conservado sea el
            # primero en el ranking de relevancia del LLM; los demas ALTA se
            # degradan a MEDIA. Asi el prompt orienta y el guard garantiza.
            self._enforce_single_alta(merged)
            # Reordenar por prioridad juridica: ALTA -> MEDIA -> BAJA -> null.
            # sort estable: dentro de igual prioridad se preserva el orden de
            # merge (ranking del LLM primero, fundamentos solo-regex despues).
            # Cap a 5 DESPUES de ordenar para que ningun ALTA se pierda frente
            # a un BAJA/null. Los solo-regex llegan con prioridad=None (rank 3).
            _rank = {"ALTA": 0, "MEDIA": 1, "BAJA": 2}
            merged.sort(key=lambda f: _rank.get(f.prioridad, 3))
            metadata.NORMATIVA.fundamentos_juridicos = merged[:5]

        return QuipuxDocument(fuente=str(pdf_path), metadata=metadata)

    @staticmethod
    def _enforce_single_alta(funds: list[FundamentoJuridico]) -> None:
        """
        Garantiza que como máximo UN fundamento tenga prioridad ALTA.

        Regla de negocio: solo el fundamento que aborda y resuelve el
        problema jurídico central puede ser ALTA. El prompt ya lo pide,
        pero si el LLM devuelve varios ALTA (alucinación) este guard
        conserva el primero —el de mayor relevancia según el orden de
        entrada (ranking del LLM)— y degrada el resto a MEDIA.

        Modifica la lista in-place. Llamar ANTES de ordenar por prioridad
        para que el ALTA conservado sea el mejor ranqueado.
        """
        visto_alta = False
        for f in funds:
            if f.prioridad == "ALTA":
                if visto_alta:
                    f.prioridad = "MEDIA"
                else:
                    visto_alta = True

    @staticmethod
    def _norm_cmp(s: str) -> str:
        """Normaliza para comparar: sin acentos, minúsculas, espacios colapsados."""
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return " ".join(s.lower().split())

    @staticmethod
    def _resolve_sigla(
        sigla: str | None, norma: str | None
    ) -> tuple[str | None, str | None]:
        """
        Resuelve (sigla, norma) contra el catálogo controlado, que es la
        FUENTE DE VERDAD. Evita siglas alucinadas por el LLM en el RAG.

          1. Si la sigla del LLM ya es oficial -> se usa, y la norma se
             normaliza al nombre canónico del catálogo.
          2. Si no, se intenta recuperar la sigla por el NOMBRE de la norma
             (reverse-lookup sin acentos). Ej.: el LLM pone "COFPP" pero
             norma="Código Orgánico de Planificación..." -> "COPLAFIP".
          3. Si nada coincide, se descarta la sigla (None) y se conserva la
             norma del LLM tal cual.
        """
        catalogo = _Catalogs.SIGLAS_NORMAS
        if sigla and sigla in catalogo:
            return sigla, catalogo[sigla]
        if norma:
            objetivo = QuipuxPipeline._norm_cmp(norma)
            for sig, nombre in catalogo.items():
                if QuipuxPipeline._norm_cmp(nombre) == objetivo:
                    return sig, nombre
        return None, (norma or None)

    @staticmethod
    def _validate_fundamentos(raw) -> list[FundamentoJuridico]:
        """
        Convierte la lista de dicts del LLM en FundamentoJuridico validados.
        Descarta items inválidos. Limita a 3.

        La identidad normativa (sigla + norma) se ancla al catálogo: ninguna
        sigla inventada por el LLM llega a la salida. La `cita` se reconstruye
        a partir del artículo + sigla canónica para mantener consistencia RAG.
        """
        if not isinstance(raw, list):
            return []
        result: list[FundamentoJuridico] = []
        seen: set[tuple[str, str | None]] = set()
        for item in raw[:5]:  # leemos hasta 5, devolvemos hasta 3 válidos
            if not isinstance(item, dict):
                continue
            articulo = item.get("articulo")
            if not (isinstance(articulo, str) and articulo.strip()):
                continue
            articulo = articulo.strip()
            sigla_raw = item.get("sigla") if isinstance(item.get("sigla"), str) else None
            norma_raw = item.get("norma") if isinstance(item.get("norma"), str) else None
            sigla, norma = QuipuxPipeline._resolve_sigla(
                sigla_raw.strip().upper() if sigla_raw else None,
                norma_raw.strip() if norma_raw else None,
            )
            tema = item.get("tema") if isinstance(item.get("tema"), str) else None
            prioridad_raw = item.get("prioridad")
            prioridad = None
            if isinstance(prioridad_raw, str):
                p = prioridad_raw.strip().upper()
                prioridad = p if p in ("ALTA", "MEDIA", "BAJA") else None
            # cita canónica: nunca arrastra una sigla inventada.
            cita = f"{articulo} {sigla}" if sigla else articulo
            key = (articulo, sigla)
            if key in seen:
                continue
            seen.add(key)
            result.append(FundamentoJuridico(
                cita=cita,
                articulo=articulo,
                norma=norma,
                sigla=sigla,
                tema=tema.strip() if tema else None,
                prioridad=prioridad,
            ))
            if len(result) >= 3:
                break
        return result

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
    parser = argparse.ArgumentParser(description="Quipux NER Extractor v5")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-beto", action="store_true")
    parser.add_argument(
        "--llm", action="store_true",
        help="Activa SemanticExtractor (requiere OPENAI_API_KEY o equivalente)",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        log.error("No encontrado: %s", args.pdf)
        sys.exit(1)

    load_llm = True if args.llm else None  # None → respeta env var
    pipeline = QuipuxPipeline(
        device=args.device,
        load_beto=not args.no_beto,
        load_llm=load_llm,
    )
    doc = pipeline.process_and_save(args.pdf, args.output)
    print(doc.to_json())


if __name__ == "__main__":
    main()