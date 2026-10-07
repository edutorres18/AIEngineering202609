"""Extracción local del texto de los adjuntos (camino B)."""

import pytest

from app.attachments.extractor import AttachmentError, enrich_transcript, extract_text
from tests.documents import docx_bytes, pdf_bytes


def test_extracts_text_from_a_pdf():
    content = pdf_bytes("Especificacion del proyecto Nimbus", "Pagos con Stripe")

    text = extract_text("spec.pdf", content, max_chars=1000)

    assert text == "Especificacion del proyecto Nimbus\nPagos con Stripe"


def test_extracts_paragraphs_and_tables_from_a_word_document():
    content = docx_bytes(
        ["Propuesta previa", "", "Alcance: contactos y oportunidades"],
        table=[["Modulo", "Horas"], ["Contactos", "40"]],
    )

    text = extract_text("Propuesta.DOCX", content, max_chars=1000)

    assert text.splitlines() == [
        "Propuesta previa",
        "Alcance: contactos y oportunidades",
        "Modulo | Horas",
        "Contactos | 40",
    ]


def test_long_documents_are_truncated():
    content = docx_bytes(["x" * 500])

    assert extract_text("spec.docx", content, max_chars=100) == "x" * 100


@pytest.mark.parametrize("filename", ["diagrama.png", "notas.txt", "antiguo.doc", "sin_extension"])
def test_unsupported_formats_are_rejected_with_415(filename):
    with pytest.raises(AttachmentError) as exc:
        extract_text(filename, b"...", max_chars=1000)

    assert exc.value.status_code == 415
    assert filename in exc.value.message


@pytest.mark.parametrize("filename", ["roto.pdf", "roto.docx"])
def test_damaged_files_are_rejected_with_422(filename):
    with pytest.raises(AttachmentError) as exc:
        extract_text(filename, b"esto no es un documento", max_chars=1000)

    assert exc.value.status_code == 422


def test_transcript_is_followed_by_each_attachment_between_fences():
    enriched = enrich_transcript(
        "  Reunión con el cliente.  ",
        [("spec.pdf", "Pagos con Stripe"), ("vacio.pdf", ""), ("plan.docx", "Fase 1")],
    )

    assert enriched == (
        "Reunión con el cliente.\n\n"
        "--- attachment: spec.pdf ---\nPagos con Stripe\n--- end attachment ---\n\n"
        "--- attachment: plan.docx ---\nFase 1\n--- end attachment ---"
    )


def test_transcript_without_attachments_is_unchanged():
    assert enrich_transcript("Reunión con el cliente.", []) == "Reunión con el cliente."
