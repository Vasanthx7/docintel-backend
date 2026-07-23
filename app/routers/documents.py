import os

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import Document
from ..schemas import DocumentOut
from ..tasks import ingest_document
from ..vectorstore import delete_document as vs_delete_document

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentOut)
def upload_document(file: UploadFile, db: Session = Depends(get_db)):
    if not file.filename:
        raise HTTPException(400, "No filename")

    doc = Document(filename=file.filename, status="processing")
    db.add(doc)
    db.commit()

    os.makedirs(settings.upload_dir, exist_ok=True)
    dest = os.path.join(settings.upload_dir, f"{doc.id}_{file.filename}")
    with open(dest, "wb") as f:
        f.write(file.file.read())

    # Hand off the slow work to Celery; return immediately.
    ingest_document.delay(doc.id, dest)
    return doc


@router.get("", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    return db.scalars(select(Document).order_by(Document.created_at.desc())).all()


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    return doc


@router.delete("/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    vs_delete_document(document_id)
    db.delete(doc)
    db.commit()
    return {"deleted": document_id}
