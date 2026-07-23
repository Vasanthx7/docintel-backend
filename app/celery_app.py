from celery import Celery

from .config import settings

celery_app = Celery(
    "docintel",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,  # heavy tasks — don't hoard
    task_acks_late=True,
)
