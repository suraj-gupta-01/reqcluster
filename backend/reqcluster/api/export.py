from fastapi import APIRouter, Response

from reqcluster.api import PathId
from reqcluster.export import export as run_export

router = APIRouter()


@router.get("/sessions/{sid}/export/{fmt}")
def export(sid: PathId, fmt: str) -> Response:
    data, media, filename = run_export(sid, fmt)
    return Response(data, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})
