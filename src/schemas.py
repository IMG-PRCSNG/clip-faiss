from pydantic import BaseModel


class ImageInfo(BaseModel):
    filename: str
    width: int
    height: int
    title: str = ""
    caption: str = ""
    copyright: str = ""
