"""Lossless spatial coordinates for native surveying-table redraws.

The source-defined map permutes every pixel without colour changes.
It never restores scene content, fits the candidate or looks up
reference pixels.
"""
from contextlib import contextmanager
from pathlib import Path
import numpy as np




class GeometryChart:
    def __init__(self,sample,turns):
        from data.geometry.core.prompt_paraphrase import prompt_paraphrase_style
        size=512
        turns=int(turns)
        if turns not in (0,1,2,3):raise ValueError('quarter turns must be 0, 1, 2 or 3')
        shape=prompt_paraphrase_style(sample.task_name)['candidate_shape']
        # Only the glyph patch stays upright. Rotating a whole marker interior
        # back would tear non-circular polygon boundaries into two poses.
        radius=3 if shape=='triangle' else 4 if shape in ('square','diamond') else 5
        indices=np.rot90(np.arange(size*size,dtype=np.int32).reshape(size,size),turns).copy()
        occupied=np.zeros((size,size),dtype=bool)
        dy,dx=np.mgrid[-radius:radius+1,-radius:radius+1]
        dx,dy=dx.ravel(),dy.ravel()
        sx,sy=dx.copy(),dy.copy()
        for _ in range(turns):sx,sy=sy,-sx
        for item in sample.semantic_input['candidates']:
            x,y=round(item['x']),round(item['y'])
            for _ in range(turns):x,y=y,size-1-x
            if not (radius<=x<size-radius and radius<=y<size-radius):
                raise ValueError('identifier disk crosses the canvas boundary')
            if occupied[y+dy,x+dx].any():raise ValueError('identifier disks overlap')
            before=indices.copy()
            indices[y+dy,x+dx]=before[y+sy,x+sx]
            occupied[y+dy,x+dx]=True
        self.forward=indices.ravel()
        if not np.all(np.bincount(self.forward,minlength=size*size)==1):
            raise ValueError('observation map is not a pixel bijection')
        self.backward=np.empty_like(self.forward)
        self.backward[self.forward]=np.arange(size*size,dtype=np.int32)

    def apply(self,image):
        a=np.asarray(image)
        if a.shape!=(512,512,3):raise ValueError('chart requires the complete 512px RGB canvas')
        return a.reshape(-1,3)[self.forward].reshape(a.shape)

    def undo(self,image):
        a=np.asarray(image)
        return a.reshape(-1,3)[self.backward].reshape(a.shape)


@contextmanager
def observe_candidate(candidate,chart):
    """Reindex decoded observations only; preserve the submitted media itself."""
    import cv2
    target=Path(candidate).resolve()
    original=cv2.VideoCapture
    class Capture:
        def __init__(self,reader):self.reader=reader
        def __getattr__(self,name):return getattr(self.reader,name)
        def read(self,*args):
            ok,frame=self.reader.read(*args)
            if ok and frame.shape==(512,512,3):frame=chart.undo(frame)
            return ok,frame
    def open_capture(path,*args,**kwargs):
        reader=original(path,*args,**kwargs)
        return Capture(reader) if isinstance(path,(str,Path)) and Path(path).resolve()==target else reader
    cv2.VideoCapture=open_capture
    try:yield
    finally:
        cv2.VideoCapture=original
