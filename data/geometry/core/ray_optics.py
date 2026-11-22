"""Specular ray tracing from visible finite mirrors and an incoming direction."""
import math


def trace_mirrors(source, tip, mirrors):
    def cross(a, b):return a[0]*b[1]-a[1]*b[0]
    p=tuple(map(float,source));d=(tip[0]-p[0],tip[1]-p[1]);length=math.hypot(*d);d=(d[0]/length,d[1]/length)
    contacts=[];indices=[]
    # Revisiting a finite mirror is allowed. A closed optical orbit has no
    # outgoing answer ray and is rejected by the source constructor.
    for _ in range(64):
        hits=[]
        for index,(a,b) in enumerate(mirrors):
            v=(b[0]-a[0],b[1]-a[1]);den=cross(d,v)
            if abs(den)<1e-10:continue
            offset=(a[0]-p[0],a[1]-p[1]);t=cross(offset,v)/den;u=cross(offset,d)/den
            if t>1e-6 and 0<=u<=1:hits.append((t,index,v))
        if not hits:return contacts,d,indices
        t,index,v=min(hits);p=(p[0]+t*d[0],p[1]+t*d[1]);contacts.append(p);indices.append(index)
        length=math.hypot(*v);normal=(-v[1]/length,v[0]/length);dot=d[0]*normal[0]+d[1]*normal[1]
        d=(d[0]-2*dot*normal[0],d[1]-2*dot*normal[1])
    raise ValueError('ray remains trapped between mirrors')
