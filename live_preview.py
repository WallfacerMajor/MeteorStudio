"""Single-flight viewport rendering with one replaceable pending snapshot."""
from concurrent.futures import ThreadPoolExecutor


class LivePreview:
    def __init__(self, owner):
        self.owner=owner;self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='live-view')
        self.future=None;self.pending=None;self.key=None;self.timer=None;self.closed=False
        owner.bind('<Destroy>',self.destroy,add=True)
    def destroy(self,event):
        if event.widget is not self.owner:return
        self.closed=True;self.pending=None
        if self.timer:
            try:self.owner.after_cancel(self.timer)
            except Exception:pass
        self.pool.shutdown(wait=False,cancel_futures=True)
    def submit(self,key,work,done,failed):
        if self.closed:return
        self.key=key;self.pending=(key,work,done,failed)
        if self.future is None:self.start()
    def start(self):
        if self.pending is None:return
        job=self.pending;self.pending=None
        self.future=self.pool.submit(job[1]);self.job=job
        self.timer=self.owner.after(16,self.poll)
    def poll(self):
        self.timer=None
        if self.closed:return
        if not self.future.done():
            self.timer=self.owner.after(16,self.poll);return
        future,job=self.future,self.job;self.future=None
        if job[0]==self.key:
            try:result=future.result()
            except Exception as exc:job[3](exc)
            else:job[2](result)
        self.start()


def submit(owner,key,work,done,failed):
    renderer=getattr(owner,'_live_renderer',None)
    if renderer is None:owner._live_renderer=renderer=LivePreview(owner)
    renderer.submit(key,work,done,failed)
