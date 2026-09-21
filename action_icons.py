"""Original line icons and non-modal action hints shared by every workspace."""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk


TOOLS = {
    '流星批量筛选': 'search', '星空对齐': 'align', '流星合成功能': 'layers',
    '视频动态': 'film', 'Siril + PTGui 控制点': 'nodes', '白平衡与改机校准': 'temperature',
    '光污染渐变校正': 'horizon', '星轨叠加': 'trails', '已对齐降噪': 'stack', '批量画质体检': 'chart',
}
TOGGLES = ('画笔', '橡皮擦', '查看原图', '原图精细预览', '候选标记', '取样', '最终效果', '来源标注', '保留音频', '拖框保护')


def action_key(text):
    plain = str(text).strip(' …')
    if plain in TOOLS:
        return TOOLS[plain]
    if plain == '设置':return 'gear'
    for words, key in (
        (('切换工具',), 'grid'),
        (('工具箱', '← 流星工具'), 'home'), (('返回',), 'back'),
        (('打开导出文件夹', '打开结果', '打开文件夹'), 'folder'),
        (('使用对齐结果',), 'export'),
        (('自动对齐图',), 'align'), (('原始状态图',), 'original'),
        (('批量应用',), 'batch'), (('拖框保护',), 'protect'),
        (('流星起点', '流星终点'), 'brush'),
        (('收起',), 'collapse'), (('展开',), 'expand'),
        (('取消', '关闭', '不是流星', '排除', '丢弃'), 'cross'),
        (('清除', '清空', '移除', '删除'), 'trash'),
        (('重做',), 'redo'), (('撤销',), 'undo'),
        (('恢复', '重置'), 'reset'), (('复制', '已复制'), 'copy'),
        (('载入',), 'load'), (('保存',), 'save'),
        (('设置', '参数'), 'sliders'),
        (('导出', '使用对齐结果'), 'export'),
        (('打开结果', '打开文件夹', '文件夹'), 'folder'),
        (('添加', '＋'), 'add'), (('锁定',), 'lock'),
        (('合并',), 'merge'), (('拆分',), 'split'), (('音频',), 'audio'),
        (('手动标记',), 'brush'),
        (('优化全部',), 'spark_all'), (('推荐', '优化'), 'spark'), (('检测全部',), 'search_all'),
        (('分析', '扫描', '检测', '估计'), 'search'),
        (('控制点',), 'nodes'), (('实验',), 'play'),
        (('应用', '完成', '继续', '是流星', '保留', '全选'), 'check'),
        (('橡皮擦',), 'eraser'), (('画笔',), 'brush'), (('取样',), 'dropper'),
        (('原图', '最终效果'), 'eye'), (('候选', '来源标注'), 'nodes'),
        (('适合', '适应'), 'fit'), (('1:1', '100%'), 'actual'),
        (('快捷键',), 'keyboard'), (('历史', '误判记录'), 'history'),
        (('创意放置',), 'move'), (('打开', '选择', '浏览'), 'open'),
    ):
        if any(word in plain for word in words):
            return key
    return {'+': 'plus', '−': 'minus', '◀': 'back', '▶': 'play'}.get(plain, 'sliders')


def icon_image(key, size=22, color='#eeeeee'):
    """Draw at 4x resolution; no symbol font or third-party artwork required."""
    image = Image.new('RGBA', (96, 96))
    d = ImageDraw.Draw(image)
    def line(points, width=1.7):
        d.line([(round(x*4), round(y*4)) for x,y in points], fill=color, width=round(width*4), joint='curve')
    def box(bounds, radius=0):
        coords=tuple(round(n*4) for n in bounds)
        d.rounded_rectangle(coords, radius=round(radius*4), outline=color, width=6)
    def oval(bounds):
        d.ellipse(tuple(round(n*4) for n in bounds), outline=color, width=6)
    def arrow(up=False):
        end=4 if up else 15
        line([(12,20 if up else 3),(12,end)])
        line([(8,end+(4 if up else -4)),(12,end),(16,end+(4 if up else -4))])
    if key in ('search','search_all'):
        oval((3,3,16,16));line([(15,15),(21,21)])
        if key=='search_all':line([(5,9),(14,9)]);line([(9.5,5),(9.5,14)])
    elif key in ('open','folder'):
        line([(3,19),(3,6),(9,6),(11,9),(21,9),(18,19),(3,19)])
        if key=='open':line([(7,14),(17,14)]);line([(12,11),(12,17)])
    elif key in ('save','load'):
        box((4,3,20,21),1);box((8,4,16,9));box((8,14,16,20))
        if key=='load':line([(1,12),(6,12),(4,10)]);line([(6,12),(4,14)])
    elif key=='export':
        arrow(True);line([(4,13),(4,21),(20,21),(20,13)])
    elif key=='check':line([(4,12),(9,17),(20,6)],2.1)
    elif key=='cross':line([(5,5),(19,19)]);line([(19,5),(5,19)])
    elif key=='plus' or key=='add':
        line([(5,12),(19,12)]);line([(12,5),(12,19)])
        if key=='add':box((2,2,22,22),2)
    elif key=='minus':line([(5,12),(19,12)])
    elif key=='trash':
        line([(3,6),(21,6)]);line([(9,3),(15,3)]);line([(6,8),(7,21),(17,21),(18,8)])
        line([(10,10),(10,18)]);line([(14,10),(14,18)])
    elif key in ('undo','redo','reset'):
        d.arc((20,24,80,80),210,530,fill=color,width=7)
        line([(3,5),(3,12),(10,12)])
        if key=='redo':image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif key=='home':line([(3,11),(12,3),(21,11),(19,11),(19,21),(14,21),(14,14),(10,14),(10,21),(5,21),(5,11)])
    elif key=='back':line([(12,5),(5,12),(12,19)]);line([(5,12),(21,12)])
    elif key=='play':line([(6,3),(21,12),(6,21),(6,3)])
    elif key=='original':box((3,3,21,21),1);line([(5,18),(10,12),(14,16),(18,9),(21,13)]);oval((6,6,9,9))
    elif key=='batch':
        box((7,7,21,21),1);line([(17,3),(3,3),(3,17)]);line([(10,14),(13,17),(18,11)])
    elif key=='protect':
        line([(12,2),(21,6),(20,14),(12,22),(4,14),(3,6),(12,2)])
        line([(7,12),(10,15),(17,8)])
    elif key in ('collapse','expand'):
        line([(5,15),(12,8),(19,15)] if key=='collapse' else [(5,8),(12,15),(19,8)])
    elif key=='grid':
        for x,y in ((3,3),(14,3),(3,14),(14,14)):box((x,y,x+7,y+7),1)
    elif key=='sliders':
        for x,y in ((5,8),(12,16),(19,6)):
            line([(x,3),(x,y-2)]);line([(x,y+2),(x,21)]);oval((x-2,y-2,x+2,y+2))
    elif key=='gear':
        import math
        oval((6,6,18,18));oval((10,10,14,14))
        for i in range(8):
            angle=i*math.pi/4
            line([(12+6*math.cos(angle),12+6*math.sin(angle)),(12+10*math.cos(angle),12+10*math.sin(angle))],2.5)
    elif key=='audio':
        line([(3,9),(7,9),(13,4),(13,20),(7,15),(3,15),(3,9)])
        d.arc((40,20,80,76),285,435,fill=color,width=6)
    elif key=='copy':box((7,7,21,21),1);line([(17,4),(3,4),(3,17)])
    elif key=='lock':box((4,10,20,21),2);d.arc((28,4,68,64),180,360,fill=color,width=7);line([(12,14),(12,17)])
    elif key in ('brush','dropper','eraser'):
        line([(4,20),(5,14),(16,3),(21,8),(10,19),(4,20)])
        if key=='dropper':line([(12,3),(21,12)])
        elif key=='brush':line([(4,20),(2,22),(9,21)])
        else:line([(8,11),(14,17)]);line([(13,21),(22,21)])
    elif key=='eye':
        line([(2,12),(6,7),(12,5),(18,7),(22,12),(18,17),(12,19),(6,17),(2,12)]);oval((8,8,16,16))
    elif key in ('fit','actual'):
        for x,y,dx,dy in ((3,3,1,1),(21,3,-1,1),(3,21,1,-1),(21,21,-1,-1)):
            line([(x,y+dy*5),(x,y),(x+dx*5,y)])
        if key=='actual':line([(9,9),(11,7),(11,17)]);line([(15,7),(15,17)])
    elif key in ('spark','spark_all','temperature'):
        line([(12,2),(15,9),(22,12),(15,15),(12,22),(9,15),(2,12),(9,9),(12,2)])
        if key=='spark_all':line([(3,1),(3,7)]);line([(0,4),(6,4)])
        if key=='temperature':
            d.polygon([(48,8),(60,36),(88,48),(60,60),(48,88)],fill='#edb94f')
            d.polygon([(48,8),(36,36),(8,48),(36,60),(48,88)],fill='#5c98ed')
    elif key in ('layers','stack'):
        line([(2,8),(12,3),(22,8),(12,13),(2,8)]);line([(2,12),(12,17),(22,12)])
        if key=='stack':line([(2,16),(12,21),(22,16)])
    elif key=='nodes':
        line([(5,6),(18,5),(15,18),(5,6)])
        for x,y in ((5,6),(18,5),(15,18)):oval((x-2,y-2,x+2,y+2))
    elif key=='align':box((3,5,13,17),1);box((11,7,21,19),1);line([(12,2),(12,22)],1)
    elif key=='film':
        box((2,4,22,20),1)
        for x in (5,11,17):line([(x,4),(x,7)]);line([(x,17),(x,20)])
        line([(9,9),(15,12),(9,15),(9,9)])
    elif key=='horizon':
        line([(2,19),(22,19)]);d.arc((16,16,80,80),180,360,fill=color,width=6)
        for x in (5,12,19):line([(x,22),(x,20)])
    elif key=='trails':
        for r in (20,32,44):d.arc((48-r,48-r,48+r,48+r),20,245,fill=color,width=6)
    elif key=='chart':
        line([(3,3),(3,21),(22,21)]);line([(7,18),(7,13)]);line([(12,18),(12,7)]);line([(18,18),(18,10)])
    elif key in ('merge','split'):
        line([(3,4),(8,4),(15,12),(21,12)]);line([(3,20),(8,20),(15,12)]);line([(18,9),(21,12),(18,15)])
        if key=='split':image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif key=='history':oval((3,3,21,21));line([(12,6),(12,12),(17,15)])
    elif key=='keyboard':
        box((2,5,22,19),2)
        for y in (9,13):
            for x in (6,10,14,18):line([(x,y),(x+.5,y)],1.5)
        line([(8,16),(16,16)])
    else:
        line([(12,2),(12,22)]);line([(2,12),(22,12)])
        for x,y in ((12,2),(12,22),(2,12),(22,12)):oval((x-1,y-1,x+1,y+1))
    return image.resize((size,size),Image.Resampling.LANCZOS)


def action_text(widget):
    variable = widget.cget('textvariable')
    return str(widget.getvar(variable) if variable else widget.cget('text')).strip()


def is_icon_action(widget):
    return isinstance(widget, (ttk.Button, ttk.Menubutton)) or (
        isinstance(widget, (ttk.Checkbutton, ttk.Radiobutton)) and
        any(word in action_text(widget) for word in TOGGLES))


def hint_text(widget):
    text = action_text(widget)
    names = {'+': '放大', '−': '缩小', '◀': '上一帧', '▶': '下一帧',
             '1:1': '原始大小', '100%': '原始大小', '取样': '在照片上选取中性参考点'}
    if text in names:
        return names[text]
    return text.replace('B ✎ 画笔', '画笔 · B').replace('E ▱ 橡皮擦', '橡皮擦 · E').lstrip('✓✕×✦↶↷↺←＋ ').removeprefix('1. ').removeprefix('2. ').removeprefix('3. ')


class ActionHint:
    def __init__(self, widget, text=None):
        self.widget, self.popup, self.timer = widget, None, None
        self.text = text
        if text is None:
            widget.bind('<Enter>', lambda e: self.delay(), add=True)
            widget.bind('<FocusIn>', lambda e: self.delay(), add=True)
            widget.bind('<ButtonPress-1>', lambda e: self.show(), add=True)
            widget.bind('<KeyPress-space>', lambda e: self.show(), add=True)
            widget.bind('<KeyPress-Return>', lambda e: self.show(), add=True)
        widget.bind('<Leave>', lambda e: self.hide(), add=True)
        widget.bind('<FocusOut>', lambda e: self.hide(), add=True)
        widget.bind('<Destroy>', lambda e: self.hide() if e.widget is widget else None, add=True)
        widget.bind('<Unmap>', lambda e: self.hide(), add=True)

    def hide(self):
        if getattr(self.widget._root(), '_active_action_hint', None) is self:
            self.widget._root()._active_action_hint = None
        if self.timer:
            try:self.widget.after_cancel(self.timer)
            except tk.TclError:pass
            self.timer=None
        if self.popup:
            try:self.popup.destroy()
            except tk.TclError:pass
            self.popup=None

    def delay(self):
        if self.popup:
            return
        self.hide()
        self.timer=self.widget.after(450,self.show)

    def update_text(self):
        if self.popup:
            text = self.text if self.text is not None else hint_text(self.widget)
            if isinstance(self.widget, ttk.Widget) and self.widget.instate(['disabled']):
                text += '（当前不可用）'
            self.popup.winfo_children()[0].configure(text=text)

    def show(self, position=None):
        self.hide()
        w=self.widget
        if not w.winfo_exists() or not w.winfo_ismapped():return
        previous=getattr(w._root(),'_active_action_hint',None)
        if previous and previous is not self:previous.hide()
        w._root()._active_action_hint=self
        popup=self.popup=tk.Toplevel(w)
        popup.withdraw()
        popup.overrideredirect(True)
        popup.attributes('-topmost',True)
        text=self.text if self.text is not None else hint_text(w)
        if isinstance(w, ttk.Widget) and w.instate(['disabled']):text+='（当前不可用）'
        popup.configure(background='#101010')
        tk.Label(popup,text=text,background='#383838',foreground='#eeeeee',
                 highlightthickness=1, highlightbackground='#515151',
                 padx=9,pady=6,wraplength=280,font='TkDefaultFont').pack(padx=(0, 3), pady=(0, 3))
        popup.update_idletasks()
        anchor_x = w.winfo_rootx() + (position[0] if position else 0)
        anchor_y = w.winfo_rooty() + (position[1] if position else 0)
        x=max(0,min(anchor_x,w.winfo_screenwidth()-popup.winfo_reqwidth()-8))
        # Above the button keeps adjacent controls and the pointer unobstructed.
        y=anchor_y-popup.winfo_reqheight()-6
        if y<0:y=w.winfo_rooty()+w.winfo_height()+6
        popup.geometry(f'+{x}+{y}')
        popup.deiconify()
        self.timer=w.after(1800,self.hide)


def show_canvas_hint(canvas, text, position):
    hint = getattr(canvas, '_canvas_action_hint', None)
    if hint is None:
        hint = canvas._canvas_action_hint = ActionHint(canvas, text)
    hint.text = text
    hint.show(position)


def iconize_actions(parent):
    _iconize_actions(parent)
    from dpi_support import scale_layout
    scale_layout(parent)


def _iconize_actions(parent):
    """Convert actions, not parameter labels; retain hidden labels for access and tests."""
    for widget in (parent,*parent.winfo_children()):
        if widget is not parent:
            _iconize_actions(widget)
    if getattr(parent,'_action_icon',False):return
    if not is_icon_action(parent):return
    is_toggle=isinstance(parent,(ttk.Checkbutton,ttk.Radiobutton))
    parent._action_icon=True
    original=parent.configure
    def refresh():
        hint = getattr(parent, '_action_hint', None)
        if hint:
            hint.update_text()
        key=action_key(action_text(parent))
        from dpi_support import pixels
        size = pixels(parent, 22)
        cache_key = (key, size)
        if getattr(parent,'_action_cache_key',None)==cache_key:return
        parent._action_key=key
        parent._action_cache_key=cache_key
        # Cache in the owning toplevel, so closed workspaces release their images.
        owner=parent.winfo_toplevel()
        if not hasattr(owner,'_action_images'):owner._action_images={}
        if cache_key not in owner._action_images:
            owner._action_images[cache_key]=ImageTk.PhotoImage(icon_image(key, size=size),master=owner)
        original(image=owner._action_images[cache_key],compound='none',width=3)
    refresh()
    # Icon targets keep a consistent size rather than stretching into empty bars.
    if parent.winfo_manager() == 'pack':
        parent.pack_configure(fill='none', expand=False, anchor='w')
    elif parent.winfo_manager() == 'grid':
        parent.grid_configure(sticky='w')
    if is_toggle:
        original(style='Icon.Toolbutton')
    else:
        style='Icon.TMenubutton' if isinstance(parent,ttk.Menubutton) else 'Primary.Icon.TButton' if 'Primary' in str(parent.cget('style')) else 'Icon.TButton'
        original(style=style)
    def configure(cnf=None,**kwargs):
        result=original(cnf,**kwargs)
        if 'text' in kwargs or 'textvariable' in kwargs or isinstance(cnf,dict) and ('text' in cnf or 'textvariable' in cnf):refresh()
        return result
    parent.configure=parent.config=configure
    variable=parent.cget('textvariable')
    if variable:
        callback=parent.register(lambda *_:refresh())
        parent.tk.call('trace','add','variable',variable,'write',callback)
        def remove_trace(event):
            if event.widget is parent:
                try:parent.tk.call('trace','remove','variable',variable,'write',callback)
                except tk.TclError:pass
        parent.bind('<Destroy>',remove_trace,add=True)
    parent._action_hint=ActionHint(parent)
