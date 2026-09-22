"""Standalone comparison UI; never imports or changes the main application."""
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import time
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
from PIL import Image, ImageTk, ImageCms
from engine import Comparison, read_rgb, reduce_local, reduce_siril_preview, preview_levels, write_rgb
if not getattr(sys,'frozen',False):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from runtime_log import append_runtime_log, show_runtime_log, install_log_access

ROOT = Path(sys.executable).resolve().parents[2] if getattr(sys,'frozen',False) else Path(__file__).resolve().parent


class App(tk.Tk):
    def __init__(self, session=None):
        super().__init__()
        self.toolbox_mode = '--toolbox' in sys.argv
        self.title('缩星 · 实验室' if self.toolbox_mode else '星野缩星 · 算法对比原型')
        self.geometry('1440x900'); self.minsize(940,600)
        self.configure(background='#242424')
        style=ttk.Style(self); style.theme_use('clam')
        style.configure('.',background='#2b2b2b',foreground='#ededed',font=('Microsoft YaHei UI',11))
        style.configure('TButton',padding=8,background='#383838')
        style.map('TButton',background=[('active','#505050')])
        style.configure('TEntry',fieldbackground='#383838',foreground='#ededed',insertcolor='#ededed')
        style.configure('Horizontal.TScale',background='#8d8d8d',troughcolor='#191919',borderwidth=0)
        style.configure('TCombobox',fieldbackground='#383838',background='#383838',arrowcolor='#eeeeee')
        style.map('TCombobox',fieldbackground=[('readonly','#383838')],foreground=[('readonly','#eeeeee')])
        self.option_add('*TCombobox*Listbox.background','#383838')
        self.option_add('*TCombobox*Listbox.foreground','#eeeeee')
        self.amount=tk.DoubleVar(value=55); self.mode=tk.StringVar(value='仅 Siril')
        self.view=tk.StringVar(value='三图对比'); self.status=tk.StringVar(value='')
        self.siril=tk.StringVar(value=os.environ.get('METEOR_STAR_SIRIL', r'C:\Program Files\Siril\bin\siril-cli.exe'))
        config_root = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) if os.name == 'nt' else Path(os.environ.get('XDG_CONFIG_HOME', str(Path.home()/'.config')))
        self.config=tk.StringVar(value=str(config_root/'siril/config.1.4.ini'))
        self.events=queue.Queue(); self.running=False; self.engine=None; self.session=None
        self.images={}; self.amounts={}; self.zoom=1.; self.center=(0.,0.); self.drag=None; self.handles=[]
        self.live_pair=None;self.live_tiles={};self.live_key=None;self.live_after=None;self.live_future=None
        self.live_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='star-preview')
        self.live_generation=0
        self.export_directory=None;self.workspaces=[]
        top=ttk.Frame(self,padding=8);top.pack(fill='x')
        self.global_log_button=install_log_access(self,top)
        if self.toolbox_mode:
            self.home_button=self.button(top,'←','返回工具箱',self.close);self.home_button.pack(side='left',padx=(0,8))
        self.open_button=self.button(top,'＋','打开照片',self.open_image);self.open_button.pack(side='left')
        self.export_button=self.button(top,'↓','导出当前强度 · 16 位 TIFF',self.export);self.export_button.pack(side='left',padx=5)
        self.export_button.configure(state='disabled')
        self.fit_button=self.button(top,'□','适合窗口',self.fit);self.fit_button.pack(side='left',padx=5)
        self.actual_button=self.button(top,'1:1','原始像素',lambda:self.set_zoom(1));self.actual_button.pack(side='left')
        self.folder_button=self.button(top,'↗','打开结果目录',self.open_folder);self.folder_button.pack(side='right')
        self.settings_button=self.button(top,'⚙','外部软件设置',self.settings);self.settings_button.pack(side='right',padx=6)
        self.view_combo=ttk.Combobox(top,textvariable=self.view,values=('三图对比','原图','本地方案','Siril'),state='readonly',width=13)
        self.view_combo.pack(side='left',padx=16);self.view_combo.bind('<<ComboboxSelected>>',lambda e:self.draw())
        self.filename=ttk.Label(top,text='');self.filename.pack(side='left',padx=10)
        body=ttk.Frame(self);body.pack(fill='both',expand=True)
        right=ttk.Frame(body,padding=16,width=260);right.pack(side='right',fill='y');right.pack_propagate(False)
        ttk.Label(right,text='缩星强度').pack(anchor='w')
        self.scale=ttk.Scale(right,from_=0,to=100,variable=self.amount,command=self.amount_changed)
        self.scale.pack(fill='x',pady=8)
        self.value_label=ttk.Label(right,text='55%');self.value_label.pack(anchor='e')
        ttk.Label(right,text='导出算法').pack(anchor='w',pady=(10,0))
        self.mode_combo=ttk.Combobox(right,textvariable=self.mode,values=('同时计算','仅本地方案','仅 Siril'),state='readonly')
        self.mode_combo.pack(fill='x',pady=14)
        self.mode_combo.bind('<<ComboboxSelected>>',self.algorithm_changed)
        actions=ttk.Frame(right);actions.pack(fill='x')
        self.run_button=self.button(actions,'▶','生成缩星预览',self.run);self.run_button.pack(side='left')
        self.siril_button=self.button(actions,'↗','在 Siril 中打开工作副本',self.open_siril);self.siril_button.pack(side='left',padx=8)
        self.result_label=ttk.Label(right,text='',wraplength=220);self.result_label.pack(fill='x',pady=20)
        self.canvas=tk.Canvas(body,bg='#171717',highlightthickness=0)
        self.canvas.pack(fill='both',expand=True)
        self.empty_button=ttk.Button(self.canvas,text='打开一张照片',command=self.open_image)
        self.canvas.bind('<Configure>',lambda e:self.draw())
        self.canvas.bind('<MouseWheel>',self.wheel)
        self.canvas.bind('<Button-4>',lambda e:self.wheel(e,1.2))
        self.canvas.bind('<Button-5>',lambda e:self.wheel(e,1/1.2))
        self.canvas.bind('<ButtonPress-1>',self.press)
        self.canvas.bind('<B1-Motion>',self.move)
        self.canvas.bind('<ButtonRelease-1>',lambda e:setattr(self,'drag',None))
        ttk.Label(self,textvariable=self.status,padding=8).pack(fill='x')
        self.protocol('WM_DELETE_WINDOW',self.close)
        self.after(100,self.poll)
        if session:
            self.after(150,lambda:self.load_session(Path(session)))

    def button(self,parent,text,hint,command):
        button=ttk.Button(parent,text=text,command=command,width=3)
        popup=[None]
        def show(event):
            hide()
            popup[0]=tk.Toplevel(self);popup[0].overrideredirect(True)
            ttk.Label(popup[0],text=hint,padding=7).pack()
            popup[0].geometry(f'+{button.winfo_rootx()}+{button.winfo_rooty()+button.winfo_height()+3}')
        def hide(event=None):
            if popup[0]:popup[0].destroy();popup[0]=None
        button.bind('<Enter>',show);button.bind('<Leave>',hide);button.bind('<Destroy>',hide)
        return button

    def pick(self,variable):
        path=filedialog.askopenfilename(parent=self)
        if path:variable.set(path)

    def settings(self):
        window=tk.Toplevel(self);window.title('外部软件设置');window.geometry('640x230')
        panel=ttk.Frame(window,padding=16);panel.pack(fill='both',expand=True)
        for label,variable in (('Siril 命令行程序',self.siril),('Siril 配置文件',self.config)):
            ttk.Label(panel,text=label).pack(anchor='w',pady=(8,0))
            row=ttk.Frame(panel);row.pack(fill='x',pady=6)
            ttk.Entry(row,textvariable=variable).pack(side='left',fill='x',expand=True)
            self.button(row,'…','选择文件',lambda v=variable:self.pick(v)).pack(side='right',padx=6)

    def algorithm_changed(self,event=None):
        self.view.set({'仅 Siril':'Siril','仅本地方案':'本地方案','同时计算':'三图对比'}[self.mode.get()])
        self.draw()

    def amount_changed(self,*args):
        self.value_label.configure(text=f'{self.amount.get():.0f}%')
        if self.live_pair is not None:
            self.result_label.configure(text=f'预览 {self.amount.get():.0f}% · 未保存')
            self.draw()

    def display_image(self,path):
        array,profile=read_rgb(path)
        im=Image.fromarray(np.rint(array*255).astype(np.uint8))
        if profile:
            try:im=ImageCms.profileToProfile(im,ImageCms.ImageCmsProfile(io.BytesIO(profile)),ImageCms.createProfile('sRGB'),outputMode='RGB')
            except (ValueError,OSError):pass
        return im

    def load_session(self,directory):
        self.live_generation+=1;self.live_pair=None;self.live_tiles={};self.live_key=None
        self.session=directory
        meta=json.loads((directory/'input.json').read_text(encoding='utf-8'))
        self.source=Path(meta['source']);self.filename.configure(text=self.source.name)
        self.images={}
        for label,name in (('原图','input.tif'),('本地方案','local-result.tif'),('Siril','siril-result.tif')):
            if (directory/name).is_file():self.images[label]=self.display_image(directory/name)
        comparison=directory/'comparison.json'
        if comparison.exists():
            data=json.loads(comparison.read_text(encoding='utf-8'))
            self.amount.set(data['amount']*100);self.amount_changed()
            self.amounts=data.get('method_amounts',{'本地方案':data['amount'],'Siril':data['amount']})
            self.result_label.configure(text=f"结果强度 {data['amount']*100:.0f}%\nSiril 参数 {data['siril_value']:.4f}")
        self.fit();self.status.set('已载入 · 所有视图同步缩放和平移')
        if (directory/'starless.tif').is_file():
            generation=self.live_generation;events=self.events
            def warm():
                try:
                    original,profile=read_rgb(directory/'input.tif')
                    levels=preview_levels(original,read_rgb(directory/'starless.tif')[0])
                    events.put(('prepared',levels,profile,generation))
                except Exception:
                    import traceback
                    events.put(('live_error',traceback.format_exc()))
            self.live_pool.submit(warm)

    def open_image(self):
        if self.running:return
        path=filedialog.askopenfilename(parent=self,title='打开缩星照片',filetypes=[('照片','*.tif *.tiff *.png *.jpg *.jpeg')])
        if not path:return
        self.running=True
        for button in (self.open_button,self.empty_button,self.run_button):button.configure(state='disabled')
        self.status.set('正在读取照片…')
        def load():
            try:self.events.put(('loaded',Path(path),self.display_image(path)))
            except Exception as exc:
                import traceback
                self.events.put(('load_error',str(exc),traceback.format_exc()))
        threading.Thread(target=load,daemon=True).start()

    def run(self):
        if self.running or not self.images:return
        if not Path(self.siril.get()).is_file() or not Path(self.config.get()).is_file():
            self.status.set('请选择 Siril 命令行程序和已配置 StarNet 的 Siril 配置文件')
            self.settings();return
        if self.session is None:
            workspace=tempfile.TemporaryDirectory(prefix='meteor-star-')
            self.workspaces.append(workspace);self.session=Path(workspace.name)
        self.running=True;self.run_button.configure(state='disabled');self.open_button.configure(state='disabled')
        source,session,siril,config=self.source,self.session,self.siril.get(),self.config.get()
        amount=self.amount.get()/100;mode=self.mode.get()
        generation=self.live_generation
        def work():
            try:
                engine=Comparison(source,session,siril,config);self.engine=engine
                original,starless,profile=engine.prepare(lambda s:self.events.put(('status',s)))
                self.events.put(('prepared',preview_levels(original,starless),profile,generation))
                self.events.put(('ready',))
            except Exception as error:
                import traceback
                detail=traceback.format_exc()
                try:
                    session.mkdir(parents=True,exist_ok=True)
                    (session/'error.log').write_text(detail,encoding='utf-8')
                except OSError:pass
                self.events.put(('error',str(error),detail))
        threading.Thread(target=work,daemon=True).start()

    def export(self):
        if self.running or self.live_pair is None:return
        output=filedialog.askdirectory(parent=self,title='选择缩星结果的保存位置')
        if not output:return
        from datetime import datetime
        destination=Path(output)/datetime.now().strftime('缩星_%Y%m%d_%H%M%S_%f')
        source,session,siril,config=self.source,self.session,self.siril.get(),self.config.get()
        amount=self.amount.get()/100;mode=self.mode.get();pair=self.live_pair
        self.running=True
        for button in (self.run_button,self.open_button,self.export_button):button.configure(state='disabled')
        self.status.set('正在导出 16 位 TIFF…')
        def work():
            try:
                original,starless=pair[0][0];profile=pair[1]
                engine=Comparison(source,session,siril,config)
                destination.mkdir(parents=True,exist_ok=False)
                if mode != '仅 Siril':
                    result=np.empty_like(original)
                    for y in range(0,len(original),256):
                        result[y:y+256]=reduce_local(original[y:y+256],starless[y:y+256],amount)
                    write_rgb(destination/'local-result.tif',result,profile)
                if mode != '仅本地方案':
                    engine.siril_reduce(amount,profile,lambda s:self.events.put(('status',s)))
                    shutil.copyfile(session/'siril-result.tif',destination/'siril-result.tif')
                (destination/'comparison.json').write_text(json.dumps({'source':str(source),'amount':amount,
                    'mode':mode,'siril_value':.5-.45*amount},ensure_ascii=False,indent=2),encoding='utf-8')
                self.events.put(('done',amount,mode,destination))
            except Exception as error:
                import traceback
                self.events.put(('error',str(error),traceback.format_exc()))
        threading.Thread(target=work,daemon=True).start()

    def poll(self):
        try:
            while True:
                event=self.events.get_nowait()
                if event[0]=='live':
                    current=self.preview_key()
                    # Show completed intermediate strengths during a drag. The
                    # single worker preserves order; generation and viewport
                    # checks still reject frames from another photo or view.
                    if event[1][0]==current[0] and event[1][2:]==current[2:]:
                        self.live_key,self.live_tiles=event[1],event[2];self.draw()
                elif event[0]=='live_error':
                    append_runtime_log('缩星预览失败',event[1]);self.status.set('预览失败，请查看运行日志')
                elif event[0]=='prepared':
                    if event[3]!=self.live_generation:continue
                    self.live_pair=(event[1],event[2]);self.live_key=None
                    self.export_button.configure(state='normal')
                    self.algorithm_changed()
                    self.draw()
                elif event[0] in ('loaded','load_error'):
                    self.running=False
                    for button in (self.open_button,self.empty_button,self.run_button):button.configure(state='normal')
                    if event[0]=='loaded':
                        self.source=event[1]
                        self.session=None;self.export_directory=None
                        self.export_button.configure(state='disabled')
                        self.view.set('原图')
                        self.images={'原图':event[2]};self.amounts={};self.error_details=''
                        self.live_generation+=1;self.live_pair=None;self.live_tiles={};self.live_key=None
                        self.run_button.configure(text='▶')
                        self.filename.configure(text=self.source.name);self.result_label.configure(text='')
                        self.fit();self.status.set('已打开照片')
                    else:
                        self.error_details=event[2];self.status.set('打开失败：'+event[1]);self.show_log()
                elif event[0]=='status':
                    self.status.set(event[1][-180:]);append_runtime_log('缩星 · '+event[1])
                elif event[0]=='image':
                    self.images[event[1]]=self.display_image(event[2]);self.amounts[event[1]]=event[3];self.draw()
                elif event[0] in ('ready','done','error'):
                    self.running=False;self.run_button.configure(state='normal');self.open_button.configure(state='normal')
                    self.export_button.configure(state='normal' if self.live_pair is not None else 'disabled')
                    if event[0]=='ready':
                        self.status.set('预览已就绪');self.amount_changed()
                    elif event[0]=='error':
                        self.error_details=event[2];self.status.set(event[1]);self.show_log()
                    else:
                        self.export_directory=event[3]
                        self.result_label.configure(text=f'已保存 · {event[1]*100:.0f}%')
                        self.status.set('处理完成 · 结果已保存为 16 位 TIFF')
                        append_runtime_log(f'缩星保存完成 · {self.export_directory} · {event[2]} · {event[1]*100:.0f}%')
                        if abs(event[1]*100-self.amount.get())>.01:self.amount_changed()
        except queue.Empty:pass
        self.after(100,self.poll)

    def names(self):
        return ('原图','本地方案','Siril') if self.view.get()=='三图对比' else (self.view.get(),)

    def preview_key(self):
        return (self.live_generation,round(self.amount.get(),6),self.zoom,self.center,self.canvas.winfo_width(),self.canvas.winfo_height(),self.names())

    def schedule_live(self):
        if self.live_after is None:
            self.live_after=self.after(60,self.render_live)

    def render_live(self):
        self.live_after=None
        if self.live_pair is None:return
        key=self.preview_key();pair=self.live_pair;names=self.names()
        zoom,center=self.zoom,self.center
        pw=max(1,self.canvas.winfo_width()//len(names));ph=max(1,self.canvas.winfo_height()-30)
        amount=self.amount.get()/100;events=self.events
        if self.live_future is not None:self.live_future.cancel()
        def work():
            try:
                import cv2
                levels,profile=pair;level=0
                while level+1<len(levels) and zoom*2**(level+1)<=1:level+=1
                original,background=levels[level];h,w=original.shape[:2]
                factor=levels[0][0].shape[1]/w;scale=zoom*factor
                left,top=center[0]/factor-pw/scale/2,center[1]/factor-ph/scale/2
                x0,y0=max(0,int(np.floor(left))),max(0,int(np.floor(top)))
                x1,y1=min(w,int(np.ceil(left+pw/scale))),min(h,int(np.ceil(top+ph/scale)))
                tiles={}
                if x1>x0 and y1>y0:
                    size=(max(1,round((x1-x0)*scale)),max(1,round((y1-y0)*scale)))
                    # Process original pixels before resampling. Stripe the ROI
                    # to keep full-frame previews from allocating several float RGB copies.
                    for name in names:
                        if name=='原图':continue
                        process=reduce_siril_preview if name=='Siril' else reduce_local
                        rgb=np.empty((y1-y0,x1-x0,3),np.uint8)
                        for y in range(y0,y1,128):
                            rgb[y-y0:min(y+128,y1)-y0]=np.rint(process(original[y:min(y+128,y1),x0:x1],background[y:min(y+128,y1),x0:x1],amount)*255).astype(np.uint8)
                        image=Image.fromarray(cv2.resize(rgb,size,interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR))
                        if profile:
                            try:image=ImageCms.profileToProfile(image,ImageCms.ImageCmsProfile(io.BytesIO(profile)),ImageCms.createProfile('sRGB'),outputMode='RGB')
                            except (ValueError,OSError):pass
                        tiles[name]=(image,round((x0-left)*scale),30+round((y0-top)*scale))
                events.put(('live',key,tiles))
            except Exception:
                import traceback
                events.put(('live_error',traceback.format_exc()))
        self.live_future=self.live_pool.submit(work)

    def fit(self):
        if not self.images:return
        w,h=next(iter(self.images.values())).size
        self.center=(w/2,h/2)
        self.zoom=min(max(1,self.canvas.winfo_width())/len(self.names())/w,max(1,self.canvas.winfo_height()-30)/h)
        self.draw()

    def set_zoom(self,value):self.zoom=value;self.draw()

    def wheel(self,event,factor=None):
        self.zoom=max(.01,min(8,self.zoom*(factor or (1.2 if event.delta>0 else 1/1.2))))
        self.draw();return 'break'

    def press(self,event):self.drag=(event.x,event.y,*self.center)

    def move(self,event):
        if self.drag:
            x,y,cx,cy=self.drag
            self.center=(cx-(event.x-x)/self.zoom,cy-(event.y-y)/self.zoom);self.draw()

    def draw(self):
        self.canvas.delete('all');self.handles=[]
        if not self.images:
            self.canvas.create_window(self.canvas.winfo_width()/2,self.canvas.winfo_height()/2,window=self.empty_button)
            return
        names=self.names();pw=max(1,self.canvas.winfo_width()//len(names));ph=max(1,self.canvas.winfo_height()-30)
        cx,cy=self.center
        key=self.preview_key()
        if self.live_pair is not None and self.live_key!=key:self.schedule_live()
        compatible=self.live_key is not None and self.live_key[0]==key[0] and self.live_key[2:]==key[2:]
        for index,name in enumerate(names):
            x=index*pw
            title=name+(f' · {self.amounts[name]*100:.0f}%' if name in self.amounts else '')
            if compatible and name in self.live_tiles:title=f'{name} · {self.live_key[1]:.0f}%'
            self.canvas.create_text(x+12,15,text=title,anchor='w',fill='#eeeeee',font=('Microsoft YaHei UI',11))
            if compatible and name in self.live_tiles:
                tile,tx,ty=self.live_tiles[name]
                photo=ImageTk.PhotoImage(tile);self.handles.append(photo)
                self.canvas.create_image(x+tx,ty,image=photo,anchor='nw')
                if index:self.canvas.create_line(x,0,x,ph+30,fill='#555555')
                continue
            im=self.images.get(name)
            if im:
                left,top=cx-pw/self.zoom/2,cy-ph/self.zoom/2
                box=(max(0,left),max(0,top),min(im.width,left+pw/self.zoom),min(im.height,top+ph/self.zoom))
                if box[2]>box[0] and box[3]>box[1]:
                    size=(max(1,round((box[2]-box[0])*self.zoom)),max(1,round((box[3]-box[1])*self.zoom)))
                    tile=im.resize(size,Image.Resampling.BILINEAR,box=box)
                    photo=ImageTk.PhotoImage(tile);self.handles.append(photo)
                    self.canvas.create_image(x+round((box[0]-left)*self.zoom),30+round((box[1]-top)*self.zoom),image=photo,anchor='nw')
            else:self.canvas.create_text(x+pw/2,ph/2,text='尚未计算',fill='#999999')
            if index:self.canvas.create_line(x,0,x,ph+30,fill='#555555')

    def open_folder(self):
        directory=self.export_directory
        if not directory or not directory.exists():return
        if os.name=='nt':os.startfile(directory)
        else:subprocess.Popen(['open' if sys.platform=='darwin' else 'xdg-open',str(directory)])

    def open_siril(self):
        if not self.session or not (self.session/'original.fits').exists():return
        gui=Path(self.siril.get()).with_name('siril.exe' if os.name=='nt' else 'siril')
        if not gui.is_file():self.status.set('未找到 Siril 图形程序');return
        subprocess.Popen([str(gui),'-i',str(self.session/'siril-local.ini'),str(self.session/'original.fits')],
                         creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)

    def show_log(self):
        if getattr(self,'error_details',''):append_runtime_log('缩星',self.error_details)
        show_runtime_log(self)

    def close(self):
        if self.running:
            self.status.set('处理仍在运行，完成后再关闭窗口');return
        self.destroy()

    def destroy(self):
        if self.live_after:
            self.after_cancel(self.live_after);self.live_after=None
        self.live_pool.shutdown(wait=False,cancel_futures=True)
        super().destroy()


if __name__=='__main__':
    if os.name=='nt':
        import ctypes
        try:ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (AttributeError,OSError):pass
    if os.environ.get('STAR_COMPARE_SMOKE_REPORT'):
        from smoke import run
        import traceback
        application=App()
        try:
            result=run(application,Path(os.environ.get('STAR_COMPARE_SMOKE_ROOT',str(ROOT))))
            Path(os.environ['STAR_COMPARE_SMOKE_REPORT']).write_text(json.dumps(result,indent=2),encoding='utf-8')
        except Exception:
            Path(os.environ['STAR_COMPARE_SMOKE_REPORT']).write_text(traceback.format_exc(),encoding='utf-8')
            raise SystemExit(1)
        finally:
            try:application.destroy()
            except tk.TclError:pass
    else:
        session=None if '--toolbox' in sys.argv else Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'runs/galaxy-20260922'
        application=App(session if session is not None and (session/'input.json').exists() else None)
        application.mainloop()
