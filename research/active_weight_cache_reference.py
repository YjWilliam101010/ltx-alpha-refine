# SPDX-License-Identifier: Apache-2.0
# EXPERIMENTAL REFERENCE ONLY. See README.md in this directory before use.
"""Experiment: exact streamed active-weight cache; no retraining or reordered arithmetic."""
from pathlib import Path
import os,json,hashlib,time,threading
import torch
def install(record, root, release):
 from ltx_core.loader.streaming_scheduler import StreamingScheduler
 from ltx_core.loader.streaming_delivery import Staged,Readiness,active_nbytes,active_views
 mode=os.environ.get('LTX_ALPHA_WEIGHT_CACHE','off')
 if mode=='off':return
 assert mode in ['build','read']
 root=Path(root);root.mkdir(exist_ok=True)
 fingerprint=hashlib.sha256((release/'config/code-manifest.json').read_bytes()+(release/'config/weights-manifest.json').read_bytes()+b'fp8-cast;alpha=1;step=1;bf16').hexdigest()
 manifest_path=root/'manifest.json'
 if mode=='build':
  assert not manifest_path.exists(),'Do not overwrite completed cache'
  manifest={'state':'building','fingerprint':fingerprint,'units':{},'started_at':time.time()}
 else:
  manifest=json.loads(manifest_path.read_text());assert manifest['state']=='complete' and manifest['fingerprint']==fingerprint and len(manifest['units'])==48
  for item in manifest['units'].values():
   p=root/item['file'];st=p.stat();assert st.st_size==item['bytes'] # Full SHA and current file identity checked by release entry; archive extraction may round mtime.
 if mode=='read' and os.environ.get('LTX_ALPHA_TRIM_FACTORS')=='1':
  from ltx_core.loader.state_loader import StateLoader
  from ltx_core.loader.units import MetadataUnit,TensorsUnit
  from dataclasses import replace
  cached_keys={row[0] for item in manifest['units'].values() for row in item['signature']}
  load_factors=StateLoader.load_factors
  trim_stats={'omitted_factor_elements':0,'retained_factor_elements':0,'adapters':0}
  record['cached_factor_elision']=trim_stats
  def only_uncached_factors(self,loras,device,source):
   loaded=[]
   for adapter in loras:
    keys={name for target,pair in adapter.bindings.items() if target not in cached_keys for name in [pair.a,pair.b]}
    selected={name:value for name,value in adapter.unit.metadata.items() if name in keys}
    for name,(shape,dtype) in adapter.unit.metadata.items():
     import math
     trim_stats['retained_factor_elements' if name in keys else 'omitted_factor_elements']+=math.prod(shape)
    trim_stats['adapters']+=1
    if selected:
     subset=replace(adapter,unit=MetadataUnit(str(adapter.unit.unit_id)+':resident-only',selected))
     loaded.append(load_factors(self,[subset],device,source)[0])
    else:loaded.append(TensorsUnit(adapter.unit.unit_id,{'__cached_no_live_factors__':torch.empty(0)}))
   return tuple(loaded)
  StateLoader.load_factors=only_uncached_factors
  from ltx_core.loader.lora import LoRAFuser
  original_patch=LoRAFuser.patch_into
  def assert_no_cached_refusion(self,spec,*args,**kwargs):
   assert not cached_keys.intersection(spec.clean.unit.metadata),'Cached weights must never request factor fusion'
   return original_patch(self,spec,*args,**kwargs)
  LoRAFuser.patch_into=assert_no_cached_refusion
 stats={'mode':mode,'reads':0,'builds':0,'stage_seconds':0.,'build_seconds':0.,'bytes':0,'fingerprint':fingerprint}
 record['active_weight_cache']=stats
 def signature(spec):
  return [[k,list(shape),str(dtype)] for k,(shape,dtype) in spec.clean.unit.metadata.items()]
 base_deliver=StreamingScheduler._deliver
 base_stage=StreamingScheduler._stage
 def dump(self,staged,cancel):
  delivered=base_deliver(self,staged,cancel);uid=staged.uid
  if uid not in manifest['units']:
   tick=time.perf_counter();spec=self.units[self.indices[uid]];n=active_nbytes(spec)
   delivered.completion.wait_host()
   buffer=self._device.buffer(delivered.device_index)
   offset=min(t.data_ptr() for t in delivered.views.values())-buffer.data_ptr()
   assert offset>=0 and offset+n<=buffer.numel()
   with torch.inference_mode():
    packed=buffer.narrow(0,offset,n).cpu()
   data=memoryview(packed.numpy());path=root/(str(self.indices[uid]).zfill(2)+'.bin');temp=path.with_suffix('.tmp')
   with temp.open('wb',buffering=0) as f:
    written=f.write(data);assert written==n
   temp.replace(path);manifest['units'][uid]={'file':path.name,'bytes':n,'sha256':hashlib.sha256(data).hexdigest(),'mtime_ns':path.stat().st_mtime_ns,'signature':signature(spec)}
   if len(manifest['units'])==48:manifest.update(state='complete',finished_at=time.time())
   q=manifest_path.with_suffix('.tmp');q.write_text(json.dumps(manifest,indent=2));q.replace(manifest_path)
   stats['builds']+=1;stats['bytes']+=n;stats['build_seconds']+=time.perf_counter()-tick
  return delivered
 def read(self,uid,cancel):
  tick=time.perf_counter();spec=self.units[self.indices[uid]];item=manifest['units'][uid]
  assert signature(spec)==item['signature'] and active_nbytes(spec)==item['bytes']
  index,previous=self._host.lease(cancel)
  try:
   if previous is not None:previous.wait_host()
   buf=self._host.buffer(index).narrow(0,0,item['bytes'])
   with (root/item['file']).open('rb',buffering=0) as f:
    data=memoryview(buf.numpy());offset=0
    while offset<len(data):
     count=f.readinto(data[offset:]);assert count>0;offset+=count
   stats['reads']+=1;stats['bytes']+=item['bytes'];stats['stage_seconds']+=time.perf_counter()-tick
   return Staged(uid,Readiness.ACTIVE,payload=buf,host_index=index)
  except BaseException:
   self._host.release(index,None);raise
 if mode=='build':StreamingScheduler._deliver=dump
 else:StreamingScheduler._stage=read
