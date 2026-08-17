(function(){
  'use strict';

  function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
  function sigmoid(x,ec50,hill){hill=hill||1.2;return 1/(1+Math.pow(ec50/Math.max(x,1e-9),hill))}
  function escapeHtml(value){
    return String(value)
      .replace(/[&<>\"]/g,function(ch){return {'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[ch]})
      .replace(/\x27/g,'&#39;');
  }
  function numericOrDefault(id, fallback){
    var value=Number(document.getElementById(id).value);
    return Number.isFinite(value)?value:fallback;
  }

  var PRESETS={
    mcm_mix:[['vaccine',0.2],['antitoxin',0.25],['antiviral',0.35],['mcm',1.0]],
    vaccine:[['vaccine',0.15],['vaccine',0.25],['vaccine',0.4],['vaccine',0.9]],
    antitoxin:[['antitoxin',0.2],['antitoxin',0.3],['antitoxin',0.45],['antitoxin',0.95]],
    antiviral:[['antiviral',0.25],['antiviral',0.4],['antiviral',0.55],['antiviral',1.0]],
    general:[['general',0.2],['general',0.35],['general',0.5],['general',1.0]]
  };
  var BASE={
    vaccine:{fpd:40,rate:60,amp:70,stv:50,tri:55},
    antitoxin:{fpd:35,rate:55,amp:50,stv:40,tri:45},
    antiviral:{fpd:12,rate:20,amp:15,stv:10,tri:14},
    mcm:{fpd:5,rate:10,amp:8,stv:4,tri:6},
    general:{fpd:25,rate:40,amp:45,stv:30,tri:35}
  };

  function generate(nComp,nConc,seed,preset,includeToxic){
    var rng=mulberry32(seed|0), template=PRESETS[preset]||PRESETS.mcm_mix, rows=[], concs=[];
    for(var i=0;i<nConc;i++) concs.push(Math.pow(10,-2+i*(4/Math.max(nConc-1,1))));
    for(var c=0;c<nComp;c++){
      var pair=template[Math.min(c,template.length-1)], kind=pair[0], tox=pair[1];
      if(includeToxic && c===nComp-1) tox=1;
      var b=BASE[kind]||BASE.general, jitter=function(v){return v*(0.85+rng()*0.3)};
      var p={fpd:jitter(b.fpd),rate:jitter(b.rate),amp:jitter(b.amp),stv:jitter(b.stv),tri:jitter(b.tri)};
      var name=kind.charAt(0).toUpperCase()+kind.slice(1)+'_'+String.fromCharCode(65+c);
      for(var w=0;w<3;w++) rows.push({compound:name,concentration_uM:0,well:'V'+(w+1),vehicle:true,fpd_ms:280+rng()*10-5,beat_rate_bpm:55+rng()*4-2,amplitude_uv:180+rng()*14-7,stv:0.04+rng()*0.01,triangulation_proxy:0.18+rng()*0.03,noise_sd_uv:6+rng()*4,n_electrodes:8+Math.floor(rng()*4),beat_detection_rate:0.9+rng()*0.08});
      concs.forEach(function(conc,ci){
        for(var w=0;w<3;w++){
          var fe=sigmoid(conc,p.fpd)*tox, re=sigmoid(conc,p.rate)*tox*0.7, ae=sigmoid(conc,p.amp)*tox*0.8, se=sigmoid(conc,p.stv)*tox, te=sigmoid(conc,p.tri)*tox*0.9;
          rows.push({compound:name,concentration_uM:conc,well:'W'+(ci*3+w+1),vehicle:false,fpd_ms:280*(1+0.25*fe)+rng()*12-6,beat_rate_bpm:55*(1-0.3*re)+rng()*5-2.5,amplitude_uv:180*(1-0.4*ae)+rng()*16-8,stv:0.04+0.12*se+rng()*0.016,triangulation_proxy:0.18+0.25*te+rng()*0.04,noise_sd_uv:7+rng()*6,n_electrodes:6+Math.floor(rng()*5),beat_detection_rate:0.75+rng()*0.2});
        }
      });
    }
    return rows;
  }

  function parseCsvLine(line){
    var cells=[], current='', quoted=false;
    for(var i=0;i<line.length;i++){
      var ch=line[i];
      if(ch==='"'){
        if(quoted && line[i+1]==='"'){current+='"';i++;}
        else quoted=!quoted;
      } else if(ch===',' && !quoted){cells.push(current);current='';}
      else current+=ch;
    }
    if(quoted) throw new Error('CSV contains an unterminated quoted field.');
    cells.push(current); return cells;
  }

  function parseCsv(text){
    var lines=text.replace(/^\uFEFF/,'').trim().split(/\r?\n/).filter(function(line){return line.trim()});
    if(lines.length<2) throw new Error('CSV must contain a header and at least one data row.');
    var hdr=parseCsvLine(lines[0]).map(function(h){return h.trim().toLowerCase().replace(/\u03bc/g,'u')});
    var required=['compound','well','concentration_um','vehicle','fpd_ms','beat_rate_bpm','amplitude_uv','stv','triangulation_proxy','noise_sd_uv','n_electrodes','beat_detection_rate'];
    var missing=required.filter(function(name){return hdr.indexOf(name)<0});
    if(missing.length) throw new Error('CSV is missing required scientific column(s): '+missing.join(', '));
    var rows=[];
    for(var i=1;i<lines.length;i++){
      var cols=parseCsvLine(lines[i]);
      if(cols.length!==hdr.length) throw new Error('CSV row '+(i+1)+' has '+cols.length+' fields; expected '+hdr.length+'.');
      var get=function(name){return cols[hdr.indexOf(name)].trim()};
      var veh=get('vehicle').toLowerCase();
      if(['true','1','yes','false','0','no'].indexOf(veh)<0) throw new Error('Invalid vehicle value on CSV row '+(i+1));
      var nums={};
      ['concentration_um','fpd_ms','beat_rate_bpm','amplitude_uv','stv','triangulation_proxy','noise_sd_uv','n_electrodes','beat_detection_rate'].forEach(function(name){
        var value=Number(get(name));
        if(!Number.isFinite(value)) throw new Error('Invalid numeric value in '+name+' on CSV row '+(i+1));
        nums[name]=value;
      });
      if(nums.concentration_um<0) throw new Error('concentration_uM cannot be negative on CSV row '+(i+1));
      rows.push({compound:get('compound'),concentration_uM:nums.concentration_um,well:get('well'),vehicle:['true','1','yes'].indexOf(veh)>=0,fpd_ms:nums.fpd_ms,beat_rate_bpm:nums.beat_rate_bpm,amplitude_uv:nums.amplitude_uv,stv:nums.stv,triangulation_proxy:nums.triangulation_proxy,noise_sd_uv:nums.noise_sd_uv,n_electrodes:nums.n_electrodes,beat_detection_rate:nums.beat_detection_rate});
    }
    return rows;
  }

  function applyQc(rows){
    var kept=[],log=[];
    rows.forEach(function(r){
      var ok=r.n_electrodes>=4 && r.noise_sd_uv<=25 && r.beat_detection_rate>=0.7;
      if(ok) kept.push(r); else log.push('Rejected '+escapeHtml(r.compound)+' '+escapeHtml(r.well)+' (noise/elec/bdr)');
    });
    log.push('QC: kept '+kept.length+'/'+rows.length+' wells'); return {rows:kept,log:log};
  }

  function mean(values){return values.length?values.reduce(function(s,x){return s+x},0)/values.length:NaN}
  function normalizeValue(val,dir,th){
    var excess=dir==='abs'?Math.max(0,Math.abs(val)-th):dir==='inc'?Math.max(0,val-th):(dir==='dec'?Math.max(0,-val-th):0);
    return Math.min(1,excess/(3*(th>0?th:1)));
  }

  function scoreRows(rows,weights,thLow,thMod){
    var byCompound={};
    rows.forEach(function(r){
      if(!byCompound[r.compound])byCompound[r.compound]=[];
      byCompound[r.compound].push(r);
    });
    var results=[];
    Object.keys(byCompound).sort().forEach(function(name){
      var rowsForCompound=byCompound[name];
      var vehicleRows=rowsForCompound.filter(function(r){return r.vehicle});
      var treatedRows=rowsForCompound.filter(function(r){return !r.vehicle});
      if(!vehicleRows.length||!treatedRows.length)return;
      var v={fpd:mean(vehicleRows.map(function(r){return r.fpd_ms})),rate:mean(vehicleRows.map(function(r){return r.beat_rate_bpm})),amp:mean(vehicleRows.map(function(r){return r.amplitude_uv})),stv:mean(vehicleRows.map(function(r){return r.stv})),tri:mean(vehicleRows.map(function(r){return r.triangulation_proxy}))};
      var byConc={};
      treatedRows.forEach(function(r){var key=String(r.concentration_uM);if(!byConc[key])byConc[key]=[];byConc[key].push(r)});
      var concentrationEffects=[];
      Object.keys(byConc).forEach(function(key){
        var group=byConc[key];
        concentrationEffects.push({concentration_uM:Number(key),fpd:mean(group.map(function(r){return 100*(r.fpd_ms-v.fpd)/v.fpd})),rate:mean(group.map(function(r){return 100*(r.beat_rate_bpm-v.rate)/v.rate})),amp:mean(group.map(function(r){return 100*(r.amplitude_uv-v.amp)/v.amp})),stv:mean(group.map(function(r){return (r.stv-v.stv)/Math.max(Math.abs(v.stv),1e-6)})),tri:mean(group.map(function(r){return (r.triangulation_proxy-v.tri)/Math.max(Math.abs(v.tri),1e-6)})});
      });
      if(!concentrationEffects.length)return;
      var endpoints={fpd_change_pct:Math.max.apply(null,concentrationEffects.map(function(e){return Math.abs(e.fpd)})),beat_rate_change_pct:Math.max.apply(null,concentrationEffects.map(function(e){return Math.abs(e.rate)})),amplitude_change_pct:Math.min.apply(null,concentrationEffects.map(function(e){return e.amp})),stv_increase:Math.max.apply(null,concentrationEffects.map(function(e){return e.stv})),triangulation_proxy:Math.max.apply(null,concentrationEffects.map(function(e){return e.tri}))};
      var defs=[['fpd_change_pct',weights.fpd,10,'abs'],['beat_rate_change_pct',weights.rate,15,'abs'],['amplitude_change_pct',weights.amp,20,'dec'],['stv_increase',weights.stv,0.15,'inc'],['triangulation_proxy',weights.tri,0.20,'inc']];
      var sum=0,tw=0,contribs=[];
      defs.forEach(function(d){var c=normalizeValue(endpoints[d[0]],d[3],d[2]);sum+=d[1]*c;tw+=d[1];contribs.push({name:d[0],raw:endpoints[d[0]],c:c,w:d[1]})});
      var score=Math.min(1,Math.max(0,tw?sum/tw:0));
      results.push({compound:name,score:score,cls:score<thLow?'Low':(score<thMod?'Moderate':'High'),maxConc:Math.max.apply(null,concentrationEffects.map(function(e){return e.concentration_uM})),nWells:treatedRows.length,nConcentrations:concentrationEffects.length,contribs:contribs});
    });
    return results.sort(function(a,b){return b.score-a.score});
  }

  var lastResults=null,lastQc=[],uploaded=null;
  function render(results,qcLog){
    lastResults=results;lastQc=qcLog||[];
    var qc=document.getElementById('qc'),ul=document.getElementById('qcList');
    if(lastQc.length){qc.style.display='block';ul.textContent='';lastQc.forEach(function(l){var li=document.createElement('li');li.textContent=l;ul.appendChild(li)})}else qc.style.display='none';
    var html='<div class="card"><h3>Summary — MCM / biodefense prioritization</h3><p class="meta">Browser mode mirrors Python well-level concentration aggregation; biological-unit inference and 4PL evidence remain Python-only.</p><table><thead><tr><th>Candidate</th><th>CardioScore</th><th>Risk</th><th>Max uM</th><th>Concentrations</th><th>Technical wells</th></tr></thead><tbody>';
    results.forEach(function(r){html+='<tr><td><strong>'+escapeHtml(r.compound)+'</strong></td><td>'+r.score.toFixed(3)+'</td><td><span class="badge badge-'+escapeHtml(r.cls)+'">'+escapeHtml(r.cls)+'</span></td><td>'+r.maxConc.toFixed(2)+'</td><td>'+r.nConcentrations+'</td><td>'+r.nWells+'</td></tr>'});
    html+='</tbody></table></div>';
    results.forEach(function(r){
      html+='<div class="card"><h3>'+escapeHtml(r.compound)+' <span class="badge badge-'+escapeHtml(r.cls)+'">'+escapeHtml(r.cls)+'</span></h3><p class="meta" style="margin:0 0 8px">CardioScore '+r.score.toFixed(3)+'</p>';
      r.contribs.forEach(function(c){var w=Math.max(2,Math.round(c.c*140));html+='<div class="bar-wrap"><div style="width:170px;font-size:12px">'+escapeHtml(c.name)+'</div><div class="bar" style="width:'+w+'px"></div><span class="meta">'+c.c.toFixed(3)+' (w='+c.w+')</span></div>'});
      html+='</div>';
    });
    document.getElementById('out').innerHTML=html;
  }
  function weights(){return{fpd:numericOrDefault('w_fpd',0.3),rate:numericOrDefault('w_rate',0.15),amp:numericOrDefault('w_amp',0.15),stv:numericOrDefault('w_stv',0.25),tri:numericOrDefault('w_tri',0.15)}}

  document.getElementById('csvFile').addEventListener('change',function(e){var f=e.target.files&&e.target.files[0];if(!f){uploaded=null;return}var reader=new FileReader();reader.onload=function(){try{uploaded=parseCsv(String(reader.result));alert('Loaded '+uploaded.length+' validated rows')}catch(err){uploaded=null;alert(err.message)}};reader.readAsText(f)});
  document.getElementById('run').onclick=function(){var rows=uploaded&&uploaded.length?uploaded:generate(numericOrDefault('nComp',4),numericOrDefault('nConc',6),numericOrDefault('seed',42),document.getElementById('preset').value,document.getElementById('includeToxic').checked);var qc=applyQc(rows);render(scoreRows(qc.rows,weights(),numericOrDefault('th_low',0.3),numericOrDefault('th_mod',0.6)),qc.log)};
  document.getElementById('clear').onclick=function(){document.getElementById('out').innerHTML='';document.getElementById('qc').style.display='none';lastResults=null};
  function download(name,text,type){var a=document.createElement('a');var url=URL.createObjectURL(new Blob([text],{type:type}));a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(function(){URL.revokeObjectURL(url)},1000)}
  document.getElementById('exportJson').onclick=function(){if(!lastResults){alert('Run scoring first');return}download('cardioscore_results.json',JSON.stringify({qc:lastQc,scores:lastResults},null,2),'application/json')};
  document.getElementById('exportCsv').onclick=function(){if(!lastResults){alert('Run scoring first');return}var lines=['compound,cardioscore,risk_class,max_concentration_uM,n_concentrations,n_technical_wells'];lastResults.forEach(function(r){lines.push([JSON.stringify(r.compound),r.score.toFixed(4),r.cls,r.maxConc,r.nConcentrations,r.nWells].join(','))});download('cardioscore_summary.csv',lines.join('\n'),'text/csv')};
  document.getElementById('run').click();
})();
