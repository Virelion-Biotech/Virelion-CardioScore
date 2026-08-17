(function(){
  function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
  function sigmoid(x,ec50,hill){hill=hill||1.2;return 1/(1+Math.pow(ec50/Math.max(x,1e-9),hill))}
  function escapeHtml(value){
    return String(value).replace(/[&<>'"]/g,function(ch){return {'&':'&amp;','<':'&lt;','>':'&gt;','\'':'&#39;','"':'&quot;'}[ch]});
  }

  var PRESETS={
    mcm_mix:[{kind:'vaccine',tox:0.2},{kind:'antitoxin',tox:0.25},{kind:'antiviral',tox:0.35},{kind:'mcm',tox:1.0}],
    vaccine:[{kind:'vaccine',tox:0.15},{kind:'vaccine',tox:0.25},{kind:'vaccine',tox:0.4},{kind:'vaccine',tox:0.9}],
    antitoxin:[{kind:'antitoxin',tox:0.2},{kind:'antitoxin',tox:0.3},{kind:'antitoxin',tox:0.45},{kind:'antitoxin',tox:0.95}],
    antiviral:[{kind:'antiviral',tox:0.25},{kind:'antiviral',tox:0.4},{kind:'antiviral',tox:0.55},{kind:'antiviral',tox:1.0}],
    general:[{kind:'general',tox:0.2},{kind:'general',tox:0.35},{kind:'general',tox:0.5},{kind:'general',tox:1.0}]
  };

  function profileFor(kind,tox,rng){
    var base={
      vaccine:{fpd:40,rate:60,amp:70,stv:50,tri:55},
      antitoxin:{fpd:35,rate:55,amp:50,stv:40,tri:45},
      antiviral:{fpd:12,rate:20,amp:15,stv:10,tri:14},
      mcm:{fpd:5,rate:10,amp:8,stv:4,tri:6},
      general:{fpd:25,rate:40,amp:45,stv:30,tri:35}
    }[kind]||{fpd:25,rate:40,amp:45,stv:30,tri:35};
    var jitter=function(v){return v*(0.85+rng()*0.3)};
    return {fpd:jitter(base.fpd),rate:jitter(base.rate),amp:jitter(base.amp),stv:jitter(base.stv),tri:jitter(base.tri),tox:tox};
  }

  function generate(nComp,nConc,seed,preset,includeToxic){
    var rng=mulberry32(seed|0);
    var concs=[];
    for(var i=0;i<nConc;i++) concs.push(Math.pow(10,-2+i*(4/(Math.max(nConc-1,1)))));
    var template=PRESETS[preset]||PRESETS.mcm_mix;
    var rows=[];
    for(var c=0;c<nComp;c++){
      var t=template[Math.min(c,template.length-1)];
      var tox=t.tox;
      if(includeToxic && c===nComp-1) tox=1.0;
      var p=profileFor(t.kind,tox,rng);
      var name=(t.kind.charAt(0).toUpperCase()+t.kind.slice(1))+'_'+String.fromCharCode(65+c);
      for(var w=0;w<3;w++){
        rows.push({compound:name,concentration_uM:0,well:'V'+(w+1),vehicle:true,
          fpd_ms:280+rng()*10-5,beat_rate_bpm:55+rng()*4-2,amplitude_uv:180+rng()*14-7,
          stv:0.04+Math.abs(rng()*0.01),triangulation_proxy:0.18+Math.abs(rng()*0.03),
          noise_sd_uv:6+rng()*4,n_electrodes:8+Math.floor(rng()*4),beat_detection_rate:0.9+rng()*0.08});
      }
      concs.forEach(function(conc,ci){
        for(var w=0;w<3;w++){
          var fe=sigmoid(conc,p.fpd)*p.tox, re=sigmoid(conc,p.rate)*p.tox*0.7;
          var ae=sigmoid(conc,p.amp)*p.tox*0.8, se=sigmoid(conc,p.stv)*p.tox, te=sigmoid(conc,p.tri)*p.tox*0.9;
          rows.push({compound:name,concentration_uM:conc,well:'W'+(ci*3+w+1),vehicle:false,
            fpd_ms:280*(1+0.25*fe)+rng()*12-6,beat_rate_bpm:55*(1-0.3*re)+rng()*5-2.5,
            amplitude_uv:180*(1-0.4*ae)+rng()*16-8,stv:0.04+0.12*se+Math.abs(rng()*0.016),
            triangulation_proxy:0.18+0.25*te+Math.abs(rng()*0.04),
            noise_sd_uv:7+rng()*6,n_electrodes:6+Math.floor(rng()*5),beat_detection_rate:0.75+rng()*0.2});
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
    cells.push(current);
    return cells;
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
      var get=function(name){var j=hdr.indexOf(name);return cols[j].trim()};
      var veh=String(get('vehicle')).toLowerCase();
      if(['true','1','yes'].indexOf(veh)<0 && ['false','0','no'].indexOf(veh)<0) throw new Error('Invalid vehicle value on CSV row '+(i+1));
      var numericNames=['concentration_um','fpd_ms','beat_rate_bpm','amplitude_uv','stv','triangulation_proxy','noise_sd_uv','n_electrodes','beat_detection_rate'];
      var parsed={};
      numericNames.forEach(function(name){
        var value=parseFloat(get(name));
        if(!Number.isFinite(value)) throw new Error('Invalid numeric value in '+name+' on CSV row '+(i+1));
        parsed[name]=value;
      });
      if(parsed.concentration_um<0) throw new Error('concentration_uM cannot be negative on CSV row '+(i+1));
      rows.push({
        compound:get('compound'),
        concentration_uM:parsed.concentration_um,
        well:get('well'),
        vehicle:veh==='true'||veh==='1'||veh==='yes',
        fpd_ms:parsed.fpd_ms,
        beat_rate_bpm:parsed.beat_rate_bpm,
        amplitude_uv:parsed.amplitude_uv,
        stv:parsed.stv,
        triangulation_proxy:parsed.triangulation_proxy,
        noise_sd_uv:parsed.noise_sd_uv,
        n_electrodes:parsed.n_electrodes,
        beat_detection_rate:parsed.beat_detection_rate
      });
    }
    return rows;
  }

  function applyQc(rows){
    var log=[], kept=[];
    rows.forEach(function(r){
      var ok=(r.n_electrodes||0)>=4 && (r.noise_sd_uv||0)<=25 && (r.beat_detection_rate||0)>=0.7;
      if(ok) kept.push(r); else log.push('Rejected '+escapeHtml(r.compound)+' '+escapeHtml(r.well)+' (noise/elec/bdr)');
    });
    log.push('QC: kept '+kept.length+'/'+rows.length+' wells');
    return {rows:kept,log:log};
  }

  function norm(val,dir,th){
    var excess=0;
    if(dir==='abs') excess=Math.max(0,Math.abs(val)-th);
    else if(dir==='inc') excess=Math.max(0,val-th);
    else excess=val<0?Math.max(0,-val-th):0;
    var scale=th>0?th:1;
    return Math.min(1,excess/(3*scale));
  }

  function scoreRows(rows,weights,thLow,thMod){
    var by={};
    rows.forEach(function(r){
      if(!by[r.compound]) by[r.compound]={veh:[],tr:[]};
      if(r.vehicle) by[r.compound].veh.push(r); else by[r.compound].tr.push(r);
    });
    var results=[];
    Object.keys(by).sort().forEach(function(name){
      var g=by[name];
      if(!g.veh.length||!g.tr.length) return;
      var mean=function(arr,k){return arr.reduce(function(s,x){return s+x[k]},0)/arr.length};
      var v={fpd:mean(g.veh,'fpd_ms'),rate:mean(g.veh,'beat_rate_bpm'),amp:mean(g.veh,'amplitude_uv'),stv:mean(g.veh,'stv'),tri:mean(g.veh,'triangulation_proxy')};
      var endpoints={
        fpd_change_pct: Math.max.apply(null,g.tr.map(function(r){return Math.abs(100*(r.fpd_ms-v.fpd)/v.fpd)})),
        beat_rate_change_pct: Math.max.apply(null,g.tr.map(function(r){return Math.abs(100*(r.beat_rate_bpm-v.rate)/v.rate)})),
        amplitude_change_pct: Math.min.apply(null,g.tr.map(function(r){return 100*(r.amplitude_uv-v.amp)/v.amp})),
        stv_increase: Math.max.apply(null,g.tr.map(function(r){return (r.stv-v.stv)/Math.max(v.stv,1e-6)})),
        triangulation_proxy: Math.max.apply(null,g.tr.map(function(r){return (r.triangulation_proxy-v.tri)/Math.max(v.tri,1e-6)}))
      };
      var defs=[
        ['fpd_change_pct',weights.fpd,10,'abs'],
        ['beat_rate_change_pct',weights.rate,15,'abs'],
        ['amplitude_change_pct',weights.amp,20,'dec'],
        ['stv_increase',weights.stv,0.15,'inc'],
        ['triangulation_proxy',weights.tri,0.2,'inc']
      ];
      var sum=0,tw=0,contribs=[];
      defs.forEach(function(d){
        var c=norm(endpoints[d[0]],d[3],d[2]);
        sum+=d[1]*c; tw+=d[1];
        contribs.push({name:d[0],raw:endpoints[d[0]],c:c,w:d[1]});
      });
      var score=Math.min(1,Math.max(0,tw?sum/tw:0));
      var cls=score<thLow?'Low':(score<thMod?'Moderate':'High');
      var maxConc=Math.max.apply(null,g.tr.map(function(r){return r.concentration_uM}));
      results.push({compound:name,score:score,cls:cls,maxConc:maxConc,nWells:g.tr.length,contribs:contribs,endpoints:endpoints});
    });
    results.sort(function(a,b){return b.score-a.score});
    return results;
  }

  var lastResults=null, lastQc=[], uploaded=null;

  function render(results,qcLog){
    lastResults=results; lastQc=qcLog||[];
    var qc=document.getElementById('qc');
    var ul=document.getElementById('qcList');
    if(lastQc.length){qc.style.display='block';ul.innerHTML=lastQc.map(function(l){return '<li>'+l+'</li>'}).join('')}
    else qc.style.display='none';
    var html='<div class="card"><h3>Summary — MCM / biodefense prioritization</h3><table><thead><tr><th>Candidate</th><th>CardioScore</th><th>Risk</th><th>Max uM</th><th>Wells</th></tr></thead><tbody>';
    results.forEach(function(r){
      html+='<tr><td><strong>'+escapeHtml(r.compound)+'</strong></td><td>'+r.score.toFixed(3)+'</td><td><span class="badge badge-'+escapeHtml(r.cls)+'">'+escapeHtml(r.cls)+'</span></td><td>'+r.maxConc.toFixed(2)+'</td><td>'+r.nWells+'</td></tr>';
    });
    html+='</tbody></table></div>';
    results.forEach(function(r){
      html+='<div class="card"><h3>'+escapeHtml(r.compound)+' <span class="badge badge-'+escapeHtml(r.cls)+'">'+escapeHtml(r.cls)+'</span></h3>';
      html+='<p class="meta" style="margin:0 0 8px">CardioScore '+r.score.toFixed(3)+'</p>';
      r.contribs.forEach(function(c){
        var w=Math.max(2,Math.round(c.c*140));
        html+='<div class="bar-wrap"><div style="width:170px;font-size:12px">'+escapeHtml(c.name)+'</div><div class="bar" style="width:'+w+'px"></div><span class="meta">'+c.c.toFixed(3)+' (w='+c.w+')</span></div>';
      });
      html+='</div>';
    });
    document.getElementById('out').innerHTML=html;
  }

  function weights(){
    return {
      fpd:+document.getElementById('w_fpd').value||0.3,
      rate:+document.getElementById('w_rate').value||0.15,
      amp:+document.getElementById('w_amp').value||0.15,
      stv:+document.getElementById('w_stv').value||0.25,
      tri:+document.getElementById('w_tri').value||0.15
    };
  }

  document.getElementById('csvFile').addEventListener('change',function(e){
    var f=e.target.files&&e.target.files[0];
    if(!f){uploaded=null;return}
    var reader=new FileReader();
    reader.onload=function(){try{uploaded=parseCsv(String(reader.result)); alert('Loaded '+uploaded.length+' validated rows');}catch(err){uploaded=null;alert(err.message)}};
    reader.readAsText(f);
  });

  document.getElementById('run').onclick=function(){
    var rows=uploaded&&uploaded.length?uploaded:generate(
      +document.getElementById('nComp').value||4,
      +document.getElementById('nConc').value||6,
      +document.getElementById('seed').value||42,
      document.getElementById('preset').value,
      document.getElementById('includeToxic').checked
    );
    var qc=applyQc(rows);
    render(scoreRows(qc.rows,weights(),+document.getElementById('th_low').value||0.3,+document.getElementById('th_mod').value||0.6),qc.log);
  };

  document.getElementById('clear').onclick=function(){
    document.getElementById('out').innerHTML='';
    document.getElementById('qc').style.display='none';
    lastResults=null;
  };

  function download(name,text,type){
    var a=document.createElement('a');
    a.href=URL.createObjectURL(new Blob([text],{type:type}));
    a.download=name; a.click();
    setTimeout(function(){URL.revokeObjectURL(a.href)},1000);
  }
  document.getElementById('exportJson').onclick=function(){
    if(!lastResults){alert('Run scoring first');return}
    download('cardioscore_results.json',JSON.stringify({qc:lastQc,scores:lastResults},null,2),'application/json');
  };
  document.getElementById('exportCsv').onclick=function(){
    if(!lastResults){alert('Run scoring first');return}
    var lines=['compound,cardioscore,risk_class,max_concentration_uM,n_wells'];
    lastResults.forEach(function(r){lines.push([JSON.stringify(r.compound),r.score.toFixed(4),r.cls,r.maxConc,r.nWells].join(','))});
    download('cardioscore_summary.csv',lines.join('\n'),'text/csv');
  };

  document.getElementById('run').click();
})();
