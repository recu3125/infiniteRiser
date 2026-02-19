(() => {
  const $ = (id) => document.getElementById(id);
  const outEl = $("out");
  const progEl = $("prog");
  const pctEl = $("pct");
  const stageEl = $("stage");
  const elapsedEl = $("elapsed");
  const etaEl = $("eta");
  const speedStatEl = $("speedStat");

  const SETTINGS = {
    octSpan: 4,
    cycleSec: 13,
    N: 4096,
    hopDiv: 4,
    shepSigma: 1.0,
    shepCenter: 0.5,
    fmin: 50,
    fmax: 20000,
    sigmaPitch: 0.85,
    isoPhon: 60,
    isoStrength: 1,
    maxPreBoostDb: 0,
    maxPostBoostDb: 18,
    outRmsDb: -16,
    peakLimit: 0.98
  };

  let aborted = false;

  function bindRange(id, vid, fmt=(v)=>v) {
    const el = $(id), vEl = $(vid);
    const sync = () => vEl.textContent = fmt(+el.value);
    el.addEventListener("input", sync);
    sync();
    return el;
  }

  const speedSlider = $("speed");
  const speedValue = $("speedV");
  const speedMin = parseFloat(speedSlider.min);
  const speedMax = parseFloat(speedSlider.max);
  const speedStep = parseFloat(speedSlider.step);

  function getSpeedSeconds() {
    const raw = parseFloat(speedSlider.value);
    const actual = speedMin + speedMax - raw;
    return Math.round(actual / speedStep) * speedStep;
  }

  function syncSpeedLabel() {
    speedValue.textContent = `${getSpeedSeconds().toFixed(1)}s`;
  }

  speedSlider.addEventListener("input", syncSpeedLabel);
  syncSpeedLabel();

  function log(msg){
    stageEl.textContent = msg;
  }
  function setStage(s) { stageEl.textContent = s; }
  function clamp(v,a,b){ return Math.max(a, Math.min(b, v)); }
  function fmtTime(sec) {
    if (!isFinite(sec) || sec < 0) return "--:--";
    const s = Math.round(sec);
    const mm = String(Math.floor(s/60)).padStart(2,"0");
    const ss = String(s%60).padStart(2,"0");
    return `${mm}:${ss}`;
  }
  function peakOf(x){
    let p=0;
    for (let i=0;i<x.length;i++){ const a=Math.abs(x[i]); if(a>p) p=a; }
    return p;
  }
  function dbToLin(db){ return Math.pow(10, db/20); }

  // ---------- RMS (middle 80% by |sample|) ----------
  function rmsOfMiddle80(x){
    const n = x.length;
    if (n <= 1) return 0;

    const a = new Float32Array(n);
    for (let i=0;i<n;i++) a[i] = Math.abs(x[i]);
    a.sort();

    const loIdx = Math.floor(0.10 * (n-1));
    const hiIdx = Math.floor(0.90 * (n-1));
    const lo = a[loIdx];
    const hi = a[hiIdx];

    let s2 = 0;
    let cnt = 0;
    for (let i=0;i<n;i++){
      const v = x[i];
      const av = Math.abs(v);
      if (av < lo || av > hi) continue;
      s2 += v*v;
      cnt++;
    }
    if (cnt <= 0) return 0;
    return Math.sqrt(s2 / cnt);
  }

  // ---------- Progress + ETA ----------
  const ETA = {
    startedAt: 0, lastT: 0, lastP: 0, emaSpeed: 0, lastUi: 0,
    reset(){
      const now = performance.now();
      this.startedAt = now; this.lastT = now; this.lastP = 0; this.emaSpeed = 0; this.lastUi = 0;
      elapsedEl.textContent = "00:00"; etaEl.textContent="--:--"; speedStatEl.textContent="-";
    },
    update(p){
      const now = performance.now();
      const elapsed = (now - this.startedAt)/1000;
      elapsedEl.textContent = fmtTime(elapsed);

      const dt = (now - this.lastT)/1000;
      const dp = p - this.lastP;
      if (dt > 0.2 && dp >= 0) {
        const inst = dp/dt;
        const alpha = 0.15;
        this.emaSpeed = this.emaSpeed === 0 ? inst : (alpha*inst + (1-alpha)*this.emaSpeed);
        this.lastT = now; this.lastP = p;
      }
      if (this.emaSpeed > 1e-6 && p > 0.02) {
        const remain = (1-p)/this.emaSpeed;
        etaEl.textContent = fmtTime(remain);
        speedStatEl.textContent = `${(this.emaSpeed*100).toFixed(2)}%/s`;
      } else {
        etaEl.textContent = "--:--"; speedStatEl.textContent = "-";
      }
    }
  };
  function setProgress(v, force=false) {
    const p = clamp(v,0,1);
    const now = performance.now();
    if (!force && now - ETA.lastUi < 120) return;
    ETA.lastUi = now;
    progEl.value = p;
    pctEl.textContent = Math.round(p*100) + "%";
    ETA.update(p);
  }

  // ---------- Window ----------
  function hann(N) {
    const w = new Float32Array(N);
    const twoPi = 2*Math.PI;
    for (let i=0;i<N;i++) w[i]=0.5*(1-Math.cos(twoPi*i/(N-1)));
    return w;
  }

  // ---------- FFT ----------
  function bitReverse(x, bits){
    let y=0;
    for(let i=0;i<bits;i++){ y=(y<<1)|(x&1); x>>>=1; }
    return y;
  }
  function fft(re, im, inverse=false){
    const n=re.length;
    const bits=(Math.log2(n)|0);
    for(let i=0;i<n;i++){
      const j=bitReverse(i,bits);
      if(j>i){
        let t=re[i]; re[i]=re[j]; re[j]=t;
        t=im[i]; im[i]=im[j]; im[j]=t;
      }
    }
    for(let len=2; len<=n; len<<=1){
      const ang=(inverse?2:-2)*Math.PI/len;
      const wlenRe=Math.cos(ang), wlenIm=Math.sin(ang);
      for(let i=0;i<n;i+=len){
        let wRe=1, wIm=0;
        const half=len>>1;
        for(let j=0;j<half;j++){
          const i0=i+j, i1=i0+half;
          const uRe=re[i0], uIm=im[i0];
          const vRe=re[i1]*wRe - im[i1]*wIm;
          const vIm=re[i1]*wIm + im[i1]*wRe;
          re[i0]=uRe+vRe; im[i0]=uIm+vIm;
          re[i1]=uRe-vRe; im[i1]=uIm-vIm;
          const nwRe=wRe*wlenRe - wIm*wlenIm;
          const nwIm=wRe*wlenIm + wIm*wlenRe;
          wRe=nwRe; wIm=nwIm;
        }
      }
    }
    if(inverse){
      const inv=1/n;
      for(let i=0;i<n;i++){ re[i]*=inv; im[i]*=inv; }
    }
  }

  // ---------- Decode ----------
  async function decodeToMonoFloat32(file){
    const buf = await file.arrayBuffer();
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const ab = await ac.decodeAudioData(buf);
    const sr = ab.sampleRate;
    const n = ab.length;
    const ch = ab.numberOfChannels;
    const mono = new Float32Array(n);
    for(let c=0;c<ch;c++){
      const d=ab.getChannelData(c);
      for(let i=0;i<n;i++) mono[i]+=d[i]/ch;
    }
    ac.close();
    return { mono, sr, duration: ab.duration };
  }

  // ---------- Shepard ----------
  function buildShepardWeights(N, sr, fmin, fmax, center, sigma){
    const half = N/2;
    const w = new Float32Array(half+1);
    const logMin = Math.log2(fmin);
    const logMax = Math.log2(fmax);
    const logSpan = logMax - logMin;

    for(let k=0;k<=half;k++){
      const f = (k*sr)/N;
      if (f < fmin || f > fmax || f===0) { w[k]=0; continue; }
      const u = (Math.log2(f) - logMin) / logSpan;
      const z = (u - center) / sigma;
      w[k] = Math.exp(-0.5*z*z);
    }
    return w;
  }

  // ---------- Pitch weight ----------
  function pitchWeightNearOriginal(a, sigmaOct){
    const dist = Math.log2(a);
    const z = dist / sigmaOct;
    return Math.exp(-0.5 * z * z);
  }

  // ---------- ISO 226:2003 ----------
  const ISO226_F = [20,25,31.5,40,50,63,80,100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12500];
  const ISO226_AF = [0.532,0.506,0.480,0.455,0.432,0.409,0.387,0.367,0.349,0.330,0.315,0.301,0.288,0.276,0.267,0.259,0.253,0.250,0.246,0.244,0.243,0.243,0.243,0.242,0.242,0.245,0.254,0.271,0.301];
  const ISO226_LU = [-31.6,-27.2,-23.0,-19.1,-15.9,-13.0,-10.3,-8.1,-6.2,-4.5,-3.1,-2.0,-1.1,-0.4,0.0,0.3,0.5,0.0,-2.7,-4.1,-1.0,1.7,2.5,1.2,-2.1,-7.1,-11.2,-10.7,-3.1];
  const ISO226_TF = [78.5,68.7,59.5,51.1,44.0,37.5,31.5,26.5,22.1,17.9,14.4,11.4,8.6,6.2,4.4,3.0,2.2,2.4,3.5,1.7,-1.3,-4.2,-6.0,-5.4,-1.5,6.0,12.6,13.9,12.3];

  function iso226SPLTable(phon){
    const LN = phon;
    const spl = new Float32Array(ISO226_F.length);
    const t1 = 4.47e-3 * (Math.pow(10, 0.025*LN) - 1.15);
    for (let i=0;i<ISO226_F.length;i++){
      const af = ISO226_AF[i];
      const Lu = ISO226_LU[i];
      const Tf = ISO226_TF[i];
      const t2 = Math.pow(0.4 * Math.pow(10, ((Tf + Lu)/10) - 9), af);
      const Af = t1 + t2;
      const Lp = (10/af) * Math.log10(Af) - Lu + 94;
      spl[i] = Lp;
    }
    return spl;
  }

  function interpLogFreq(xFreqs, yVals, f){
    const n = xFreqs.length;
    if (f <= xFreqs[0]) return yVals[0];
    if (f >= xFreqs[n-1]) return yVals[n-1];
    const lf = Math.log(f);
    let lo=0, hi=n-1;
    while (hi - lo > 1){
      const mid = (lo+hi)>>1;
      if (xFreqs[mid] <= f) lo = mid; else hi = mid;
    }
    const f0 = xFreqs[lo], f1 = xFreqs[hi];
    const t = (lf - Math.log(f0)) / (Math.log(f1) - Math.log(f0));
    return yVals[lo] + (yVals[hi] - yVals[lo]) * t;
  }

  function buildISOWeightsPerBin(N, sr, phon, strength01, maxPreBoostDb, maxPostBoostDb){
    const splTable = iso226SPLTable(phon);
    const Dtable = new Float32Array(splTable.length);
    for (let i=0;i<splTable.length;i++) Dtable[i] = splTable[i] - phon;

    const half = N/2;
    const pre = new Float32Array(half+1);
    const post = new Float32Array(half+1);

    const preBoostLinMax = dbToLin(maxPreBoostDb);
    const postBoostLinMax = dbToLin(maxPostBoostDb);

    for (let k=0;k<=half;k++){
      const f = (k*sr)/N;
      if (f <= 0) { pre[k]=1; post[k]=1; continue; }

      const D = interpLogFreq(ISO226_F, Dtable, f);
      const Dscaled = D * strength01;

      let gPre = dbToLin(-Dscaled);
      if (gPre > preBoostLinMax) gPre = preBoostLinMax;

      let gPost = 1 / Math.max(1e-12, gPre);
      if (gPost > postBoostLinMax) gPost = postBoostLinMax;

      pre[k] = gPre;
      post[k] = gPost;
    }
    return { pre, post };
  }

  // ---------- Output: RMS(mid80) normalize then hard limiter ----------
  function normalizeToRmsMiddle80(y, targetDb){
    const targetRms = dbToLin(targetDb);
    const eps = 1e-12;
    const rms = Math.max(eps, rmsOfMiddle80(y));
    const g = targetRms / rms;
    for (let i=0;i<y.length;i++) y[i] *= g;
  }

  function hardLimit(y, peakLimit){
    const lim = Math.max(0.0001, peakLimit);
    for (let i=0;i<y.length;i++){
      if (y[i] > lim) y[i] = lim;
      else if (y[i] < -lim) y[i] = -lim;
    }
  }

  // ---------- STFT main ----------
  async function processInfiniteRisingSTFT(xIn, sr, params){
    const {
      octSpan, cycleSec, N, hopDiv,
      fmin, fmax,
      shepCenter, shepSigma,
      isoPhon, isoStrength,
      maxPreBoostDb, maxPostBoostDb,
      sigmaPitch
    } = params;

    const x = (xIn.length < N) ? (() => {
      const z = new Float32Array(N);
      z.set(xIn);
      return z;
    })() : xIn;

    const Ha = Math.max(1, (N / hopDiv) | 0);
    const Hs = Ha;
    const win = hann(N);
    const half = N/2;

    const shep = buildShepardWeights(N, sr, fmin, fmax, shepCenter, shepSigma);
    const iso = buildISOWeightsPerBin(N, sr, isoPhon, isoStrength, maxPreBoostDb, maxPostBoostDb);

    const nFrames = Math.max(1, Math.floor((x.length - N) / Ha) + 1);
    const outLen = (nFrames - 1) * Hs + N;
    const y = new Float32Array(outLen);
    const norm = new Float32Array(outLen);

    const reA = new Float32Array(N);
    const imA = new Float32Array(N);
    const prevPhaseA = new Float32Array(half+1);
    const magA = new Float32Array(half+1);
    const omegaA = new Float32Array(half+1);

    const reO = new Float32Array(N);
    const imO = new Float32Array(N);

    const L = 2*octSpan + 1;
    const phaseL = Array.from({length:L}, () => new Float32Array(half+1));

    const twoPi = 2*Math.PI;

    const kMin = Math.max(1, Math.floor(fmin * N / sr));
    const kMax = Math.min(half, Math.ceil(fmax * N / sr));

    const magScale = 2 / N;

    setStage("Analyze & Synthesize");
    const MAIN_W = 0.92;

    for (let m=0; m<nFrames; m++){
      if (aborted) throw new Error("aborted");
      const inPos = m*Ha;

      for(let i=0;i<N;i++){
        reA[i] = (x[inPos+i] || 0) * win[i];
        imA[i] = 0;
      }
      fft(reA, imA, false);

      for(let k=0;k<=half;k++){
        const r = reA[k], ii = imA[k];
        const mag = Math.hypot(r, ii);
        const ph = Math.atan2(ii, r);
        const delta = ph - prevPhaseA[k];
        prevPhaseA[k] = ph;

        const expected = twoPi * k * Ha / N;
        let d = delta - expected;
        d -= twoPi * Math.round(d / twoPi);
        const phaseAdv = expected + d;

        magA[k] = mag * magScale;
        omegaA[k] = phaseAdv / Ha;
      }

      reO.fill(0); imO.fill(0);

      const tSec = inPos / sr;
      const tIn = ((tSec % cycleSec) + cycleSec) % cycleSec;
      const log2r = tIn / cycleSec;
      const rNow = Math.pow(2, log2r);

      for (let li=0; li<L; li++){
        const okt = li - octSpan;
        const a = rNow * Math.pow(2, okt);
        const phAcc = phaseL[li];

        const wPitch = pitchWeightNearOriginal(a, sigmaPitch);
        if (wPitch < 1e-5) continue;

        for (let k=kMin; k<=kMax; k++){
          const wShep = shep[k];
          if (wShep === 0) continue;

          const s = k / a;
          if (s < 1 || s >= half) continue;

          const si = s | 0;
          const frac = s - si;

          const mag = magA[si] + (magA[si+1] - magA[si]) * frac;
          let omega = omegaA[si] + (omegaA[si+1] - omegaA[si]) * frac;

          const wIsoPre = iso.pre[si] + (iso.pre[si+1] - iso.pre[si]) * frac;

          omega *= a;
          const ph = (phAcc[k] += omega * Hs);

          const g = wIsoPre * wShep * wPitch;
          if (g === 0) continue;

          const amp = mag * g;
          reO[k] += amp * Math.cos(ph);
          imO[k] += amp * Math.sin(ph);
        }
      }

      for (let k=kMin; k<=kMax; k++){
        const gPost = iso.post[k];
        reO[k] *= gPost;
        imO[k] *= gPost;
      }

      for (let k=1; k<half; k++){
        reO[N-k] = reO[k];
        imO[N-k] = -imO[k];
      }
      imO[0]=0;
      if ((N & 1) === 0) imO[half]=0;

      fft(reO, imO, true);

      const outPos = m*Hs;
      for(let i=0;i<N;i++){
        const wv = win[i];
        const v = reO[i] * wv;
        y[outPos+i] += v;
        norm[outPos+i] += wv*wv;
      }

      if ((m & 7) === 0) {
        setProgress(MAIN_W * ((m+1)/nFrames));
        await new Promise(r=>setTimeout(r,0));
      }
    }

    setStage("Normalize");
    for (let i=0;i<y.length;i++){
      const d = norm[i];
      if (d > 1e-8) y[i] /= d;
    }

    setProgress(0.96, true);

    return y.subarray(0, xIn.length);
  }

  // ---------- WAV encode ----------
  function encodeWavMono(samples, sampleRate) {
    const n = samples.length;
    const bytesPerSample = 2;
    const dataSize = n * bytesPerSample;
    const buf = new ArrayBuffer(44 + dataSize);
    const dv = new DataView(buf);

    function writeStr(off, s){ for(let i=0;i<s.length;i++) dv.setUint8(off+i, s.charCodeAt(i)); }

    writeStr(0, "RIFF");
    dv.setUint32(4, 36 + dataSize, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    dv.setUint32(16, 16, true);
    dv.setUint16(20, 1, true);
    dv.setUint16(22, 1, true);
    dv.setUint32(24, sampleRate, true);
    dv.setUint32(28, sampleRate * bytesPerSample, true);
    dv.setUint16(32, bytesPerSample, true);
    dv.setUint16(34, 16, true);
    writeStr(36, "data");
    dv.setUint32(40, dataSize, true);

    let off = 44;
    for(let i=0;i<n;i++){
      let s = clamp(samples[i], -1, 1);
      dv.setInt16(off, s < 0 ? (s*0x8000) : (s*0x7fff), true);
      off += 2;
    }
    return new Blob([buf], { type: "audio/wav" });
  }

  function setUiBusy(b){
    $("run").disabled=b;
    $("cancel").disabled=!b;
    $("file").disabled=b;
    $("speed").disabled=b;
  }

  $("cancel").addEventListener("click", () => { aborted = true; setStage("Canceling..."); });

  $("run").addEventListener("click", async () => {
    outEl.innerHTML = "";
    aborted = false;

    ETA.reset();
    setProgress(0, true);
    setStage("-");

    const f = $("file").files?.[0];
    if (!f) { setStage("Please select a file"); return; }

    setUiBusy(true);
    try {
      setStage("Decoding");
      const { mono, sr } = await decodeToMonoFloat32(f);

      const {
        octSpan, N, hopDiv,
        fmin, fmax,
        shepCenter, shepSigma,
        isoPhon, isoStrength,
        maxPreBoostDb, maxPostBoostDb,
        sigmaPitch,
        outRmsDb,
        peakLimit
      } = SETTINGS;
      const cycleSec = getSpeedSeconds();

      if (!(fmin > 0 && fmax > fmin && fmax <= sr/2)) {
        throw new Error(`Frequency range error: fmin=${fmin}, fmax=${fmax}, Nyquist=${(sr/2).toFixed(1)}`);
      }
      if ((N & (N-1)) !== 0) throw new Error("FFT size must be a power of 2.");
      if (sigmaPitch <= 0) throw new Error("Pitch width must be greater than 0.");
      if (cycleSec <= 0) throw new Error("Rise speed must be greater than 0.");

      setStage("Synthesizing");
      const y = await processInfiniteRisingSTFT(mono, sr, {
        octSpan, cycleSec, N, hopDiv,
        fmin, fmax,
        shepCenter, shepSigma,
        isoPhon, isoStrength,
        maxPreBoostDb, maxPostBoostDb,
        sigmaPitch
      });

      setStage("Normalizing level");
      normalizeToRmsMiddle80(y, outRmsDb);

      setStage("Limiting");
      hardLimit(y, peakLimit);

      setStage("Encoding WAV");
      setProgress(0.98, true);
      const wav = encodeWavMono(y, sr);

      const url = URL.createObjectURL(wav);
      const nameBase = (f.name || "audio").replace(/\.[^.]+$/, "");
      const a = document.createElement("a");
      a.href = url;
      a.download = `${nameBase}_infinite_rising.wav`;
      a.textContent = "Download";
      outEl.appendChild(a);

      const audio = document.createElement("audio");
      audio.controls = true;
      audio.src = url;
      outEl.appendChild(audio);

      const credit = document.createElement("div");
      credit.className = "credit";
      credit.innerHTML = "Credit recommended : <span class=\"credit-text\">Made with Infinite Riser by <span class=\"credit-name\">recu3125</span></span>";
      outEl.appendChild(credit);

      setStage("Done");
      setProgress(1, true);
    } catch(e) {
      setStage("Error");
      const msg = e?.message || String(e);
      const warn = document.createElement("div");
      warn.textContent = msg;
      warn.style.marginTop = "10px";
      warn.style.color = "#b54628";
      outEl.appendChild(warn);
      console.error(e);
    } finally {
      setUiBusy(false);
    }
  });
})();
