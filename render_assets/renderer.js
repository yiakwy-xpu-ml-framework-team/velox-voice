"use strict";

// These display adaptations leave the saved ABC unchanged.
function expandRests(abc) {
  let header = true, voice = null, meter = [4, 4], unit = [1, 8];
  const meters = new Map();
  return abc.split(/\r?\n/).map(line => {
    const field = line.match(/^([A-Za-z]):\s*(.*)$/);
    if (field) {
      const [, key, value] = field;
      if (key === "V") {
        voice = value.split(/\s/)[0];
        if (!meters.has(voice)) meters.set(voice, meter);
      }
      if (key === "M") {
        const match = value.match(/^(\d+)\/(\d+)$/);
        if (match) {
          const next = match.slice(1).map(Number);
          if (header || !voice) meter = next;
          else meters.set(voice, next);
        }
      }
      if (key === "L") unit = value.split("/").map(Number);
      if (key === "K") header = false;
      return line;
    }
    if (header || line.startsWith("%")) return line;
    return line.replace(/"[^"\n]*"|\[[^\]\n]*\]|Z(\d*)\|/g, (token, count) => {
      if (count === undefined) return token;
      const [n, d] = meters.get(voice) || meter;
      return (`z${n * unit[1] / (d * unit[0])}|`).repeat(Number(count || 1));
    });
  }).join("\n");
}

function fixSystemKeys(tune) {
  for (const line of tune.lines) for (const staff of line.staff || []) {
    for (const voice of staff.voices) {
      for (let i = 0; i < voice.length && voice[i].el_type !== "note";) {
        const event = voice[i];
        if (event.el_type === "key" || event.el_type === "keySignature") {
          staff.key = {...event, accidentals: event.accidentals.map(a => ({...a}))};
          delete staff.key.impliedNaturals;
          voice.splice(i, 1);
        } else i++;
      }
    }
  }
}

window.renderScore = async ({abc, tracks}) => {
  document.body.innerHTML = '<div id="source"></div><main id="pages"></main>';
  const fontStyle = document.createElement("style");
  fontStyle.textContent = '#source text, #source span, #pages text {font-family:SheetSageSans !important}';
  document.head.append(fontStyle);
  const source = document.getElementById("source");
  source.style.cssText = "position:absolute;left:-2000px;width:714px";
  const expanded = expandRests(abc);
  const tunes = ABCJS.renderAbc(source, expanded, {
    staffwidth: 674, paddingtop: 12, paddingbottom: 12, paddingleft: 20, paddingright: 20,
    oneSvgPerLine: true, print: true, add_classes: true,
    wrap: {minSpacing: 1.8, maxSpacing: 2.7, preferredMeasuresPerLine: 4},
    afterParsing: fixSystemKeys,
  });
  if (!tunes.length || !tunes.some(t => t.lines.some(l => l.staff?.length)))
    throw new Error("ABC contains no music to render.");
  const warnings = tunes.flatMap(t => t.warnings || []);
  if (warnings.length) throw new Error("Invalid ABC: " + warnings.join("; "));
  await document.fonts.ready;
  // ── per-voice measure geometry + note tagging for score-following ──
  // Voice of a staff line = the last "V:" header before its first event's
  // startChar (char offsets refer to the expanded ABC text). Measure indices
  // are counted per voice via barline events, matching playback.json order.
  const voiceAt = (() => {
    const marks = [];
    expanded.split(/\r?\n/).reduce((pos, line) => {
      const m = line.match(/^V:\s*([^\s%]+)/);
      if (m) marks.push([pos, m[1]]);
      return pos + line.length + 1;
    }, 0);
    return startChar => {
      let name = null;
      for (const [pos, v] of marks) { if (pos <= startChar) name = v; else break; }
      return name || "V1";
    };
  })();
  // Tag every SVG element of a note with data-voice/data-m/data-ni, so the
  // client can highlight the notes that are sounding. data-m is the measure
  // index (matches playback.json measures order), data-ni the ordinal of the
  // note among the non-tied notes of that measure in the same voice.
  // A note tied across a barline (`g2-|g2`) is one sounding note written as
  // two ABC elements: the continuation carries the tied note's (m, ni), so
  // both halves light up together.
  const tagNote = (ev, m, ni, vname) => {
    const set = ev.abselem && ev.abselem.elemset;
    if (!set) return;
    for (const el of set) {
      el.setAttribute("data-m", m);
      el.setAttribute("data-ni", ni);
      el.setAttribute("data-voice", vname);
    }
  };
  const voiceState = {};
  const musicLines = tunes[0].lines.filter(l => l.staff && l.staff.length).map(line => {
    let geometry = null;
    for (const staff of line.staff) {
      const voice = (staff.voices || [])[0] || [];
      if (!voice.length) continue;
      const vname = voiceAt(voice[0].startChar);
      const S = voiceState[vname] ||
        (voiceState[vname] = {m: 0, ni: 0, pendingTie: false, lastTied: null});
      const mStart = S.m;
      for (const ev of voice) {
        if (ev.el_type === "bar") {
          S.m++; S.ni = 0;
        }
        if (ev.el_type !== "note" || ev.rest) continue;
        const text = expanded.slice(ev.startChar ?? 0, ev.endChar ?? 0);
        const tieStart = /-\s*$/.test(text);
        if (S.pendingTie) {
          // continuation of the tied note started earlier: same (m, ni)
          S.pendingTie = tieStart;
          if (S.lastTied) tagNote(ev, S.lastTied.m, S.lastTied.ni, vname);
          continue;
        }
        S.pendingTie = tieStart;
        tagNote(ev, S.m, S.ni, vname);
        if (tieStart) S.lastTied = {m: S.m, ni: S.ni};
        S.ni++;
      }
      if (!geometry) geometry = {voice: vname, m0: mStart, m1: S.m - 1};
    }
    return geometry;
  }).filter(Boolean);
  const ns = "http://www.w3.org/2000/svg";
  const pages = [];
  let page, content, used = 0, systems = 0;
  function addPage() {
    page = document.createElementNS(ns, "svg");
    for (const [k, v] of Object.entries({xmlns:ns, width:794, height:1123, viewBox:"0 0 794 1123"}))
      page.setAttribute(k,v);
    page.setAttribute("class", "score-page");
    const fontStyle = document.createElementNS(ns, "style");
    fontStyle.textContent = '@font-face{font-family:SheetSageSans;src:url(data:font/ttf;base64,' + window.renderFontData + ')}text{font-family:SheetSageSans !important}';
    page.append(fontStyle);
    const license = document.createElementNS(ns, "metadata");
    license.textContent = window.renderFontLicense;
    page.append(license);
    const background = document.createElementNS(ns, "rect");
    for (const [k,v] of Object.entries({width:794,height:1123,fill:"white"})) background.setAttribute(k,v);
    page.append(background);
    content = document.createElementNS(ns, "g");
    content.setAttribute("transform", "translate(40 40)");
    page.append(content);
    document.getElementById("pages").append(page);
    pages.push({el: page, systems: []});
    used = 0;
  }
  let lineIndex = 0;
  for (const svg of source.querySelectorAll("svg")) {
    const box = svg.getBoundingClientRect();
    if (!(box.width > 0 && box.height > 0)) continue;
    const scale = Math.min(1, 714 / box.width);
    const height = box.height * scale;
    if (height > 1043) throw new Error("A staff system is taller than one page.");
    if (!page || used + height > 1043) addPage();
    const group = document.createElementNS(ns, "g");
    group.setAttribute("transform", `translate(0 ${used}) scale(${scale})`);
    const clone = svg.cloneNode(true);
    clone.setAttribute("width",box.width);
    clone.setAttribute("height",box.height);
    clone.style.overflow = "visible";
    group.append(clone);
    content.append(group);
    if (lineIndex < musicLines.length)
      pages[pages.length - 1].systems.push(
        Object.assign({}, musicLines[lineIndex], {y: used, h: height}));
    lineIndex++;
    used += height;
    systems++;
  }
  source.remove();
  if (!pages.length) throw new Error("No score pages were produced.");
  return {pages:pages.map(p => new XMLSerializer().serializeToString(p.el)), systems, warnings,
          page_info: pages.map(p => ({
            measures: p.systems.length
              ? [Math.min(...p.systems.map(s => s.m0)), Math.max(...p.systems.map(s => s.m1))]
              : null,
            systems: p.systems.map(({el, ...rest}) => rest)}))};
};

window.renderAudio = async ({tracks, duration}) => {
  const context = new AudioContext({sampleRate:44100});
  await context.resume();
  const sequence = new ABCJS.synth.SynthSequence();
  for (const track of tracks) {
    const id = sequence.addTrack();
    sequence.setInstrument(id, 0);
    for (const note of track.notes) sequence.tracks[id].push({
      cmd:"note", instrument:0, pitch:note.pitch, volume:note.velocity,
      start:note.start, duration:note.end-note.start, gap:0,
    });
  }
  sequence.totalDuration = duration;
  const synth = new ABCJS.synth.CreateSynth();
  const loaded = await synth.init({audioContext:context, sequence, millisecondsPerMeasure:1000,
    options:{soundFontUrl:"https://render.invalid/soundfonts/", fadeLength:35,
      programOffsets:{acoustic_grand_piano:0}, soundFontVolumeMultiplier:1}});
  if (loaded.error?.length) throw new Error("Could not load piano samples: " + loaded.error.join(", "));
  await synth.prime();
  const buffer = synth.getAudioBuffer();
  if (!buffer) throw new Error("The renderer returned no audio.");
  // Simultaneous voices retain their balance; a peak guard avoids PCM clipping.
  let peak = 0;
  for (let c=0; c<buffer.numberOfChannels; c++) {
    const channel = buffer.getChannelData(c);
    for (let i=0; i<channel.length; i++) peak = Math.max(peak, Math.abs(channel[i]));
  }
  const gain = Math.min(0.7, peak > 0 ? 0.98 / peak : 0.7);
  window.renderedAudio = {buffer, gain};
  await context.close();
  return {frames:buffer.length, sample_rate:buffer.sampleRate, channels:buffer.numberOfChannels, gain};
};

window.audioBlock = ({start, count}) => {
  const {buffer, gain} = window.renderedAudio;
  const frames = Math.min(count, buffer.length-start);
  const bytes = new Uint8Array(frames*buffer.numberOfChannels*2);
  const view = new DataView(bytes.buffer);
  for (let c=0; c<buffer.numberOfChannels; c++) {
    const channel = buffer.getChannelData(c);
    for (let i=0; i<frames; i++) {
      const sample = Math.max(-1,Math.min(1,channel[start+i]*gain));
      view.setInt16((i*buffer.numberOfChannels+c)*2, Math.round(sample*32767), true);
    }
  }
  let binary="";
  for (let offset=0; offset<bytes.length; offset+=8192)
    binary += String.fromCharCode(...bytes.subarray(offset,offset+8192));
  return btoa(binary);
};
