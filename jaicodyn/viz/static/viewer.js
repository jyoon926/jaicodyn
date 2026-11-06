"use strict";

const $ = (id) => document.getElementById(id);
const COLORS = { H: "#0a84ff", A: "#ff9f0a", D: "#48484a" };
const TOKENS = {
  pitch: "#17231d", stripe: "rgba(255,255,255,0.016)", line: "rgba(255,255,255,0.3)",
  ball: "#ffd60a", playhead: "#ff453a", lane: "#222224", header: "#1c1c1e",
  text: "rgba(235,235,245,0.62)", faint: "rgba(235,235,245,0.42)", rule: "rgba(255,255,255,0.09)",
};
const SANS = '-apple-system, BlinkMacSystemFont, "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif';
const MONO = 'ui-monospace, "SF Mono", "Cascadia Mono", Consolas, monospace';
const LAYERS = {
  lines: ["Pitch lines", true], arrows: ["Velocity", true], trail: ["Ball trail", true],
  nearest: ["Nearest player", true], ids: ["Track IDs", false], hulls: ["Team shape", false],
  centroids: ["Team centres", false],
};
const TRAIL_FRAMES = 25;
const HEADER = 118;
const RULER = 22;
const TRACK = 20;
const OVERVIEW = 30;

const state = {
  info: null, frame: 0, playing: false, rate: 1, chunks: new Map(), pending: new Set(),
  range: [0, 1], span: 1500, view: [0, 0], viewFrame: null, scrubbing: false, showVideo: false, seeking: null, followed: null,
  layers: Object.fromEntries(Object.entries(LAYERS).map(([k, v]) => [k, v[1]])),
};

/* ---------- data ---------- */

const json = async (url) => {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
};

function loadChunk(index) {
  if (index < 0 || state.chunks.has(index) || state.pending.has(index)) return;
  if (state.pending.size >= 2) return;  // a fast drag must not queue every chunk it crosses
  state.pending.add(index);
  const key = state.info.key;
  json(`/api/chunk/${key}/${index}`).then((chunk) => {
    if (state.info.key !== key) return;
    state.chunks.set(index, chunk);
    state.pending.delete(index);
    if (state.chunks.size > 40) state.chunks.delete(state.chunks.keys().next().value);
    draw();
  }).catch(() => {
    state.pending.delete(index);  // retried on the next draw
    setStatus("Could not load tracking", true);
  });
}

function dataAt(frame) {
  const size = state.info.chunk;
  const chunk = state.chunks.get(Math.floor(frame / size));
  if (!chunk) return null;
  const i = frame - chunk.start;
  return { ball: chunk.ball[i], players: chunk.players[i], homography: chunk.homography[i] };
}

function runAt(runs, frame) {
  let low = 0, high = runs.length - 1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (frame < runs[mid][0]) high = mid - 1;
    else if (frame >= runs[mid][1]) low = mid + 1;
    else return mid;
  }
  return -1;
}

const primaryRuns = () => (state.info.rows[0] ? state.info.rows[0].runs : []);
const possessionRuns = () => primaryRuns().filter((run) => run[2] === "H" || run[2] === "A");

/* ---------- clock ---------- */

const video = $("video");
const hasVideo = () => state.info && state.info.video;
const clamp = (frame) => Math.max(state.range[0], Math.min(state.range[1], Math.round(frame)));

function seek(frame) {
  frame = clamp(frame);
  state.frame = frame;
  if (hasVideo()) {
    if (video.seeking) state.seeking = frame;
    else moveVideo(frame);
  }
  draw();
}

function moveVideo(frame) {
  const time = (frame + 0.5) / state.info.fps;  // mid-frame, so rounding is safe
  if (state.scrubbing && video.fastSeek) video.fastSeek(time);
  else video.currentTime = time;
}

video.addEventListener("seeked", () => {
  if (state.seeking !== null) {
    const next = state.seeking;
    state.seeking = null;
    moveVideo(next);
  }
});

function setPlaying(playing) {
  state.playing = playing;
  $("play").classList.toggle("playing", playing);
  if (hasVideo()) {
    if (playing) video.play();
    else {
      video.pause();
      seek(Math.floor(video.currentTime * state.info.fps));  // the picture can lag the clock
    }
  } else if (playing) {
    lastTick = performance.now();
    exact = state.frame;
    requestAnimationFrame(tick);
  }
  if (!playing) saveLocation();
}

let lastTick = 0, exact = 0;
function tick(now) {
  if (!state.playing || hasVideo()) return;
  exact += ((now - lastTick) / 1000) * state.info.fps * state.rate;
  lastTick = now;
  state.frame = clamp(exact);
  if (state.frame >= state.range[1]) setPlaying(false);
  draw();
  requestAnimationFrame(tick);
}

function onVideoFrame(_, metadata) {
  const frame = Math.round(metadata.mediaTime * state.info.fps);
  if (state.playing && frame !== state.frame) {
    state.frame = frame;
    draw();
  }
  video.requestVideoFrameCallback(onVideoFrame);
}

function saveLocation() {
  if (state.info) history.replaceState(null, "", `#${state.info.key}@${state.frame}`);
}

/* ---------- geometry ---------- */

function pitchLines(length, width) {
  const m = (x, y) => [x / length, y / width];
  const rect = (x0, y0, x1, y1) => [m(x0, y0), m(x1, y0), m(x1, y1), m(x0, y1), m(x0, y0)];
  const arc = (cx, cy, r, from, to) => {
    const points = [];
    for (let i = 0; i <= 32; i++) {
      const t = from + ((to - from) * i) / 32;
      points.push(m(cx + r * Math.cos(t), cy + r * Math.sin(t)));
    }
    return points;
  };
  const midY = width / 2, box = 40.32 / 2, six = 18.32 / 2;
  const off = Math.acos(5.5 / 9.15);
  return [
    rect(0, 0, length, width),
    [m(length / 2, 0), m(length / 2, width)],
    arc(length / 2, midY, 9.15, 0, 2 * Math.PI),
    rect(0, midY - box, 16.5, midY + box),
    rect(length - 16.5, midY - box, length, midY + box),
    rect(0, midY - six, 5.5, midY + six),
    rect(length - 5.5, midY - six, length, midY + six),
    arc(11, midY, 9.15, -off, off),
    arc(length - 11, midY, 9.15, Math.PI - off, Math.PI + off),
  ];
}

function hull(points) {
  const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (sorted.length < 3) return sorted;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (list) => {
    const out = [];
    for (const p of list) {
      while (out.length > 1 && cross(out[out.length - 2], out[out.length - 1], p) <= 0) out.pop();
      out.push(p);
    }
    out.pop();
    return out;
  };
  return [...half(sorted), ...half([...sorted].reverse())];
}

function distanceM(a, bx, by) {
  const [length, width] = state.info.pitch;
  return Math.hypot((a[2] - bx) * length, (a[3] - by) * width);
}

function nearestPlayer(data) {
  if (!data || !data.ball || !data.players || !data.players.length) return null;
  let best = null, bestDistance = Infinity;
  for (const player of data.players) {
    const d = distanceM(player, data.ball[0], data.ball[1]);
    if (d < bestDistance) { best = player; bestDistance = d; }
  }
  return { player: best, distance: bestDistance };
}

/* ---------- drawing ---------- */

function fit(canvas) {
  const scale = window.devicePixelRatio || 1;
  const width = Math.round(canvas.clientWidth * scale), height = Math.round(canvas.clientHeight * scale);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  context.setTransform(scale, 0, 0, scale, 0, 0);
  context.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  return context;
}

function pitchProjector(canvas, margin) {
  const [length, width] = state.info.pitch;
  const w = canvas.clientWidth - 2 * margin, h = canvas.clientHeight - 2 * margin;
  const scale = Math.min(w / length, h / width);
  const pw = length * scale, ph = width * scale;
  const left = (canvas.clientWidth - pw) / 2, top = (canvas.clientHeight - ph) / 2;
  return (x, y) => [left + x * pw, top + y * ph];
}

function videoProjector(canvas, matrix) {
  const w = video.videoWidth, h = video.videoHeight;
  const scale = Math.min(canvas.clientWidth / w, canvas.clientHeight / h);
  const left = (canvas.clientWidth - w * scale) / 2, top = (canvas.clientHeight - h * scale) / 2;
  const toPixels = 1920 / w;  // the matrices are fitted to 1920x1080
  return (x, y) => {
    const d = matrix[6] * x + matrix[7] * y + matrix[8];
    const px = (matrix[0] * x + matrix[1] * y + matrix[2]) / d / toPixels;
    const py = (matrix[3] * x + matrix[4] * y + matrix[5]) / d / toPixels;
    return [left + px * scale, top + py * scale];
  };
}

function path(context, project, points, close) {
  context.beginPath();
  points.forEach(([x, y], i) => {
    const [px, py] = project(x, y);
    if (i) context.lineTo(px, py); else context.moveTo(px, py);
  });
  if (close) context.closePath();
}

function drawSurface(context, project) {
  path(context, project, [[0, 0], [1, 0], [1, 1], [0, 1]], true);
  context.fillStyle = TOKENS.pitch;
  context.fill();
  context.fillStyle = TOKENS.stripe;
  for (let i = 0; i < 12; i += 2) {
    path(context, project, [[i / 12, 0], [(i + 1) / 12, 0], [(i + 1) / 12, 1], [i / 12, 1]], true);
    context.fill();
  }
}

function drawTrail(context, project, frame) {
  const points = [];
  for (let f = frame - TRAIL_FRAMES; f <= frame; f++) {
    const past = dataAt(f);
    if (past && past.ball) points.push(project(past.ball[0], past.ball[1]));
  }
  context.lineCap = "round";
  for (let i = 1; i < points.length; i++) {
    context.beginPath();
    context.moveTo(...points[i - 1]);
    context.lineTo(...points[i]);
    context.strokeStyle = `rgba(255,214,10,${(0.75 * i) / points.length})`;
    context.lineWidth = 0.75 + (1.75 * i) / points.length;
    context.stroke();
  }
  context.lineCap = "butt";
}

function drawPlayerLabel(context, text, x, y) {
  context.font = `500 10.5px ${MONO}`;
  const width = context.measureText(text).width + 8;
  context.fillStyle = "rgba(0,0,0,0.62)";
  context.beginPath();
  context.roundRect(x - width / 2, y - 7.5, width, 15, 4);
  context.fill();
  context.fillStyle = "#fff";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, x, y + 0.5);
}

function drawScene(context, project, data, frame, compact) {
  const layers = state.layers, [length, width] = state.info.pitch;
  const players = (data && data.players) || [];

  if (layers.lines || compact) {
    context.strokeStyle = TOKENS.line;
    context.lineWidth = 1;
    for (const line of pitchLines(length, width)) {
      path(context, project, line);
      context.stroke();
    }
  }

  for (const side of ["H", "A"]) {
    const points = players.filter((p) => p[0] === side).map((p) => [p[2], p[3]]);
    if (!points.length) continue;
    if (layers.hulls && !compact) {
      path(context, project, hull(points), true);
      context.fillStyle = COLORS[side] + "1f";
      context.strokeStyle = COLORS[side] + "99";
      context.lineWidth = 1;
      context.fill();
      context.stroke();
    }
    if (layers.centroids && !compact) {
      const [cx, cy] = project(points.reduce((s, p) => s + p[0], 0) / points.length,
                               points.reduce((s, p) => s + p[1], 0) / points.length);
      context.strokeStyle = COLORS[side];
      context.lineWidth = 1.5;
      context.beginPath();
      context.moveTo(cx - 6, cy);
      context.lineTo(cx + 6, cy);
      context.moveTo(cx, cy - 6);
      context.lineTo(cx, cy + 6);
      context.stroke();
    }
  }

  if (layers.trail) drawTrail(context, project, frame);

  const near = data && data.ball ? nearestPlayer(data) : null;
  if (near && layers.nearest && !compact) {
    path(context, project, [[near.player[2], near.player[3]], data.ball]);
    context.strokeStyle = "rgba(255,255,255,0.7)";
    context.lineWidth = 1;
    context.setLineDash([3, 3]);
    context.stroke();
    context.setLineDash([]);
  }

  const radius = compact ? 3 : 6.5;
  const before = Math.max(1, Math.round(state.info.fps / 5));
  const earlier = dataAt(frame - before)?.players;
  for (const player of players) {
    const [x, y] = project(player[2], player[3]);
    const velocity = layers.arrows && !compact && earlier?.find((p) => p[1] === player[1]);
    if (velocity) {  // where they will be in one second
      const [ex, ey] = project(player[2] + (player[2] - velocity[2]) * 5,
                               player[3] + (player[3] - velocity[3]) * 5);
      context.beginPath();
      context.moveTo(x, y);
      context.lineTo(ex, ey);
      context.strokeStyle = COLORS[player[0]] + "b3";
      context.lineWidth = 1.5;
      context.stroke();
    }
  }

  for (const player of players) {
    const [x, y] = project(player[2], player[3]);
    const followed = player[1] === state.followed;
    context.shadowColor = "rgba(0,0,0,0.5)";
    context.shadowBlur = compact ? 0 : 4;
    context.beginPath();
    context.arc(x, y, radius, 0, 2 * Math.PI);
    context.fillStyle = COLORS[player[0]];
    context.fill();
    context.shadowBlur = 0;
    context.strokeStyle = "rgba(255,255,255,0.9)";
    context.lineWidth = compact ? 0.75 : 1;
    context.stroke();
    if (followed) {
      context.beginPath();
      context.arc(x, y, radius + (compact ? 2.5 : 4), 0, 2 * Math.PI);
      context.strokeStyle = "#fff";
      context.lineWidth = 1.5;
      context.stroke();
    }
    if (!compact && (layers.ids || followed)) {
      drawPlayerLabel(context, shortTrack(player[1]), x, y - radius - (followed ? 15 : 11));
    }
  }

  if (data && data.ball) {
    const [x, y] = project(data.ball[0], data.ball[1]);
    context.shadowColor = "rgba(255,214,10,0.7)";
    context.shadowBlur = compact ? 3 : 8;
    context.beginPath();
    context.arc(x, y, compact ? 2.5 : 4.5, 0, 2 * Math.PI);
    context.fillStyle = TOKENS.ball;
    context.fill();
    context.shadowBlur = 0;
    context.strokeStyle = "rgba(0,0,0,0.75)";
    context.lineWidth = 1;
    context.stroke();
  }
}

let projectMain = null;

function drawMain(data) {
  const canvas = $("overlay"), context = fit(canvas);
  const onVideo = state.showVideo && hasVideo();
  projectMain = null;
  if (onVideo) {
    if (!video.videoWidth) return;
    if (!data || !data.homography) return;
    projectMain = videoProjector(canvas, data.homography);
  } else {
    projectMain = pitchProjector(canvas, 36);
    drawSurface(context, projectMain);
  }
  if (data) drawScene(context, projectMain, data, state.frame, false);
}

function drawMini(data) {
  const canvas = $("mini");
  canvas.hidden = !(state.showVideo && hasVideo());
  if (canvas.hidden) return;
  const context = fit(canvas);
  const project = pitchProjector(canvas, 10);
  drawSurface(context, project);
  drawScene(context, project, data, state.frame, true);
}

/* ---------- timelines ---------- */

const titleCase = (text) => text.charAt(0).toUpperCase() + text.slice(1).toLowerCase();
const rowName = (name) => (name === "labels" ? "Possession" : name.charAt(0).toUpperCase() + name.slice(1));

function fitText(context, text, width) {
  if (context.measureText(text).width <= width) return text;
  let cut = text;
  while (cut && context.measureText(`${cut}…`).width > width) cut = cut.slice(0, -1);
  return `${cut}…`;
}

function timelineRows() {
  const info = state.info;
  const rows = info.rows.map((row) => ({
    name: rowName(row.name), runs: row.runs, color: (r) => COLORS[r[2]],
  }));
  rows.push({ name: "Ball seen", runs: info.seen, thin: true, color: (r) => (r[2] ? "#b8a436" : null) });
  if (info.tags.length) {
    const clips = (test) => info.tags.filter(test).map((t) => [t.start, t.stop, t.team, titleCase(t.type || t.code)]);
    const setPiece = (t) => /SET PIECES/i.test(t.code);
    const tint = (r) => (r[2] ? COLORS[r[2]] + "8c" : "#48484a");
    rows.push({ name: "Set pieces", runs: clips(setPiece), color: tint, named: true });
    rows.push({ name: "Phases", runs: clips((t) => !setPiece(t)), color: tint, named: true });
  }
  return rows;
}

const niceStep = (seconds) => [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800].find((s) => s >= seconds) || 3600;

function formatTime(seconds) {
  const s = Math.floor(seconds);
  const pad = (n) => String(n).padStart(2, "0");
  const hours = Math.floor(s / 3600);
  return hours ? `${hours}:${pad(Math.floor(s / 60) % 60)}:${pad(s % 60)}` : `${Math.floor(s / 60)}:${pad(s % 60)}`;
}

function timecode(frame) {
  const fps = state.info.fps, pad = (n) => String(n).padStart(2, "0");
  const seconds = Math.floor(frame / fps);
  return [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60, frame % fps].map(pad).join(":");
}

function clip(context, x, y, width, height, radius) {
  context.beginPath();
  context.roundRect(x, y, width, height, Math.min(radius, width / 2));
  context.fill();
}

function drawPlayhead(context, x, height, cap) {
  context.fillStyle = TOKENS.playhead;
  context.fillRect(Math.round(x) - 0.5, cap ? 6 : 0, 1, height);
  if (!cap) return;
  context.beginPath();
  context.moveTo(x - 5.5, 1);
  context.lineTo(x + 5.5, 1);
  context.lineTo(x + 5.5, 6);
  context.lineTo(x, 12);
  context.lineTo(x - 5.5, 6);
  context.closePath();
  context.fill();
}

function shadeOutOfPlay(context, x, from, to, top, height, width) {
  context.fillStyle = "rgba(0,0,0,0.38)";
  let cursor = from;
  for (const [, start, stop] of [...state.info.periods, [0, to, to]]) {
    if (start > cursor) {
      const left = Math.max(HEADER, x(cursor)), right = Math.min(HEADER + width, x(start));
      if (right > left) context.fillRect(left, top, right - left, height);
    }
    cursor = Math.max(cursor, stop);
  }
}

function drawDetail(canvas, [from, to], rows) {
  canvas.style.height = `${RULER + rows.length * TRACK + 5}px`;
  const context = fit(canvas);
  const width = canvas.clientWidth - HEADER, height = canvas.clientHeight;
  const x = (frame) => HEADER + ((frame - from) / (to - from)) * width;
  context.textBaseline = "middle";

  context.fillStyle = TOKENS.header;
  context.fillRect(0, 0, canvas.clientWidth, height);

  rows.forEach((row, i) => {
    const y = RULER + i * TRACK;
    context.fillStyle = i % 2 ? "#1f1f21" : TOKENS.lane;
    context.fillRect(HEADER, y, width, TRACK);
  });
  shadeOutOfPlay(context, x, from, to, RULER, rows.length * TRACK, width);

  context.save();
  context.beginPath();
  context.rect(HEADER, RULER, width, height - RULER);
  context.clip();
  rows.forEach((row, i) => {
    const y = RULER + i * TRACK;
    const inset = row.thin ? 7 : 2, tall = TRACK - 2 * inset;
    for (const run of row.runs) {
      if (run[1] < from || run[0] > to) continue;
      const color = row.color(run);
      if (!color) continue;
      const left = x(run[0]), wide = Math.max(1, x(run[1]) - left - 1);
      context.fillStyle = color;
      clip(context, left, y + inset, wide, tall, row.thin ? 1 : 2.5);
      if (row.named && wide > 44) {
        context.save();
        context.beginPath();
        context.rect(left, y, wide - 4, TRACK);
        context.clip();
        context.font = `500 10.5px ${SANS}`;
        context.textAlign = "left";
        context.fillStyle = "rgba(255,255,255,0.92)";
        context.fillText(run[3], Math.max(left, HEADER) + 5, y + TRACK / 2 + 0.5);
        context.restore();
      }
    }
  });
  context.restore();

  context.fillStyle = TOKENS.header;
  context.fillRect(0, RULER, HEADER, height - RULER);
  context.font = `400 11px ${SANS}`;
  context.textAlign = "left";
  context.fillStyle = TOKENS.text;
  rows.forEach((row, i) => {
    context.fillText(fitText(context, row.name, HEADER - 20), 12, RULER + i * TRACK + TRACK / 2 + 0.5);
  });

  const fps = state.info.fps, step = niceStep((to - from) / fps / (width / 90));
  context.font = `400 10px ${MONO}`;
  context.textAlign = "left";
  for (let n = Math.floor(from / fps / step) * 5; (n * step * fps) / 5 < to; n++) {
    const px = x((n * step * fps) / 5);
    if (px < HEADER) continue;
    const major = n % 5 === 0;
    context.fillStyle = major ? TOKENS.faint : TOKENS.rule;
    context.fillRect(Math.round(px), major ? RULER - 8 : RULER - 4, 1, major ? 8 : 4);
    if (major) {
      context.fillStyle = TOKENS.text;
      context.fillText(formatTime((n * step) / 5), px + 4, 9);
    }
  }
  context.fillStyle = TOKENS.rule;
  context.fillRect(0, RULER - 0.5, canvas.clientWidth, 1);
  context.fillRect(HEADER - 0.5, 0, 1, height);

  const head = x(state.frame);
  if (head >= HEADER && head <= HEADER + width) drawPlayhead(context, head, height, true);
}

function drawOverview(canvas, [from, to], rows) {
  canvas.style.height = `${OVERVIEW}px`;
  const context = fit(canvas);
  const width = canvas.clientWidth - HEADER;
  const x = (frame) => HEADER + ((frame - from) / (to - from)) * width;

  context.fillStyle = "#161618";
  context.fillRect(0, 0, canvas.clientWidth, OVERVIEW);
  context.fillStyle = TOKENS.lane;
  context.fillRect(HEADER, 7, width, 12);
  if (rows[0] !== rows[rows.length - 1] || rows.length === 1) {
    for (const run of rows[0].runs) {
      const color = rows[0].color(run);
      if (!color) continue;
      context.fillStyle = color;
      context.fillRect(x(run[0]), 7, Math.max(0.75, x(run[1]) - x(run[0])), 12);
    }
  }
  const seen = rows.find((row) => row.thin);
  context.fillStyle = "#8f7f2c";
  for (const run of seen.runs) {
    if (run[2]) context.fillRect(x(run[0]), 21, Math.max(0.5, x(run[1]) - x(run[0])), 2);
  }
  shadeOutOfPlay(context, x, from, to, 7, 16, width);

  context.font = `400 11px ${SANS}`;
  context.textAlign = "left";
  context.textBaseline = "middle";
  context.fillStyle = TOKENS.faint;
  context.fillText("Full match", 12, OVERVIEW / 2);

  const left = Math.max(HEADER, x(state.view[0])), right = Math.min(HEADER + width, x(state.view[1]));
  if (right > left) {
    context.fillStyle = "rgba(255,255,255,0.1)";
    context.fillRect(left, 3, right - left, OVERVIEW - 6);
    context.strokeStyle = "rgba(255,255,255,0.75)";
    context.lineWidth = 1;
    context.beginPath();
    context.roundRect(left + 0.5, 3.5, Math.max(2, right - left - 1), OVERVIEW - 7, 2);
    context.stroke();
  }
  drawPlayhead(context, x(state.frame), OVERVIEW, false);
}

function detailRange() {
  const [from, to] = state.view;
  const moved = state.frame !== state.viewFrame;  // a pan may leave the playhead out of view
  state.viewFrame = state.frame;
  if (moved && !state.scrubbing && (state.frame < from || state.frame >= to)) {
    state.view = [state.frame - state.span / 10, state.frame + (state.span * 9) / 10];
  }
  return state.view;
}

/* ---------- information ---------- */

const escapeHtml = (text) => String(text).replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);
const rowHtml = (name, value, muted) => `<tr><th>${name}</th><td${muted ? ' class="muted"' : ""}>${value}</td></tr>`;
const dot = (side) => `<span class="dot ${side}"></span>`;
const team = (t) => (t === "H" ? state.info.home : t === "A" ? state.info.away : "Dead ball");
const panel = (title, body) => `<section class="panel"><h2>${title}</h2>${body}</section>`;
const shortTrack = (track) => String(track).replace(/^DFL-OBJ-0*/, "").slice(-5);

function matchClock(frame) {
  const period = state.info.periods.find((p) => frame >= p[1] && frame < p[2]);
  if (!period) return "Not in play";
  const clock = formatTime((frame - period[1]) / state.info.fps);
  return `${period[0] === 1 ? "1st" : "2nd"} half ${clock}`;
}

function heroHtml() {
  const { info, frame } = state, row = info.rows[0];
  const run = row ? row.runs[runAt(row.runs, frame)] : null;
  if (!run) {
    return `<section class="panel hero"><div class="team"><span class="dot"></span><span>No label</span></div>
      <div class="detail"><span>${row ? "Outside the labelled window" : "This match has no labels"}</span></div></section>`;
  }
  const [start, stop, label] = run;
  const possessions = possessionRuns(), index = runAt(possessions, frame);
  const elapsed = (frame - start) / info.fps, total = (stop - start) / info.fps;
  return `<section class="panel hero">
    <div class="team">${dot(label)}<span>${escapeHtml(team(label))}</span></div>
    <div class="detail"><span>${index >= 0 ? `Possession ${index + 1} of ${possessions.length}` : "Stoppage"}</span>
      <span>${elapsed.toFixed(1)} / ${total.toFixed(1)} s</span></div>
    <div class="progress"><div class="${label}" style="width:${(100 * (frame - start)) / (stop - start)}%"></div></div>
  </section>`;
}

function labelsHtml() {
  const rows = state.info.rows;
  if (rows.length < 2) return "";
  const body = rows.map((row) => {
    const run = row.runs[runAt(row.runs, state.frame)];
    return rowHtml(escapeHtml(rowName(row.name)), run ? `${dot(run[2])}${escapeHtml(team(run[2]))}` : "—", !run);
  });
  return panel("Labels", `<table>${body.join("")}</table>`);
}

function ballHtml(data) {
  const [length, width] = state.info.pitch;
  if (!data.ball) return panel("Ball", '<p class="quiet">Not seen in this frame</p>');
  const near = nearestPlayer(data);
  return panel("Ball", `<table>
    ${rowHtml("Position", `${(data.ball[0] * length).toFixed(1)}, ${(data.ball[1] * width).toFixed(1)} m`)}
    ${rowHtml("Speed", data.ball[2] === null ? "—" : `${data.ball[2].toFixed(1)} m/s`, data.ball[2] === null)}
    ${near ? rowHtml("Nearest", `${dot(near.player[0])}<span class="track">${escapeHtml(shortTrack(near.player[1]))}</span> · ${near.distance.toFixed(1)} m`) : ""}
  </table>`);
}

function teamStats(players, side) {
  const [length, width] = state.info.pitch;
  const group = players.filter((p) => p[0] === side);
  if (!group.length) return { count: 0, speed: "—", width: "—", depth: "—" };
  const speeds = group.map((p) => p[4]).filter((v) => v !== null);
  const xs = group.map((p) => p[2] * length), ys = group.map((p) => p[3] * width);
  return {
    count: group.length,
    speed: speeds.length ? (speeds.reduce((a, b) => a + b, 0) / speeds.length).toFixed(1) : "—",
    depth: (Math.max(...xs) - Math.min(...xs)).toFixed(0),
    width: (Math.max(...ys) - Math.min(...ys)).toFixed(0),
  };
}

function teamsHtml(data) {
  if (!data.players || !data.players.length) return panel("Teams", '<p class="quiet">No players tracked</p>');
  const home = teamStats(data.players, "H"), away = teamStats(data.players, "A");
  const line = (name, key) => `<tr><th>${name}</th><td>${home[key]}</td><td>${away[key]}</td></tr>`;
  return panel("Teams", `<table class="teams"><thead><tr><th></th><th>${dot("H")}</th><th>${dot("A")}</th></tr></thead><tbody>
    ${line("Players", "count")}${line("Speed, m/s", "speed")}${line("Depth, m", "depth")}${line("Width, m", "width")}
  </tbody></table>`);
}

function drawInfo(data) {
  const { info, frame } = state, [length, width] = info.pitch;
  const parts = [heroHtml(), labelsHtml()];
  if (data) {
    parts.push(ballHtml(data), teamsHtml(data));
    const followed = data.players?.find((p) => p[1] === state.followed);
    if (followed) {
      parts.push(panel("Following", `<table>
        ${rowHtml("Track", `${dot(followed[0])}<span class="track">${escapeHtml(shortTrack(followed[1]))}</span>`)}
        ${rowHtml("Position", `${(followed[2] * length).toFixed(1)}, ${(followed[3] * width).toFixed(1)} m`)}
        ${rowHtml("Speed", followed[4] === null ? "—" : `${followed[4].toFixed(1)} m/s`, followed[4] === null)}
      </table>`));
    }
    if (info.homography) {
      parts.push(panel("Camera", `<table>${rowHtml("Homography", data.homography ? "Valid" : "None", !data.homography)}</table>`));
    }
  } else {
    parts.push(panel("Tracking", '<p class="quiet">Loading…</p>'));
  }
  const active = info.tags.filter((t) => frame >= t.start && frame < t.stop);
  if (active.length) {
    const body = active.map((t) => rowHtml(`${dot(t.team)}${escapeHtml(titleCase(t.code))}`, escapeHtml(t.type || "")));
    parts.push(panel("Smart Tagging", `<table>${body.join("")}</table>`));
  }
  $("info").innerHTML = parts.join("");
}

/* ---------- frame loop ---------- */

let scheduled = false;
function draw() {
  if (!state.info) return;
  const size = state.info.chunk, index = Math.floor(state.frame / size);
  loadChunk(index);  // here, because hidden pages pause animation frames
  loadChunk(index + 1);
  if (state.frame % size < TRAIL_FRAMES) loadChunk(index - 1);
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(() => {
    scheduled = false;
    if (!state.info) return;
    const data = dataAt(state.frame);
    const rows = timelineRows();
    drawMain(data);
    drawMini(data);
    drawInfo(data);
    drawDetail($("detail"), detailRange(), rows);
    drawOverview($("overview"), state.range, rows);
    if (document.activeElement !== $("goto")) $("goto").value = state.frame;
    $("timecode").textContent = timecode(state.frame);
    $("period").textContent = matchClock(state.frame);
  });
}

/* ---------- loading ---------- */

async function openMatch(key, frame) {
  setStatus("Loading…");
  if (state.playing) setPlaying(false);
  state.info = null;
  state.chunks.clear();
  state.pending.clear();
  const info = await json(`/api/info/${key}`);
  state.info = info;
  state.followed = null;
  state.span = info.fps * 60;
  state.view = [0, 0];
  state.viewFrame = null;
  state.range = [info.first, info.last];
  state.showVideo = info.video;
  $("match").value = key;
  $("title").innerHTML = `${dot("H")}<span>${escapeHtml(info.home)}</span><span class="versus">vs</span>${dot("A")}<span>${escapeHtml(info.away)}</span>`;
  $("view").hidden = !info.video;
  showView(info.video);

  if (info.video) {
    video.src = `/video/${key}`;
    video.addEventListener("loadedmetadata", () => {
      state.range = [0, Math.floor(video.duration * info.fps) - 1];
      seek(frame ?? info.first);
    }, { once: true });
    video.playbackRate = state.rate;
  } else {
    video.removeAttribute("src");
    video.load();
    seek(frame ?? info.first);
  }
  setStatus(info.video ? "" : "No video for this match");
  draw();
}

function setStatus(text, error) {
  $("status").textContent = text;
  $("status").classList.toggle("error", Boolean(error));
}

/* ---------- input ---------- */

function step(frames) {
  if (state.playing) setPlaying(false);
  seek(state.frame + frames);
  saveLocation();
}

function jumpPossession(direction) {
  const runs = possessionRuns();
  if (!runs.length) return;
  const later = runs.find((r) => r[0] > state.frame);
  let target;
  if (direction > 0) target = later;
  else {
    const earlier = runs.filter((r) => r[0] < state.frame - state.info.fps);
    target = earlier[earlier.length - 1];
  }
  if (target) step(target[0] - state.frame);
}

function setRate(direction) {
  const select = $("rate"), index = select.selectedIndex + direction;
  if (index >= 0 && index < select.options.length) {
    select.selectedIndex = index;
    applyRate();
  }
}

function applyRate() {
  state.rate = Number($("rate").value);
  video.playbackRate = state.rate;
  if (!hasVideo() && state.playing) { exact = state.frame; lastTick = performance.now(); }
}

function showView(onVideo) {
  state.showVideo = onVideo;
  video.hidden = !onVideo;
  $("viewVideo").setAttribute("aria-pressed", onVideo);
  $("viewPitch").setAttribute("aria-pressed", !onVideo);
  draw();
}

function swapView() {
  if (hasVideo()) showView(!state.showVideo);
}

document.addEventListener("keydown", (event) => {
  if (!state.info || $("shortcuts").open) return;
  if (event.target.tagName === "INPUT" && event.target.type !== "checkbox") return;
  if (event.target.tagName === "SELECT") return;
  const fps = state.info.fps;
  const jump = event.altKey ? 10 * fps : event.shiftKey ? fps : 1;
  const actions = {
    " ": () => setPlaying(!state.playing),
    ArrowLeft: () => step(-jump), ArrowRight: () => step(jump),
    "[": () => jumpPossession(-1), "]": () => jumpPossession(1),
    "-": () => setRate(-1), "=": () => setRate(1), v: swapView, "?": () => $("shortcuts").showModal(),
  };
  if (actions[event.key]) {
    event.preventDefault();
    actions[event.key]();
  }
});

function bindTimeline(canvas, rangeOf) {
  const offset = (event) => event.clientX - canvas.getBoundingClientRect().left;
  const frameAt = (event) => {
    const [from, to] = rangeOf();
    const share = (offset(event) - HEADER) / (canvas.clientWidth - HEADER);
    return from + Math.max(0, Math.min(1, share)) * (to - from);
  };
  const release = () => {
    if (!state.scrubbing) return;
    state.scrubbing = false;
    seek(state.frame);  // exact, even if the drag used fast seeks
    saveLocation();
  };

  canvas.addEventListener("pointerdown", (event) => {
    if (!state.info || offset(event) < HEADER) return;
    canvas.setPointerCapture(event.pointerId);
    if (state.playing) setPlaying(false);
    const frame = frameAt(event);  // first, so the window can settle before it freezes
    state.scrubbing = true;
    seek(frame);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!state.info) return;
    if (state.scrubbing) seek(frameAt(event));
    const frame = Math.round(frameAt(event)), tip = $("tip");
    const lines = [`${timecode(Math.max(0, frame))}  ·  ${matchClock(frame)}`];
    for (const row of state.info.rows) {
      const i = runAt(row.runs, frame);
      if (i >= 0) lines.push(`${row.name}: ${team(row.runs[i][2])}`);
    }
    tip.textContent = lines.join("\n");
    tip.hidden = false;
    tip.style.left = `${Math.min(offset(event), canvas.clientWidth - 200)}px`;
  });
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("pointerleave", () => { $("tip").hidden = true; });
}

bindTimeline($("detail"), detailRange);
bindTimeline($("overview"), () => state.range);

$("detail").addEventListener("wheel", (event) => {
  event.preventDefault();
  if (!state.info) return;
  const canvas = $("detail"), width = canvas.clientWidth - HEADER;
  const unit = event.deltaMode === 1 ? 16 : 1;
  const across = (event.shiftKey ? event.deltaX || event.deltaY : event.deltaX) * unit;
  const [from, to] = state.view;

  if (Math.abs(across) > Math.abs(event.deltaY * unit) || event.shiftKey) {
    const moved = (across / width) * (to - from);
    const low = state.range[0] - state.span / 2, high = state.range[1] - state.span / 2;
    const start = Math.max(low, Math.min(high, from + moved));
    state.view = [start, start + state.span];
  } else {
    const box = canvas.getBoundingClientRect();
    const share = Math.max(0, Math.min(1, (event.clientX - box.left - HEADER) / width));
    const anchor = from + share * (to - from);  // the frame under the cursor stays put
    const zoom = Math.exp(event.deltaY * unit * 0.002);
    state.span = Math.max(state.info.fps * 5, Math.min(state.info.fps * 900, state.span * zoom));
    state.view = [anchor - share * state.span, anchor + (1 - share) * state.span];
  }
  draw();
}, { passive: false });

$("overlay").addEventListener("click", (event) => {
  const data = state.info && dataAt(state.frame);
  if (!data || !data.players || !projectMain) return;
  const box = event.target.getBoundingClientRect();
  let best = null, bestDistance = 18;
  for (const player of data.players) {
    const [x, y] = projectMain(player[2], player[3]);
    const d = Math.hypot(x - (event.clientX - box.left), y - (event.clientY - box.top));
    if (d < bestDistance) { best = player; bestDistance = d; }
  }
  state.followed = best ? best[1] : null;
  draw();
});

const buttons = {
  play: () => setPlaying(!state.playing),
  stepBack: () => step(-1),
  stepForward: () => step(1),
  previousPossession: () => jumpPossession(-1),
  nextPossession: () => jumpPossession(1),
  viewVideo: () => { if (hasVideo()) showView(true); },
  viewPitch: () => showView(false),
};
for (const [id, action] of Object.entries(buttons)) {
  $(id).addEventListener("click", (event) => { if (state.info) action(); event.currentTarget.blur(); });
}
$("help").addEventListener("click", () => $("shortcuts").showModal());

$("rate").addEventListener("change", () => { applyRate(); $("rate").blur(); });
$("goto").addEventListener("change", () => { step(Number($("goto").value) - state.frame); $("goto").blur(); });
$("match").addEventListener("change", () => { openMatch($("match").value); $("match").blur(); });
video.addEventListener("error", () => {
  if (video.getAttribute("src")) setStatus("This browser cannot play the video file", true);
});
video.addEventListener("waiting", () => setStatus("Buffering"));
video.addEventListener("playing", () => setStatus(""));
video.addEventListener("ended", () => setPlaying(false));
new ResizeObserver(draw).observe($("stage"));

for (const [key, [name, on]] of Object.entries(LAYERS)) {
  const label = document.createElement("label"), box = document.createElement("input");
  box.type = "checkbox";
  box.setAttribute("role", "switch");
  box.checked = on;
  box.addEventListener("change", () => { state.layers[key] = box.checked; box.blur(); draw(); });
  label.append(name, box);
  $("layers").append(label);
}

async function start() {
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(onVideoFrame);
  else {
    const poll = () => {
      if (hasVideo() && state.playing) { state.frame = Math.floor(video.currentTime * state.info.fps); draw(); }
      requestAnimationFrame(poll);
    };
    poll();
  }

  const matches = await json("/api/matches");
  if (!matches.length) {
    setStatus("No matches yet. Run python -m scripts.prepare_data");
    return;
  }
  for (const match of matches) {
    const option = document.createElement("option");
    option.value = match.key;
    option.textContent = `${match.home} vs ${match.away}`;
    $("match").append(option);
  }
  const [key, frame] = location.hash.slice(1).split("@");
  const known = matches.some((m) => m.key === key);
  openMatch(known ? key : matches[0].key, frame ? Number(frame) : undefined);
}

start();
