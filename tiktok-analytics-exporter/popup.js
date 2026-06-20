const VIDEO_CSV_COLUMNS = [
  'video_id','post_date','post_time','caption','duration_ms','views','likes','comments','shares',
  'ECR','avg_watch_time_s','NAWP','watched_full_pct',
  'traffic_foryou_pct','traffic_follow_pct','traffic_profile_pct','traffic_search_pct',
  'new_followers','creator_uid','creator_handle','follower_count','account_created_date','data_quality'
];
const FOLLOWER_CSV_COLUMNS = ['date','follower_count','daily_net','creator_handle','creator_uid','data_quality'];

const STUDIO_CONTENT_URL   = 'https://www.tiktok.com/tiktokstudio/content';
const STUDIO_FOLLOWERS_URL = 'https://www.tiktok.com/tiktokstudio/analytics/followers?dateRange=%7B%22type%22%3A%22fixed%22%2C%22pastDay%22%3A365%7D';

let activeTabId = null;
let pollHandle = null;
let footerClickCount = 0;
let footerClickTimer = null;

document.addEventListener('DOMContentLoaded', () => {
  init().catch((err) => console.error('[popup] init failed', err));
});

async function init() {
  const tab = await getActiveTab();
  activeTabId = tab?.id ?? null;
  wireTabs();
  wireDebugReveal();
  wireModules();                 // <-- ADD THIS LINE
  setDefaultDates();
  await refresh();
  pollHandle = setInterval(refresh, 750);
  window.addEventListener('unload', () => clearInterval(pollHandle));
}

function wireTabs() {
  for (const btn of document.querySelectorAll('.tab')) {
    btn.addEventListener('click', () => {
      const name = btn.dataset.tab;
      for (const t of document.querySelectorAll('.tab')) t.classList.toggle('on', t.dataset.tab === name);
      for (const p of document.querySelectorAll('.panel')) p.classList.toggle('on', p.id === `p-${name}`);
    });
  }
}

function wireDebugReveal() {
  const foot = document.getElementById('ver-foot');
  if (!foot) return;
  foot.addEventListener('click', () => {
    footerClickCount += 1;
    clearTimeout(footerClickTimer);
    footerClickTimer = setTimeout(() => (footerClickCount = 0), 600);
    if (footerClickCount >= 3) {
      footerClickCount = 0;
      document.getElementById('tab-debug').classList.remove('hidden');
    }
  });
}

function setDefaultDates() {
  const today = new Date();
  const ninety = new Date();
  ninety.setDate(today.getDate() - 90);
  const startEl = document.getElementById('m1-start');
  const endEl   = document.getElementById('m1-end');
  if (startEl) startEl.value = isoDate(ninety);
  if (endEl)   endEl.value   = isoDate(today);
}

function isoDate(d) { return d.toISOString().slice(0,10); }

async function refresh() {
  await checkPageStates();
  const res = await sendBg({ type: 'get-state' }).catch(() => null);
  const state = res?.state;
  if (!state) return;
  renderModule1(state.videoStep);
  renderModule2(state.followerStep);
  renderFooter(state);
  renderDebugCounters(state);
  renderDebugUrls(state);
}

function renderDebugCounters(state) {
  const c = state.interceptCounts || {};
  setText('dbg-vl',   c.videoList ?? 0);
  setText('dbg-ins',  c.insight   ?? 0);
  setText('dbg-prof', c.profile   ?? 0);
}

function setText(id, v) { const el = document.getElementById(id); if (el) el.textContent = String(v); }

async function getActiveTab() {
  return new Promise((resolve) => chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => resolve(tabs[0])));
}
function sendBg(msg) {
  return new Promise((resolve) => chrome.runtime.sendMessage(msg, (response) => {
    void chrome.runtime.lastError;
    resolve(response);
  }));
}

let m1OnPage = false;
let m2OnPage = false;

function wireModules() {
  document.getElementById('m1-open').addEventListener('click', () => openTab(STUDIO_CONTENT_URL));
  document.getElementById('m1-extract').addEventListener('click', startVideoExtract);
  document.getElementById('m1-save').addEventListener('click', saveVideoCSV);
  document.getElementById('m1-cancel').addEventListener('click', () => sendBg({ type: 'cancel-video-export' }));
  for (const btn of document.querySelectorAll('#m1-ready .presets button')) {
    btn.addEventListener('click', () => applyPreset(btn));
  }
  document.getElementById('m2-open').addEventListener('click', () => openTab(STUDIO_FOLLOWERS_URL));
  document.getElementById('m2-extract').addEventListener('click', startFollowerExtract);
  document.getElementById('m2-save').addEventListener('click', saveFollowerCSV);
}

async function openTab(url) {
  if (activeTabId) {
    try { await chrome.tabs.update(activeTabId, { url, active: true }); return; }
    catch (_e) { /* fall through to create */ }
  }
  await chrome.tabs.create({ url });
}

function applyPreset(btn) {
  for (const b of document.querySelectorAll('#m1-ready .presets button')) b.classList.remove('on');
  btn.classList.add('on');
  const days = btn.dataset.days;
  const today = new Date();
  document.getElementById('m1-end').value = isoDate(today);
  if (!days) { document.getElementById('m1-start').value = ''; return; }
  const start = new Date();
  start.setDate(today.getDate() - Number(days));
  document.getElementById('m1-start').value = isoDate(start);
}

async function startVideoExtract() {
  hideErr('m1-error');
  if (!activeTabId) { showErr('m1-error', 'Open TikTok Studio first.'); return; }
  const dateRange = {
    start: document.getElementById('m1-start').value || null,
    end:   document.getElementById('m1-end').value   || null
  };
  await sendBg({ type: 'reset-video-step' });
  const res = await sendBg({ type: 'start-video-export', dateRange, tabId: activeTabId });
  if (!res?.ok) showErr('m1-error', res?.error || 'Failed to start');
  await refresh();
}

async function saveVideoCSV() {
  const res = await sendBg({ type: 'get-state' });
  const rows = res?.state?.videoStep?.rows || [];
  if (!rows.length) return;
  const handle = res.state.profile?.creator_handle || 'unknown';
  const today = isoDate(new Date());
  const filename = `tiktok_videos_${sanitize(handle)}_${today}.csv`;
  await downloadCSV(filename, buildCSV(rows, VIDEO_CSV_COLUMNS));
}

async function checkPageStates() {
  if (!activeTabId) { m1OnPage = false; m2OnPage = false; return; }
  const [a, b] = await Promise.all([
    chrome.tabs.sendMessage(activeTabId, { type: 'is-studio-page' }).catch(() => null),
    chrome.tabs.sendMessage(activeTabId, { type: 'is-followers-page' }).catch(() => null)
  ]);
  m1OnPage = !!a?.isStudio;
  m2OnPage = !!b?.isFollowers;
}

function renderModule1(vs) {
  const phase = vs.phase;
  showOnly('mod1', phaseToPaneM1(phase, m1OnPage));
  setPill('pill1', phase, m1OnPage, 1);
  setDot('dot1', phase === 'done');
  if (phase === 'fetching-insights' || phase === 'collecting' || phase === 'fetching-profile') {
    const total = vs.progress?.total || 0;
    const cur = vs.progress?.current || 0;
    const pct = total > 0 ? Math.min(100, (cur/total)*100) : phaseFallbackPct(phase);
    setBar('bar1', pct);
    setText('meta1', vs.progress?.message || `${cur}/${total}`);
  }
  if (phase === 'done') {
    setText('m1-summary', `${vs.rows.length} videos${vs.skipped.length ? ` · ${vs.skipped.length} skipped` : ''}`);
    renderSkipped(vs.skipped || []);
  }
  if (phase === 'error') showErr('m1-error', vs.error || 'Unknown error');
  else hideErr('m1-error');
}

function phaseToPaneM1(phase, onPage) {
  if (phase === 'idle' || phase === 'cancelled') return onPage ? 'ready' : 'idle';
  if (phase === 'collecting' || phase === 'fetching-insights' || phase === 'fetching-profile') return 'run';
  if (phase === 'done')  return 'done';
  if (phase === 'error') return onPage ? 'ready' : 'idle';
  return 'idle';
}

function phaseFallbackPct(phase) {
  if (phase === 'collecting') return 10;
  if (phase === 'fetching-profile') return 95;
  return 50;
}

function showOnly(modId, paneName) {
  for (const p of ['idle','ready','run','done']) {
    const el = document.getElementById(`${modId === 'mod1' ? 'm1' : 'm2'}-${p}`);
    if (el) el.classList.toggle('hide', p !== paneName);
  }
}

function setPill(id, phase, onPage, n) {
  const el = document.getElementById(id);
  if (!el) return;
  if (phase === 'done') { el.className = 'pill done'; el.textContent = 'Done'; return; }
  if (phase === 'collecting' || phase === 'fetching-insights' || phase === 'fetching-profile' || phase === 'fetching') {
    el.className = 'pill ready'; el.textContent = 'Running'; return;
  }
  if (onPage) { el.className = 'pill ready'; el.textContent = 'On page'; return; }
  el.className = 'pill todo'; el.textContent = `Step ${n}`;
}

function setDot(id, done) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle('done', done);
}

function setBar(id, pct) {
  const el = document.getElementById(id);
  if (el) el.style.width = `${pct}%`;
}

function renderSkipped(list) {
  setText('m1-skipped-n', list.length);
  const ul = document.getElementById('m1-skipped');
  if (!ul) return;
  ul.innerHTML = '';
  for (const s of list) {
    const li = document.createElement('li');
    li.textContent = `${s.aweme_id} — ${s.reason}`;
    ul.appendChild(li);
  }
}

function renderFooter(state) {
  const done1 = state.videoStep.phase === 'done' ? 1 : 0;
  const done2 = state.followerStep.phase === 'done' ? 1 : 0;
  setText('cnt', done1 + done2);
  document.getElementById('finish').classList.toggle('show', done1 + done2 === 2);
}

function buildCSV(rows, columns) {
  const lines = [columns.join(',')];
  for (const row of rows) lines.push(columns.map((c) => escapeCSV(row[c])).join(','));
  return lines.join('\n');
}
function escapeCSV(v) {
  if (v == null) return '';
  const s = String(v);
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}
function sanitize(s) { return String(s).replace(/[^a-z0-9_-]/gi, '_').slice(0, 64); }
async function downloadCSV(filename, csv) {
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  await chrome.downloads.download({ url, filename, saveAs: true });
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
function showErr(id, msg) { const el = document.getElementById(id); if (el) { el.textContent = msg; el.classList.remove('hide'); } }
function hideErr(id) { const el = document.getElementById(id); if (el) { el.textContent = ''; el.classList.add('hide'); } }

// Placeholder for Step 2 — implemented in Task 14.
function renderModule2() {}
function startFollowerExtract() {}
function saveFollowerCSV() {}
function renderDebugUrls() {}
