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
  const res = await sendBg({ type: 'get-state' }).catch(() => null);
  const state = res?.state;
  if (!state) return;
  renderDebugCounters(state);
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
